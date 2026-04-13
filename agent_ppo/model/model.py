#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model with entity attention for Gorge Chase PPO.
峡谷追猎 PPO 实体注意力神经网络模型。

Architecture:
  标量输入拆分为 5 组，分别编码后用注意力融合:
    Hero(10D)     → MLP → hero_emb(64D)      [Query]
    Monster(20D)  → MLP → monster_emb(64D)    [Key/Value]
    Treasure(14D) → MLP → treasure_emb(64D)   [Key/Value]
    MapEscape(16D)→ MLP → escape_emb(64D)     [Key/Value]
    Status(8D)    → MLP → status_emb(32D)     [直接拼接]

  地图(21×21) → CNN → map_emb(64D)

  Combined = [hero_emb, attention_fused, map_emb, status_emb]
           → Backbone MLP → Actor(16D) + Critic(1D)
"""

import torch
import torch.nn as nn
import numpy as np

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class EntityAttention(nn.Module):
    """Lightweight scaled dot-product attention for entity fusion.

    轻量级注意力机制：用 hero embedding 作为 Query，
    怪物/宝箱/逃生 embedding 作为 Key/Value，
    让模型自动决定当前更该关注哪类实体。
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = make_fc_layer(embed_dim, embed_dim)
        self.key_proj = make_fc_layer(embed_dim, embed_dim)
        self.value_proj = make_fc_layer(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, query, entities):
        """
        Args:
            query: (batch, embed_dim) - hero embedding as query
            entities: list of (batch, embed_dim) tensors - entity embeddings

        Returns:
            (batch, embed_dim) - attention-weighted fusion of entities
        """
        # Stack entities: (batch, num_entities, embed_dim)
        E = torch.stack(entities, dim=1)

        Q = self.query_proj(query).unsqueeze(1)   # (batch, 1, embed_dim)
        K = self.key_proj(E)                       # (batch, num_entities, embed_dim)
        V = self.value_proj(E)                     # (batch, num_entities, embed_dim)

        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch, 1, N)
        attn = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn, V)               # (batch, 1, embed_dim)

        return output.squeeze(1)                   # (batch, embed_dim)


class Model(nn.Module):
    """Structured model with entity attention + CNN for map."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_attention_v2"
        self.device = device

        embed_dim = 64
        status_dim_out = 32
        mid_dim = 256  # 扩展 Backbone 宽度以匹配更多特征

        # ============ Entity Encoders ============
        self.hero_mlp = nn.Sequential(
            make_fc_layer(Config.HERO_DIM, embed_dim),
            nn.ReLU(),
            make_fc_layer(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.monster_mlp = nn.Sequential(
            make_fc_layer(Config.MONSTER_DIM, embed_dim),
            nn.ReLU(),
            make_fc_layer(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.treasure_mlp = nn.Sequential(
            make_fc_layer(Config.TREASURE_DIM, embed_dim),
            nn.ReLU(),
            make_fc_layer(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.escape_mlp = nn.Sequential(
            make_fc_layer(Config.MAP_ESCAPE_DIM, embed_dim),
            nn.ReLU(),
            make_fc_layer(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.status_mlp = nn.Sequential(
            make_fc_layer(Config.STATUS_DIM, status_dim_out),
            nn.ReLU()
        )

        # ============ Attention Fusion ============
        self.attention = EntityAttention(embed_dim)

        # ============ CNN for Local Map (21×21) ============
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 21→10
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10→5
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, embed_dim),
            nn.ReLU()
        )

        # ============ Backbone ============
        # hero(64) + fused(64) + map(64) + status(32) + move_pass(8) = 232
        combined_dim = embed_dim * 3 + status_dim_out + Config.MOVE_PASS_DIM
        self.backbone = nn.Sequential(
            make_fc_layer(combined_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            make_fc_layer(mid_dim, mid_dim),
            nn.ReLU()
        )

        # ============ Output Heads ============
        self.actor_head = make_fc_layer(mid_dim, Config.ACTION_NUM)
        self.critic_head = make_fc_layer(mid_dim, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        scalar_len = Config.SCALAR_LEN
        scalars = obs[:, :scalar_len]
        maps = obs[:, scalar_len:].view(-1, 1, Config.MAP_SIDE, Config.MAP_SIDE)

        # Split scalar features into entity groups
        hero_in = scalars[:, Config.HERO_START:Config.HERO_START + Config.HERO_DIM]
        monster_in = scalars[:, Config.MONSTER_START:Config.MONSTER_START + Config.MONSTER_DIM]
        treasure_in = scalars[:, Config.TREASURE_START:Config.TREASURE_START + Config.TREASURE_DIM]
        escape_in = scalars[:, Config.MAP_ESCAPE_START:Config.MAP_ESCAPE_START + Config.MAP_ESCAPE_DIM]
        status_in = scalars[:, Config.STATUS_START:Config.STATUS_START + Config.STATUS_DIM]
        move_pass_in = scalars[:, Config.MOVE_PASS_START:Config.MOVE_PASS_START + Config.MOVE_PASS_DIM]

        # Encode entities
        hero_emb = self.hero_mlp(hero_in)           # (batch, 64)
        monster_emb = self.monster_mlp(monster_in)   # (batch, 64)
        treasure_emb = self.treasure_mlp(treasure_in) # (batch, 64)
        escape_emb = self.escape_mlp(escape_in)     # (batch, 64)
        status_emb = self.status_mlp(status_in)     # (batch, 32)

        # Attention: hero queries entities
        fused_emb = self.attention(
            hero_emb, [monster_emb, treasure_emb, escape_emb]
        )

        # CNN for map
        map_emb = self.cnn(maps)                     # (batch, 64)

        # Combine all and pass through backbone
        # move_pass_in is 0/1 binary, directly concatenated (no MLP needed)
        combined = torch.cat([hero_emb, fused_emb, map_emb, status_emb, move_pass_in], dim=1)
        hidden = self.backbone(combined)

        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
