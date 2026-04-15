#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
import torch
import torch.nn as nn
import numpy as np
from agent_ppo.conf.conf import Config

def make_fc_layer(in_features, out_features, init_gain=1.0):
    """创建正交初始化的线性层。"""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=init_gain)
    nn.init.zeros_(fc.bias.data)
    return fc

class EntityAttention(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = make_fc_layer(embed_dim, embed_dim)
        self.key_proj = make_fc_layer(embed_dim, embed_dim)
        self.value_proj = make_fc_layer(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, query, entities):
        E = torch.stack(entities, dim=1)
        Q = self.query_proj(query).unsqueeze(1)
        K = self.key_proj(E)
        V = self.value_proj(E)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn, V)
        return output.squeeze(1)

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_multichannel_v3"
        self.device = device

        embed_dim = 64
        status_dim_out = 32
        # 【修复点 1】必须定义 map_dim_out，因为你在 cnn 层最后用到了它
        map_dim_out = 128  
        mid_dim = 512  # 扩大 Backbone 宽度以匹配更多特征

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

        self.attention = EntityAttention(embed_dim)

        # ============ CNN for Multi-Channel Map (6 × 21 × 21) ============
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, map_dim_out),
            nn.ReLU()
        )

        # ============ Backbone ============
        # 【修复点 2】维度计算逻辑对齐：
        # 根据你下方的拼接：hero(64) + monster(64) + treasure(64) + escape(64) + fused(64) + map(128) + status(32) + move_pass(8)
        # 总维度应该是：64*5 + 128 + 32 + 8 = 488
        combined_dim = embed_dim * 5 + map_dim_out + status_dim_out + Config.MOVE_PASS_DIM
        
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

        # 【修复点 3】初始化 Gain 设置（PPO 惯例）
        self.actor_head = make_fc_layer(mid_dim, Config.ACTION_NUM, init_gain=0.01)
        self.critic_head = make_fc_layer(mid_dim, Config.VALUE_NUM, init_gain=1.0)

    def forward(self, obs, inference=False):
        bs = obs.size(0)
        scalar_len = Config.SCALAR_LEN
        scalars = obs[:, :scalar_len]
        maps = obs[:, scalar_len:].reshape(bs, 6, Config.MAP_SIDE, Config.MAP_SIDE)

        # 编码
        hero_emb = self.hero_mlp(scalars[:, Config.HERO_START:Config.HERO_START + Config.HERO_DIM])
        monster_emb = self.monster_mlp(scalars[:, Config.MONSTER_START:Config.MONSTER_START + Config.MONSTER_DIM])
        treasure_emb = self.treasure_mlp(scalars[:, Config.TREASURE_START:Config.TREASURE_START + Config.TREASURE_DIM])
        escape_emb = self.escape_mlp(scalars[:, Config.MAP_ESCAPE_START:Config.MAP_ESCAPE_START + Config.MAP_ESCAPE_DIM])
        status_emb = self.status_mlp(scalars[:, Config.STATUS_START:Config.STATUS_START + Config.STATUS_DIM])
        move_pass_in = scalars[:, Config.MOVE_PASS_START:Config.MOVE_PASS_START + Config.MOVE_PASS_DIM]

        # 注意力提取交叉特征
        fused_emb = self.attention(hero_emb, [monster_emb, treasure_emb, escape_emb])

        # 地图特征
        map_emb = self.cnn(maps)

        # 【修复点 4】拼接顺序必须和上面 __init__ 里的 combined_dim 严格对应
        # 建议直接把原始特征全部拼接，防止信息丢失
        combined = torch.cat([
            hero_emb, monster_emb, treasure_emb, escape_emb, 
            fused_emb, map_emb, status_emb, move_pass_in
        ], dim=1)
        
        hidden = self.backbone(combined)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self): self.train()
    def set_eval_mode(self): self.eval()