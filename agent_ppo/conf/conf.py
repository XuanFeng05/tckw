#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # ======================== Feature Dimensions ========================
    # Hero self features: pos(2) + flash(2) + buff(2) + score(1) + progress(1) + stuck(1) + loop(1) + speed(1) + expiring(1)
    HERO_DIM = 12
    HERO_START = 0

    # Monster features: 2 monsters × 8D + global(4)
    MONSTER_DIM = 20
    MONSTER_START = HERO_START + HERO_DIM  # 12

    # Treasure/Buff features: nearest_t(5) + 2nd_t(4) + count(1) + nearest_b(4)
    TREASURE_DIM = 14
    TREASURE_START = MONSTER_START + MONSTER_DIM  # 32

    # Map/Escape features: rays(8) + derived(8)
    MAP_ESCAPE_DIM = 16
    MAP_ESCAPE_START = TREASURE_START + TREASURE_DIM  # 46

    # Status/Legal features
    STATUS_DIM = 8
    STATUS_START = MAP_ESCAPE_START + MAP_ESCAPE_DIM  # 62

    # Move passability features: 8 directions (can this direction actually move?)
    # 8方向即时可通行性特征（考虑斜向移动规则）
    MOVE_PASS_DIM = 8
    MOVE_PASS_START = STATUS_START + STATUS_DIM  # 70

    # Total scalar feature length
    SCALAR_LEN = MOVE_PASS_START + MOVE_PASS_DIM  # 78

    # Local map: 21×21 = 441D
    MAP_SIDE = 21

    # Total observation dimension
    # DIM_OF_OBSERVATION = SCALAR_LEN + (MAP_SIDE * MAP_SIDE)  # 519
    DIM_OF_OBSERVATION = 2724
    # ======================== Action Space ========================
    # 16 actions: 8 move + 8 flash / 16个动作：8个移动 + 8个闪现
    ACTION_NUM = 16

    # ======================== Value Head ========================
    # Single value head / 单头价值
    VALUE_NUM = 1

    # ======================== PPO Hyperparameters ========================
    GAMMA = 0.995     # 经验推荐值，长期存活任务需更看重远期回报
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.02   # 大幅提高熵正则，抑制策略过早坍塌（原 0.008 导致熵崩塌）
    CLIP_PARAM = 0.2
    VF_COEF = 0.5      # 降低 Critic loss 权重，防止 Critic 梯度干扰 Actor
    GRAD_CLIP_RANGE = 0.5
