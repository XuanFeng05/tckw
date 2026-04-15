#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Curriculum learning configuration for Gorge Chase PPO.
峡谷追猎 PPO 课程学习配置。

4 阶段课程：
  1. warmup_stable   (0~300)   — 资源充足，压力低，学习基础拿分
  2. mid_pressure    (300~800) — 逐步加压，学习平衡拿分与避险
  3. late_survival   (800~1300) — 高压生存，学习后期保命
  4. hard_general    (1300+)    — 完全泛化，适应各种随机配置
"""

import copy
import random


# ============================================================
# Curriculum Stage Definitions
# ============================================================
CURRICULUM_STAGES = [
    {
        "name": "warmup_stable",
        "max_episode": 300,         # 0~300: 延长预热期，充分学习基础
        "treasure_count": (9, 10),
        "buff_count": (2, 2),         # 比赛规则限制最大为2
        "monster_interval": (420, 500),
        "monster_speedup": (600, 700), # 怪物更晚加速，降低早期压力
        "max_step": 2000,
    },
    {
        "name": "mid_pressure",
        "max_episode": 800,         # 301~800: 给足中压阶段训练时间 (修正了累加bug)
        "treasure_count": (8, 10),
        "buff_count": (2, 2),
        "monster_interval": (360, 480),
        "monster_speedup": (440, 620),
        "max_step": 2000,
    },
    {
        "name": "late_survival",
        "max_episode": 1300,        # 801~1300: 延长高压训练 (修正了累加bug)
        "treasure_count": (7, 10),
        "buff_count": (2, 2),
        "monster_interval": (320, 420),
        "monster_speedup": (380, 520),
        "max_step": 2000,
    },
    {
        "name": "hard_generalization",
        "max_episode": float("inf"),# 1300+: 完全泛化
        "treasure_count": (6, 10),
        "buff_count": (2, 2),
        "monster_interval": (320, 520),
        "monster_speedup": (340, 620),
        "max_step": 2000,
    },
]

# Training maps: 1-8, Validation maps: 9-10
TRAIN_MAPS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_MAPS = [9, 10]


def _get_stage(episode_cnt):
    """Return the curriculum stage dict for a given episode count."""
    for stage in CURRICULUM_STAGES:
        if episode_cnt <= stage["max_episode"]:
            return stage
    return CURRICULUM_STAGES[-1]


def get_stage_name(episode_cnt):
    """Return the name of the current curriculum stage."""
    return _get_stage(episode_cnt)["name"]


def get_curriculum_config(episode_cnt, base_conf):
    """Generate a training config with curriculum-based randomization.

    根据当前 episode 数，从对应课程阶段的范围中随机采样参数。

    Args:
        episode_cnt: Current episode number.
        base_conf: Base configuration dict (from toml).

    Returns:
        Deep copy of base_conf with curriculum overrides applied.
    """
    conf = copy.deepcopy(base_conf)
    stage = _get_stage(episode_cnt)

    # Get or create env_conf sub-dict
    if "env_conf" in conf:
        env_conf = conf["env_conf"]
    else:
        conf["env_conf"] = {}
        env_conf = conf["env_conf"]

    env_conf["treasure_count"] = random.randint(*stage["treasure_count"])
    env_conf["buff_count"] = random.randint(*stage["buff_count"])
    env_conf["monster_interval"] = random.randint(*stage["monster_interval"])
    env_conf["monster_speedup"] = random.randint(*stage["monster_speedup"])
    
    # 引入 max_step 混训：50% 1000步(官方目标)，30% 1200步，20% 2000步(保鲁棒性)
    rand_step = random.random()
    if rand_step < 0.5:
        env_conf["max_step"] = 1000
    elif rand_step < 0.8:
        env_conf["max_step"] = 1200
    else:
        env_conf["max_step"] = stage.get("max_step", 2000)

    env_conf["map"] = TRAIN_MAPS
    env_conf["map_random"] = True

    return conf


def get_val_config(base_conf):
    """Generate a validation config with fixed settings on maps 9-10.

    验证配置使用固定参数，在地图 9-10 上评估模型泛化能力。

    Args:
        base_conf: Base configuration dict (from toml).

    Returns:
        Deep copy of base_conf with validation overrides applied.
    """
    conf = copy.deepcopy(base_conf)

    if "env_conf" in conf:
        env_conf = conf["env_conf"]
    else:
        conf["env_conf"] = {}
        env_conf = conf["env_conf"]

    env_conf["map"] = VAL_MAPS
    env_conf["map_random"] = True
    env_conf["treasure_count"] = 10
    env_conf["buff_count"] = 2
    
    # 修改验证集配置，严格对齐官方默认难度
    env_conf["monster_interval"] = 300      # 官方：300步出第一怪
    env_conf["monster_speedup"] = 500       # 官方：500步第一怪加速
    env_conf["max_step"] = 1000             # 官方默认结束步数
    env_conf["buff_cooldown"] = 200         # 官方默认 buff CD
    env_conf["talent_cooldown"] = 100       # 官方闪现 CD

    return conf