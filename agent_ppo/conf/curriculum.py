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
        "max_episode": 300,         # 延长预热期，充分学习基础
        "treasure_count": (9, 10),
        "buff_count": (2, 2),         # 比赛规则限制最大为2
        "monster_interval": (370, 450),
        "monster_speedup": (550, 650), # 怪物更晚加速，降低早期压力
        "max_step": 2000,
    },
    {
        "name": "mid_pressure",
        "max_episode": 500,        # 给足中压阶段训练时间
        "treasure_count": (8, 10),
        "buff_count": (1, 2),
        "monster_interval": (310, 430),
        "monster_speedup": (390, 570),
        "max_step": 2000,
    },
    {
        "name": "late_survival",
        "max_episode": 500,        # 延长高压训练
        "treasure_count": (7, 10),
        "buff_count": (1, 2),
        "monster_interval": (270, 370),
        "monster_speedup": (330, 470),
        "max_step": 2000,
    },
    {
        "name": "hard_generalization",
        "max_episode": float("inf"),
        "treasure_count": (6, 10),
        "buff_count": (0, 2),
        "monster_interval": (270, 470),
        "monster_speedup": (290, 570),
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
    env_conf["max_step"] = stage["max_step"]
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
    env_conf["monster_interval"] = 300
    env_conf["monster_speedup"] = 500
    env_conf["max_step"] = 1000
    env_conf["buff_cooldown"] = 200
    env_conf["talent_cooldown"] = 100

    return conf
