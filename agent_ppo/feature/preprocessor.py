#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 增强版特征预处理与奖励设计。

特征结构 (76D scalar + 441D map = 517D total):
  - Hero self      (10D): 位置、闪现、buff、分数进展、卡住/绕圈检测
  - Monster        (20D): 两只怪物状态 + 包夹/距离趋势/加速预警
  - Treasure/Buff  (14D): 最近宝箱方向距离安全度 + buff状态
  - Map/Escape     (16D): 8方向射线 + 逃生深度 + 开阔度 + 死角检测
  - Status/Legal    (8D): 高压标识、阶段、合法动作摘要
  - Move Passability(8D): 8方向即时可通行性（含斜向移动规则）

奖励设计 (17项):
  稠密: 生存、步数分、宝箱接近、怪物距离shaping、加速缓冲、后期生存、
        corridor、包夹惩罚、死角惩罚、危险惩罚、重复探索惩罚、第二怪压力
  稀疏: 宝箱分、buff、撞墙/无效移动、闪现脱险、闪现滥用
"""

import numpy as np
from collections import deque

from agent_ppo.conf.conf import Config

# ============================================================
# Constants
# ============================================================
MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
DIAG = np.sqrt(2) * MAP_SIZE  # ~181, max possible distance on 128x128 map

# 8 movement directions: E, NE, N, NW, W, SW, S, SE
# Using (dr, dc) in grid coordinates (r=z downward, c=x rightward)
RAY_DIRS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]

# Pre-computed ray angles in atan2(dz, dx) convention
RAY_ANGLES = np.array([np.arctan2(dr, dc) for dr, dc in RAY_DIRS])

# 8 move directions: action 0-7 mapped to (drow, dcol) in map grid
# 动作0=右(E), 1=右上(NE), 2=上(N), 3=左上(NW), 4=左(W), 5=左下(SW), 6=下(S), 7=右下(SE)
# map_info 坐标系: row=z(向下), col=x(向右)
MOVE_DELTAS = [
    (0, 1),   # 0: 右 E
    (-1, 1),  # 1: 右上 NE
    (-1, 0),  # 2: 上 N
    (-1, -1), # 3: 左上 NW
    (0, -1),  # 4: 左 W
    (1, -1),  # 5: 左下 SW
    (1, 0),   # 6: 下 S
    (1, 1),   # 7: 右下 SE
]


# ============================================================
# Utility Functions
# ============================================================
def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _compute_dist(pos1, pos2):
    """Euclidean distance between two positions {x, z}."""
    dx = pos1["x"] - pos2["x"]
    dz = pos1["z"] - pos2["z"]
    return np.sqrt(dx * dx + dz * dz)


def _dir_sincos(from_pos, to_pos):
    """Compute (sin, cos) of direction from from_pos to to_pos."""
    dx = to_pos["x"] - from_pos["x"]
    dz = to_pos["z"] - from_pos["z"]
    dist = np.sqrt(dx * dx + dz * dz) + 1e-6
    # angle = atan2(dz, dx), return (sin(angle), cos(angle))
    return dz / dist, dx / dist


def _angular_distance(a1, a2):
    """Shortest angular distance between two angles (radians)."""
    diff = abs(a1 - a2) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)


# ============================================================
# Preprocessor
# ============================================================
class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all per-episode tracking state."""
        self.step_no = 0
        self.max_step = 2000
        # Monster state tracking
        self.last_min_monster_dist_norm = 1.0
        self.last_min_monster_raw_dist = DIAG
        # Score tracking
        self.last_treasure_score = 0.0
        self.last_step_score = 0.0
        self.last_buff_count = 0
        # Position tracking
        self.last_hero_pos = None
        self.position_history = deque(maxlen=30)
        # Treasure tracking
        self.last_nearest_treasure_dist = None
        # Buff tracking
        self.last_nearest_buff_dist = None
        # Openness tracking (for flash evaluation)
        self.last_openness = 0.5
        # Move passability tracking (for wall-collision reward)
        self.last_move_passability = [1.0] * 8
        # First step flag
        self.is_first_step = True
        # Action tracking for inertia
        self.prev_action = -1

    # ----------------------------------------------------------------
    # Legal Action Mask
    # ----------------------------------------------------------------
    def _get_legal_act(self, observation, hero):
        """Get 16D legal action mask from observation.

        兼容不同环境字段命名：优先 legal_act，其次 legal_action。
        若均不存在，则按比赛规则兜底推断：0-7移动恒为True，8-15闪现取决于冷却是否结束。
        """
        legal = observation.get("legal_act", None)
        if legal is None:
            legal = observation.get("legal_action", None)

        if legal is None:
            flash_cd = int(hero.get("flash_cooldown", 0) or 0)
            flash_ok = flash_cd <= 0
            legal = [True] * 8 + [flash_ok] * 8

        # Normalize to python list[bool] with length 16
        legal_list = list(legal)
        if len(legal_list) < 16:
            legal_list = legal_list + [False] * (16 - len(legal_list))
        elif len(legal_list) > 16:
            legal_list = legal_list[:16]

        return [bool(x) for x in legal_list]

    # ----------------------------------------------------------------
    # Main Entry Point
    # ----------------------------------------------------------------
    def feature_process(self, env_obs, last_action):
        """Process env_obs into 68D scalar + 441D map features + reward.

        主入口：处理观测，返回 (特征向量, 合法动作掩码, 奖励)。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 2000)

        hero = frame_state["heroes"]
        h_pos = hero["pos"]
        monsters = frame_state.get("monsters", [])
        organs = frame_state.get("organs", [])
        legal_act_raw = self._get_legal_act(observation, hero)

        # ===== Shared intermediate computations =====
        flash_available = any(legal_act_raw[i] for i in range(8, 16))

        # Displacement from last position
        if self.last_hero_pos is not None:
            displacement = _compute_dist(h_pos, self.last_hero_pos)
        else:
            displacement = 1.0

        is_stuck = displacement < 0.5 if self.last_hero_pos is not None else False

        # Loop detection: how many recent positions are close to current
        if len(self.position_history) > 5:
            repeat_count = sum(
                1 for p in self.position_history
                if abs(p[0] - h_pos["x"]) + abs(p[1] - h_pos["z"]) < 3
            )
            is_looping = repeat_count / len(self.position_history) > 0.3
        else:
            is_looping = False

        # Monster processing
        min_monster_dist = DIAG
        nearest_monster_pos = None
        monster_dists = []
        for m in monsters:
            m_pos = m.get("pos", {"x": 0, "z": 0})
            d = _compute_dist(h_pos, m_pos)
            monster_dists.append(d)
            if d < min_monster_dist:
                min_monster_dist = d
                nearest_monster_pos = m_pos

        monsters_sped_up = (
            any(m.get("speed", 1) > 1 for m in monsters) if monsters else False
        )
        second_monster_active = len(monsters) >= 2
        is_high_pressure = (
            min_monster_dist < 20
            or self.step_no > self.max_step * 0.5
            or monsters_sped_up
        )

        # Treasure / Buff classification
        treasures = [
            o for o in organs
            if o.get("sub_type", 0) == 1 and o.get("status", 0) == 1
        ]
        buffs = [
            o for o in organs
            if o.get("sub_type", 0) == 2 and o.get("status", 0) == 1
        ]

        # Ray features
        ray_feats = self._compute_rays(map_info)
        openness = sum(ray_feats) / 8.0
        is_dead_end = sum(ray_feats) < 1.5

        # Hero speed: 1 normally, 2 with buff
        buff_remaining = hero.get("buff_remaining_time", 0)
        hero_speed = 2 if buff_remaining > 0 else 1

        # Move passability: 8D, whether each direction is truly passable
        # 根据英雄当前速度检查可通行性（速度2时检查2格）
        move_passability = self._compute_move_passability(map_info, hero_speed)

        # 核心优化：利用高精度的可通行性直接进行硬掩码 (Action Masking)
        # 这样模型在采样时概率直接为0，彻底消除撞墙原地卡死的可能
        for i in range(8):
            if move_passability[i] < 0.5:
                legal_act_raw[i] = False
                
        # 安全兜底：防止极端地形计算bug导致所有动作被屏蔽
        if not any(legal_act_raw):
            legal_act_raw[0] = True  # 至少给一个动作以防报错

        # ===== Build feature groups =====
        hero_feat = self._build_hero_features(
            hero, h_pos, flash_available, is_stuck, is_looping
        )
        monster_feat = self._build_monster_features(
            monsters, h_pos, monster_dists, monsters_sped_up, second_monster_active
        )
        treasure_feat = self._build_treasure_features(
            treasures, buffs, h_pos, nearest_monster_pos
        )
        map_escape_feat = self._build_map_escape_features(
            map_info, ray_feats, openness, is_dead_end,
            nearest_monster_pos, h_pos
        )
        status_feat = self._build_status_features(
            legal_act_raw, last_action, monsters_sped_up,
            second_monster_active, is_high_pressure
        )

        scalar_feature = np.array(
            hero_feat + monster_feat + treasure_feat + map_escape_feat + status_feat + move_passability,
            dtype=np.float32
        )

        # Map image (21x21 flattened)
        map_img = np.zeros(Config.MAP_SIDE * Config.MAP_SIDE, dtype=np.float32)
        if map_info is not None:
            map_img = np.array(map_info, dtype=np.float32).flatten()

        total_feature = np.concatenate([scalar_feature, map_img])

        # ===== Compute reward =====
        total_reward = self._compute_reward(
            hero=hero, h_pos=h_pos, monsters=monsters,
            treasures=treasures, buffs=buffs, env_info=env_info,
            min_monster_dist=min_monster_dist,
            nearest_monster_pos=nearest_monster_pos,
            monster_dists=monster_dists,
            ray_feats=ray_feats, openness=openness, is_dead_end=is_dead_end,
            is_high_pressure=is_high_pressure,
            last_action=last_action, displacement=displacement,
            is_stuck=is_stuck, monsters_sped_up=monsters_sped_up,
            second_monster_active=second_monster_active,
            is_looping=is_looping, map_info=map_info,
            flash_available=flash_available
        )

        # ===== Update tracking state =====
        self._update_state(
            h_pos, min_monster_dist, openness,
            last_action, hero, env_info, treasures, buffs,
            move_passability
        )

        return total_feature, [int(b) for b in legal_act_raw], total_reward

    # ================================================================
    # Feature Group Builders
    # ================================================================
    def _build_hero_features(self, hero, h_pos, flash_available, is_stuck, is_looping):
        """Hero self features. 12D.

        英雄主特征：位置、闪现、buff、速度、分数、状态。
        """
        buff_remaining = hero.get("buff_remaining_time", 0)
        hero_speed = 2.0 if buff_remaining > 0 else 1.0
        # buff即将消失的紧迫感：剩余<10步时线性增加
        buff_urgency = max(0.0, 1.0 - buff_remaining / 10.0) if buff_remaining > 0 else 0.0

        return [
            _norm(h_pos["x"], MAP_SIZE),                                 # 0: pos_x
            _norm(h_pos["z"], MAP_SIZE),                                 # 1: pos_z
            _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD),          # 2: flash_cd
            float(flash_available),                                      # 3: flash_ok
            float(buff_remaining > 0),                                   # 4: buff_active
            _norm(buff_remaining, MAX_BUFF_DURATION),                    # 5: buff_remain
            _norm(hero.get("treasure_score", 0), 1000.0),               # 6: treasure_prog
            _norm(self.step_no, self.max_step),                          # 7: step_prog
            float(is_stuck),                                             # 8: is_stuck
            float(is_looping),                                           # 9: is_looping
            _norm(hero_speed, 2.0),                                      # 10: hero_speed
            buff_urgency,                                                # 11: buff_expiring
        ]

    def _build_monster_features(self, monsters, h_pos, monster_dists,
                                monsters_sped_up, second_monster_active):
        """Monster features. 20D = 2×8 + 4.

        怪物特征：每只怪物8D（存在/方向/距离/速度/位置/桶距离），
        加上全局4D（加速预警/二怪倒计时/包夹角/距离趋势）。
        """
        feats = []

        # Per-monster features (8D each)
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                m_pos = m.get("pos", {"x": 0, "z": 0})
                sin_d, cos_d = _dir_sincos(h_pos, m_pos)
                feats.extend([
                    1.0,                                          # exists
                    sin_d,                                        # dir_sin
                    cos_d,                                        # dir_cos
                    _norm(monster_dists[i], DIAG),                # raw_dist_norm
                    _norm(m.get("speed", 1), MAX_MONSTER_SPEED),  # speed_norm
                    _norm(m_pos["x"], MAP_SIZE),                  # pos_x
                    _norm(m_pos["z"], MAP_SIZE),                  # pos_z
                    _norm(m.get("hero_l2_distance", 5), 5.0),    # bucket_dist
                ])
            else:
                feats.extend([0.0] * 8)

        # Global monster features (4D)
        # 1. Speedup urgency: 0 if already sped up, ramps up as episode progresses
        if monsters_sped_up:
            speedup_urgency = 0.0
        else:
            speedup_urgency = min(1.0, self.step_no / max(self.max_step * 0.4, 1))

        # 2. Time to 2nd monster
        if second_monster_active:
            time_to_2nd = 0.0
        elif len(monsters) >= 1:
            interval = monsters[0].get("monster_interval", 300)
            time_to_2nd = _norm(max(0, interval - self.step_no), self.max_step)
        else:
            time_to_2nd = 1.0

        # 3. Pincer angle: 1.0 = two monsters on opposite sides (worst case)
        if len(monsters) >= 2 and len(monster_dists) >= 2:
            m1p = monsters[0].get("pos", {"x": 0, "z": 0})
            m2p = monsters[1].get("pos", {"x": 0, "z": 0})
            v1 = (m1p["x"] - h_pos["x"], m1p["z"] - h_pos["z"])
            v2 = (m2p["x"] - h_pos["x"], m2p["z"] - h_pos["z"])
            d1 = monster_dists[0] + 1e-6
            d2 = monster_dists[1] + 1e-6
            cos_a = np.clip((v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2), -1, 1)
            pincer_angle = np.arccos(cos_a) / np.pi
        else:
            pincer_angle = 0.0

        # 4. Distance trend: positive = getting farther (good)
        min_dist_norm = _norm(
            min(monster_dists) if monster_dists else DIAG, DIAG
        )
        dist_trend = np.clip(
            min_dist_norm - self.last_min_monster_dist_norm, -1, 1
        )

        feats.extend([speedup_urgency, time_to_2nd, pincer_angle, dist_trend])

        return feats

    def _build_treasure_features(self, treasures, buffs, h_pos, nearest_monster_pos):
        """Treasure and buff features. 14D = 5 + 4 + 1 + 4.

        宝箱/buff特征：最近宝箱(5D) + 次近宝箱(4D) + 剩余数(1D) + 最近buff(4D)。
        """
        feats = []

        # Sort treasures by distance
        if treasures:
            t_sorted = sorted(
                [(t, _compute_dist(h_pos, t["pos"])) for t in treasures],
                key=lambda x: x[1]
            )
        else:
            t_sorted = []

        # Nearest treasure (5D)
        if t_sorted:
            t1, d1 = t_sorted[0]
            sin_d, cos_d = _dir_sincos(h_pos, t1["pos"])
            # Safety score: angle between treasure dir and monster dir
            # 1.0 = treasure opposite to monster (safest)
            if nearest_monster_pos is not None:
                t_angle = np.arctan2(
                    t1["pos"]["z"] - h_pos["z"], t1["pos"]["x"] - h_pos["x"]
                )
                m_angle = np.arctan2(
                    nearest_monster_pos["z"] - h_pos["z"],
                    nearest_monster_pos["x"] - h_pos["x"]
                )
                safety = _angular_distance(t_angle, m_angle) / np.pi
            else:
                safety = 1.0
            feats.extend([1.0, sin_d, cos_d, _norm(d1, DIAG), safety])
        else:
            feats.extend([0.0] * 5)

        # 2nd nearest treasure (4D)
        if len(t_sorted) >= 2:
            t2, d2 = t_sorted[1]
            sin_d, cos_d = _dir_sincos(h_pos, t2["pos"])
            feats.extend([1.0, sin_d, cos_d, _norm(d2, DIAG)])
        else:
            feats.extend([0.0] * 4)

        # Remaining treasure count (1D)
        feats.append(_norm(len(treasures), 10.0))

        # Nearest buff (4D)
        if buffs:
            b_sorted = sorted(
                [(b, _compute_dist(h_pos, b["pos"])) for b in buffs],
                key=lambda x: x[1]
            )
            b1, bd1 = b_sorted[0]
            sin_d, cos_d = _dir_sincos(h_pos, b1["pos"])
            feats.extend([1.0, sin_d, cos_d, _norm(bd1, DIAG)])
        else:
            feats.extend([0.0] * 4)

        return feats

    def _build_map_escape_features(self, map_info, ray_feats, openness, is_dead_end,
                                   nearest_monster_pos, h_pos):
        """Map and escape features. 16D = 8 rays + 8 derived.

        地图逃生特征：8方向射线深度 + 开阔度 + 死角 + 逃离方向深度 + 最长通路方向。
        """
        feats = list(ray_feats)  # 8D ray depths (normalized 0~1)

        # Derived features (8D)
        feats.append(openness)                    # avg ray depth
        feats.append(_norm(sum(ray_feats), 10.0)) # openness score (sum-based)
        feats.append(float(is_dead_end))          # dead-end indicator

        # Escape direction analysis
        if nearest_monster_pos is not None:
            # Direction AWAY from nearest monster
            escape_angle = np.arctan2(
                h_pos["z"] - nearest_monster_pos["z"],
                h_pos["x"] - nearest_monster_pos["x"]
            )

            # Find closest ray to escape direction
            diffs = [_angular_distance(escape_angle, ra) for ra in RAY_ANGLES]
            closest_idx = int(np.argmin(diffs))
            escape_away = ray_feats[closest_idx]

            # Perpendicular escape depth
            perp_1 = escape_angle + np.pi / 2
            perp_2 = escape_angle - np.pi / 2
            diffs_1 = [_angular_distance(perp_1, ra) for ra in RAY_ANGLES]
            diffs_2 = [_angular_distance(perp_2, ra) for ra in RAY_ANGLES]
            idx_1 = int(np.argmin(diffs_1))
            idx_2 = int(np.argmin(diffs_2))
            escape_perp = (ray_feats[idx_1] + ray_feats[idx_2]) / 2.0

            feats.append(escape_away)
            feats.append(escape_perp)
        else:
            feats.extend([0.5, 0.5])

        # Longest ray direction (sin, cos) - indicates best escape corridor
        longest_idx = int(np.argmax(ray_feats))
        ray_angle = RAY_ANGLES[longest_idx]
        feats.append(np.sin(ray_angle))
        feats.append(np.cos(ray_angle))

        # Inner ring passability: immediate 4-direction neighbors
        if map_info is not None:
            center = len(map_info) // 2
            cardinal = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # E, N, W, S
            passable = 0
            for dr, dc in cardinal:
                r, c = center + dr, center + dc
                if (0 <= r < len(map_info) and 0 <= c < len(map_info[0])
                        and map_info[r][c] != 0):
                    passable += 1
            inner_pass = passable / 4.0
        else:
            inner_pass = 0.5
        feats.append(inner_pass)

        return feats

    def _build_status_features(self, legal_act, last_action,
                               monsters_sped_up, second_monster_active,
                               is_high_pressure):
        """Status and legal action features. 8D.

        状态特征：高压标识、上一步动作、闪现可用、合法动作统计、阶段、怪物状态。
        """
        n_legal_moves = sum(1 for i in range(8) if legal_act[i])
        n_legal_flash = sum(1 for i in range(8, 16) if legal_act[i])

        # Phase indicator: 0=early, 0.33=mid, 0.67=late, 1.0=endgame
        progress = self.step_no / max(self.max_step, 1)
        if progress < 0.25:
            phase = 0.0
        elif progress < 0.5:
            phase = 0.33
        elif progress < 0.75:
            phase = 0.67
        else:
            phase = 1.0

        return [
            float(is_high_pressure),                       # 0: high_pressure
            _norm(max(0, last_action), 15.0),              # 1: last_action
            float(n_legal_flash > 0),                      # 2: any_flash_legal
            n_legal_moves / 8.0,                           # 3: move_freedom
            n_legal_flash / 8.0,                           # 4: flash_freedom
            phase,                                         # 5: phase_indicator
            float(monsters_sped_up),                       # 6: monsters_fast
            float(second_monster_active),                  # 7: 2nd_monster_on
        ]

    # ================================================================
    # Ray Casting
    # ================================================================
    def _compute_rays(self, map_info):
        """Compute 8-direction ray depths from local map. Returns 8D [0,1] list."""
        ray_feats = [0.0] * 8
        if map_info is None:
            return ray_feats
        center = len(map_info) // 2
        for idx, (dr, dc) in enumerate(RAY_DIRS):
            steps = 0
            for d in range(1, 11):
                r, c = center + dr * d, center + dc * d
                if (0 <= r < len(map_info) and 0 <= c < len(map_info[0])
                        and map_info[r][c] != 0):
                    steps += 1
                else:
                    break
            ray_feats[idx] = _norm(steps, 10.0)
        return ray_feats

    def _compute_move_passability(self, map_info, hero_speed=1):
        """Compute 8D move passability: 1.0 = can move, 0.0 = wall/blocked.

        计算8个移动方向的真实可通行性。
        - 速度=1时检查1格邻居
        - 速度=2时检查2格（加速buff状态），但碰墙会停在最后可通行格
        考虑比赛规则中斜向移动的特殊限制。
        """
        passability = [0.0] * 8
        if map_info is None:
            return passability

        center = len(map_info) // 2  # 应为 10 (21//2)
        steps = min(int(hero_speed), 2)  # 1 or 2

        for action_id, (dr, dc) in enumerate(MOVE_DELTAS):
            # 速度=1: 检查1格; 速度=2: 需要第1格可通行
            # (速度>1碰墙时会停在最后可通行格，所以第1格必须可通行)
            tr1, tc1 = center + dr, center + dc

            # 第1格必须在视野内且可通行
            if not (0 <= tr1 < len(map_info) and 0 <= tc1 < len(map_info[0])):
                passability[action_id] = 0.0
                continue

            if map_info[tr1][tc1] == 0:  # 第1格是障碍物
                passability[action_id] = 0.0
                continue

            # 检查斜向移动的邻边约束（第1格）
            if dr != 0 and dc != 0:
                horiz_r, horiz_c = center, center + dc
                vert_r, vert_c = center + dr, center

                horiz_ok = (0 <= horiz_r < len(map_info) and
                           0 <= horiz_c < len(map_info[0]) and
                           map_info[horiz_r][horiz_c] != 0)
                vert_ok = (0 <= vert_r < len(map_info) and
                          0 <= vert_c < len(map_info[0]) and
                          map_info[vert_r][vert_c] != 0)

                if not (horiz_ok or vert_ok):
                    passability[action_id] = 0.0
                    continue

            # 第1格可通行，速度=1时直接成功
            passability[action_id] = 1.0

        return passability

    # ================================================================
    # Reward Computation
    # ================================================================
    def _compute_reward(self, hero, h_pos, monsters, treasures, buffs, env_info,
                        min_monster_dist, nearest_monster_pos, monster_dists,
                        ray_feats, openness, is_dead_end, is_high_pressure,
                        last_action, displacement, is_stuck,
                        monsters_sped_up, second_monster_active, is_looping,
                        map_info, flash_available):
        """Compute comprehensive shaped reward. Returns [float].

        综合奖励函数：前期鼓励拿资源+稳定推进，后期鼓励保命+拉空间。
        """
        # Skip on first step (no valid comparison state)
        if self.is_first_step:
            return [0.01]

        reward = 0.0

        # --- 1. Base survival reward (稠密) ---
        reward += 0.05  # 提高基础存活奖励，让 Agent 更珍惜生存

        # --- 2. Step score reward (稠密) ---
        cur_step_score = hero.get("step_score", self.step_no * 1.5)
        if cur_step_score > self.last_step_score:
            reward += 0.005

        # --- 3. Treasure reward (稀疏, 核心得分项) ---
        cur_treasure_score = hero.get("treasure_score", 0)
        if cur_treasure_score > self.last_treasure_score:
            t_score_diff = cur_treasure_score - self.last_treasure_score
            # 每个宝箱值100分，计算吃到的宝箱个数 (支持闪现一串多)
            chests_collected = t_score_diff / 100.0
            reward += 5.0 * chests_collected  # 极致诱惑：将宝箱价值拉升到“命可以不要，宝箱必须吃”的高度

        # --- 4. Buff拾取与管理奖励 (资源囤积) ---
        cur_buff_count = env_info.get("collected_buff", 0)
        if cur_buff_count > self.last_buff_count:
            buff_diff = cur_buff_count - self.last_buff_count
            if is_high_pressure:
                reward += 0.8 * buff_diff  # 高压期吃 buff：救命神技，重赏！
            else:
                reward += 0.1 * buff_diff  # 低压期吃 buff：顺手可以，但仍不如留到危急时刻

        # --- 4b. Buff 接近奖励 ---
        if buffs:
            nearest_b_dist = min(_compute_dist(h_pos, b["pos"]) for b in buffs)
        else:
            nearest_b_dist = None

        if (nearest_b_dist is not None and self.last_nearest_buff_dist is not None):
            b_delta = self.last_nearest_buff_dist - nearest_b_dist
            b_delta_clipped = max(min(b_delta, 3.0), -3.0)  # 双向截断防止闪现暴增暴涨
            
            if is_high_pressure:
                reward += 0.03 * b_delta_clipped  # 高压：严格对齐进退双向梯度
            else:
                reward += 0.01 * b_delta_clipped  # 低压：顺手就捎上
            if not is_high_pressure and nearest_b_dist is not None and 0 < nearest_b_dist <= 5:
                reward += 0.005

        # --- 5. Treasure approach reward (稠密，无条件激励) ---
        if treasures:
            nearest_t_dist = min(_compute_dist(h_pos, t["pos"]) for t in treasures)
        else:
            nearest_t_dist = None

        if (nearest_t_dist is not None
                and self.last_nearest_treasure_dist is not None):
            delta = self.last_nearest_treasure_dist - nearest_t_dist
            delta_clipped = max(min(delta, 3.0), -3.0)  # 双向对称，消除刷分BUG
            
            if min_monster_dist > 18:
                reward += 0.05 * delta_clipped
            else:
                reward += 0.02 * delta_clipped

        # --- 4c. Buff efficiency reward (稠密, reward good use of buff) ---
        buff_remaining = hero.get("buff_remaining_time", 0)
        if buff_remaining > 0:
            # 有buff时远离怪物额外加分（加速逃跑的价值）
            if min_monster_dist > 20:
                reward += 0.01
            # 有buff时接近宝箱额外加分（加速拿宝的价值）
            if (nearest_t_dist is not None
                    and self.last_nearest_treasure_dist is not None):
                t_delta = self.last_nearest_treasure_dist - nearest_t_dist
                if t_delta > 0:
                    reward += 0.01 * min(t_delta, 3.0)

        # --- 6. Monster distance shaping (稠密, CORE signal) ---
        dist_norm = _norm(min_monster_dist, DIAG)
        # 视野内无危险时（>18），不再受距离拉踩加分诱惑，防止为了无意义的安全距离跑进死角反复横跳
        if min_monster_dist > 18:
            shaping = 0.0
        else:
            pressure_weight = 0.3 + 0.4 * float(is_high_pressure)  # 加大核心稠密信号
            shaping = pressure_weight * (dist_norm - self.last_min_monster_dist_norm)
        reward += shaping

        # --- 7. Pre-speedup buffer reward (稠密) ---
        if not monsters_sped_up and self.step_no > self.max_step * 0.3:
            urgency = min(1.0, (self.step_no - self.max_step * 0.3) /
                          max(self.max_step * 0.2, 1))
            buffer_bonus = 0.01 * urgency * _norm(min_monster_dist, 50.0)
            reward += buffer_bonus
            if len(monster_dists) >= 2:
                reward += 0.005 * urgency * _norm(monster_dists[1], 50.0)

        # --- 8. Post-speedup survival bonus (稠密) ---
        if monsters_sped_up:
            reward += 0.005 * min(min_monster_dist / 30.0, 1.0)

        # --- 9. Corridor reward (稠密) ---
        reward += 0.005 * openness

        # --- 10. Pincer penalty (稠密) ---
        if len(monsters) >= 2 and len(monster_dists) >= 2:
            m1p = monsters[0].get("pos", {"x": 0, "z": 0})
            m2p = monsters[1].get("pos", {"x": 0, "z": 0})
            v1 = (m1p["x"] - h_pos["x"], m1p["z"] - h_pos["z"])
            v2 = (m2p["x"] - h_pos["x"], m2p["z"] - h_pos["z"])
            d1 = monster_dists[0] + 1e-6
            d2 = monster_dists[1] + 1e-6
            cos_a = np.clip((v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2), -1, 1)
            angle = np.arccos(cos_a)
            if angle > np.pi * 0.6 and d1 < 30 and d2 < 40:
                severity = (angle / np.pi) * min(1, 30 / d1) * min(1, 40 / d2)
                reward -= 0.03 * severity

        # --- 11. Dead-end penalty (稠密) ---
        if is_dead_end:
            reward -= 0.05
        elif sum(ray_feats) < 2.0:
            reward -= 0.02

        # --- 12. Danger penalty (稠密) ---
        danger_threshold = 8.0 if not monsters_sped_up else 12.0
        if min_monster_dist < danger_threshold:
            danger = 1.0 - min_monster_dist / danger_threshold
            reward -= 0.15 * danger  # 加大危险惩罚，更强调避险

        # --- 13. Wall collision / ineffective movement penalty (稀疏) ---
        flash_just_used = (8 <= last_action <= 15) if last_action >= 0 else False
        if is_stuck and not flash_just_used:
            reward -= 0.1  # 大幅加重普通撞墙惩罚，但免除闪现撞墙的额外惩罚（鼓励勇敢尝试交闪）

        # --- 13b. Chose a blocked direction penalty (稀疏) ---
        # 如果上一步选了明知不通的方向，额外惩罚
        if (0 <= last_action <= 7) and hasattr(self, 'last_move_passability'):
            if self.last_move_passability[last_action] < 0.5:
                reward -= 0.08

        # --- 14. Repeated exploration penalty (稠密) ---
        if is_looping and len(self.position_history) > 5:
            repeat_count = sum(
                1 for p in self.position_history
                if abs(p[0] - h_pos["x"]) + abs(p[1] - h_pos["z"]) < 3
            )
            repeat_ratio = repeat_count / max(len(self.position_history), 1)
            reward -= 0.02 * repeat_ratio  # 提高惩罚力度，逼迫探索新区域

        # --- 15 & 16. Flash escape & Wall-crossing reward (稀疏) ---
        # --- 15 & 16. Flash escape & Wall-crossing reward (稀疏) ---
        if flash_just_used:
            flashed_over_wall = False
            
            # 使用真实的物理位移，适应闪现直线10格/斜向8格或边缘截断等各种情况
            if getattr(self, 'last_hero_pos', None) is not None:
                dz = h_pos["z"] - self.last_hero_pos["z"]
                dx = h_pos["x"] - self.last_hero_pos["x"]
                steps = max(abs(dz), abs(dx))
                
                if steps > 1:
                    step_z = dz // steps
                    step_x = dx // steps
                    
                    center_r = len(map_info) // 2
                    center_c = len(map_info[0]) // 2 if len(map_info) > 0 else center_r
                    
                    # 遍历从起点的下一格到终点的前一格
                    for i in range(1, int(steps)):
                        # 计算在最新（以h_pos为中心）的视野地图 map_info 中的坐标
                        # 起点相当于 (center_r - dz), 沿矢量每步走 step_z
                        mid_r = center_r - dz + i * step_z
                        mid_c = center_c - dx + i * step_x
                        
                        if 0 <= mid_r < len(map_info) and 0 <= mid_c < len(map_info[0]):
                            if map_info[mid_r][mid_c] == 0:  # 完整捕捉到越墙瞬间！
                                flashed_over_wall = True
                                break

            dist_improvement = min_monster_dist - self.last_min_monster_raw_dist
            openness_improvement = openness - self.last_openness
            
            # 史诗级奖励：成功用闪现穿过障碍物，按局势发奖！
            if flashed_over_wall:
                if is_high_pressure:
                    reward += 1.5  # 战时：极限翻墙逃生，继续给予救命重赏！
                else:
                    reward += 0.2  # 和平期：跑酷穿墙只给极其微小的奖励，不再值得交掉技能

            # 闪现脱险判定（英雄闪现8-10格，怪最高追5格，若真正向外逃生或翻墙，拉开距离应显著起效）
            # 提高脱险判定阈值，防止平地乱交闪现随便白嫖 1.5 奖励
            if dist_improvement > 3.0 or (openness_improvement > 0.1 and min_monster_dist > 5):
                reward += 1.5  # 闪现救命/破局奖励，大幅提升
            elif not flashed_over_wall:
                # 既没脱险也没穿墙的平地白交闪现，给予严重的浪费技能扣分
                reward -= 0.5

        # --- 17. Second monster pressure (稠密) ---
        if second_monster_active and len(monster_dists) >= 2:
            m2_dist = monster_dists[1]
            if m2_dist < 20:
                reward -= 0.02 * (1.0 - m2_dist / 20.0)
                
        # --- 18. 憋闪现威慑力奖励 (Flash hoarding) ---
        # 如果到了中后期 (怪物可能出现或加压)，尽量把闪现留在手里
        if self.step_no > self.max_step * 0.15 and flash_available:
            reward += 0.002
            
        # --- 19. Directional Inertia (宏观探索动量) ---
        # 鼓励走直线探索，避免无危险时原地左右摇摆、乱转圈
        if getattr(self, 'prev_action', -1) == last_action and 0 <= last_action <= 7:
            reward += 0.005

        return [reward]

    # ================================================================
    # State Update
    # ================================================================
    def _update_state(self, h_pos, min_monster_dist, openness,
                      last_action, hero, env_info, treasures, buffs=None,
                      move_passability=None):
        """Update persistent state after each feature_process call."""
        self.last_min_monster_dist_norm = _norm(min_monster_dist, DIAG)
        self.last_min_monster_raw_dist = min_monster_dist
        self.last_treasure_score = hero.get("treasure_score", 0)
        self.last_step_score = hero.get("step_score", self.step_no * 1.5)
        self.last_buff_count = env_info.get("collected_buff", 0)
        self.last_hero_pos = {"x": h_pos["x"], "z": h_pos["z"]}
        self.last_openness = openness

        if treasures:
            self.last_nearest_treasure_dist = min(
                _compute_dist(h_pos, t["pos"]) for t in treasures
            )
        else:
            self.last_nearest_treasure_dist = None

        # Track nearest buff distance for approach reward
        if buffs:
            self.last_nearest_buff_dist = min(
                _compute_dist(h_pos, b["pos"]) for b in buffs
            )
        else:
            self.last_nearest_buff_dist = None

        self.position_history.append((h_pos["x"], h_pos["z"]))
        if move_passability is not None:
            self.last_move_passability = move_passability
        self.is_first_step = False
        self.prev_action = last_action
