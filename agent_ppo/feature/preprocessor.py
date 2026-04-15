#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Enhanced feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 增强版特征预处理与奖励设计。

特征结构 (76D scalar + 2646D map = 2722D total):
  - Scalar 特征维持 76D 不变，包含英雄、怪物、距离、雷达射线等。
  - Map 升级为 6通道多维局部地图 (6 x 21 x 21 = 2646D)：
    [0]: 障碍物/道路
    [1]: 宝箱位置
    [2]: Buff 位置
    [3]: 怪物位置
    [4]: 探索热图 (Visit Recency)
    [5]: 最近轨迹 (Breadcrumbs)
"""

import numpy as np
from collections import deque

from agent_ppo.conf.conf import Config

# ============================================================
# Constants
# ============================================================
MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 100.0  # [修Bug] 从 2000.0 修正为 100.0，使特征敏感
MAX_BUFF_DURATION = 50.0
DIAG = np.sqrt(2) * MAP_SIZE

RAY_DIRS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
RAY_ANGLES = np.array([np.arctan2(dr, dc) for dr, dc in RAY_DIRS])

MOVE_DELTAS = [
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
    (0, -1), (1, -1), (1, 0), (1, 1)
]

# ============================================================
# Utility Functions
# ============================================================
def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

def _compute_dist(pos1, pos2):
    dx = pos1["x"] - pos2["x"]
    dz = pos1["z"] - pos2["z"]
    return np.sqrt(dx * dx + dz * dz)

def _dir_sincos(from_pos, to_pos):
    dx = to_pos["x"] - from_pos["x"]
    dz = to_pos["z"] - from_pos["z"]
    dist = np.sqrt(dx * dx + dz * dz) + 1e-6
    return dz / dist, dx / dist

def _angular_distance(a1, a2):
    diff = abs(a1 - a2) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)

# ============================================================
# Preprocessor
# ============================================================
class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 1.0
        self.last_min_monster_raw_dist = DIAG

        self.last_total_score = 0.0
        self.last_hero_pos = None
        self.position_history = deque(maxlen=60)

        self.last_nearest_treasure_dist = None
        self.last_nearest_buff_dist = None
        self.last_openness = 0.5
        self.last_move_passability = [1.0] * 8
        self.is_first_step = True
        self.prev_action = -1

        # 用于奖励计算的历史量
        self.last_collected_buff = 0
        self.last_buff_remaining = 0
        self.last_flash_cd = 0
        self.last_visit_age = 999.0
        self.last_pincer_pressure = 0.0
        self.last_min_exit_monster_dist = DIAG

        # 全局记忆矩阵：用于建立探索热图 (128x128)
        self.visited_time = np.full((128, 128), -1, dtype=np.int32)

    def _get_legal_act(self, observation, hero):
        legal = observation.get("legal_act", None) or observation.get("legal_action", None)
        if legal is None:
            flash_cd = int(hero.get("flash_cooldown", 0) or 0)
            legal = [True] * 8 + [flash_cd <= 0] * 8
        legal_list = list(legal)[:16] + [False] * max(0, 16 - len(legal))
        return [bool(x) for x in legal_list]

    # ----------------------------------------------------------------
    # Main Entry Point
    # ----------------------------------------------------------------
    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        hero = frame_state["heroes"]
        h_pos = hero["pos"]
        monsters = frame_state.get("monsters", [])
        organs = frame_state.get("organs", [])
        legal_act_raw = self._get_legal_act(observation, hero)

        # 先读取当前位置“上次来过多久”，后续 reward 要用
        hz, hx = int(h_pos["z"]), int(h_pos["x"])
        visit_age = 999.0
        if 0 <= hz < 128 and 0 <= hx < 128:
            last_visit_step = self.visited_time[hz][hx]
            if last_visit_step >= 0:
                visit_age = float(self.step_no - last_visit_step)
            else:
                visit_age = 999.0

        flash_available = any(legal_act_raw[i] for i in range(8, 16))
        displacement = _compute_dist(h_pos, self.last_hero_pos) if self.last_hero_pos else 1.0
        is_stuck = displacement < 0.5 if self.last_hero_pos else False

        repeat_count = sum(1 for p in self.position_history if abs(p[0]-h_pos["x"])+abs(p[1]-h_pos["z"]) < 3)
        is_looping = (repeat_count / len(self.position_history) > 0.3) if len(self.position_history)>5 else False

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

        monsters_sped_up = any(m.get("speed", 1) > 1 for m in monsters) if monsters else False
        second_monster_active = len(monsters) >= 2

        # 包夹压力：两怪都较近且夹角越接近 180 度越危险
        pincer_pressure = 0.0
        if len(monsters) >= 2 and len(monster_dists) >= 2:
            v1 = (monsters[0]["pos"]["x"] - h_pos["x"], monsters[0]["pos"]["z"] - h_pos["z"])
            v2 = (monsters[1]["pos"]["x"] - h_pos["x"], monsters[1]["pos"]["z"] - h_pos["z"])
            cos_a = np.clip(
                (v1[0] * v2[0] + v1[1] * v2[1]) / ((monster_dists[0] + 1e-6) * (monster_dists[1] + 1e-6)),
                -1.0,
                1.0,
            )
            opposite_score = np.clip((-cos_a - 0.2) / 0.8, 0.0, 1.0)  # 越接近相对方向越高
            close_score = np.clip((18.0 - max(monster_dists[0], monster_dists[1])) / 10.0, 0.0, 1.0)
            pincer_pressure = opposite_score * close_score

        # 出口封锁压力：面向可走方向时，那个方向上最近怪物的投影距离
        min_exit_monster_dist = DIAG
        for i, (dr, dc) in enumerate(MOVE_DELTAS):
            vx, vz = float(dc), float(dr)
            v_norm = np.sqrt(vx * vx + vz * vz) + 1e-6
            vx, vz = vx / v_norm, vz / v_norm
            for m in monsters:
                mx = m["pos"]["x"] - h_pos["x"]
                mz = m["pos"]["z"] - h_pos["z"]
                proj = mx * vx + mz * vz
                lateral = abs(mx * vz - mz * vx)
                if proj > 0.0 and lateral < 2.5:
                    min_exit_monster_dist = min(min_exit_monster_dist, proj)

        # 高压判定不再只看最近怪距离
        pressure_score = max(
            np.clip((15.0 - min_monster_dist) / 8.0, 0.0, 1.0),
            1.0 if monsters_sped_up else 0.0,
            pincer_pressure,
            np.clip((8.0 - min_exit_monster_dist) / 5.0, 0.0, 1.0),
        )
        is_high_pressure = pressure_score > 0.35

        treasures = [o for o in organs if o.get("sub_type", 0) == 1 and o.get("status", 0) == 1]
        buffs = [o for o in organs if o.get("sub_type", 0) == 2 and o.get("status", 0) == 1]

        ray_feats = self._compute_rays(map_info)
        openness = sum(ray_feats) / 8.0
        is_dead_end = sum(ray_feats) < 1.5

        hero_speed = 2 if hero.get("buff_remaining_time", 0) > 0 else 1

        # [修Bug] 正确计算加速状态下的路况
        move_passability = self._compute_move_passability(map_info, hero_speed)
        for i in range(8):
            if move_passability[i] == 0.0:  # 彻底不可走才屏蔽，0.5代表只能走1格（合法）
                legal_act_raw[i] = False

        # 仅在“非常安全”时收紧闪现，不做一刀切硬封
        very_safe = (
            min_monster_dist > 20.0
            and pincer_pressure < 0.15
            and min_exit_monster_dist > 10.0
            and not monsters_sped_up
        )
        if very_safe:
            for i in range(8, 16):
                legal_act_raw[i] = False

        if not any(legal_act_raw):
            legal_act_raw[0] = True

        hero_feat = self._build_hero_features(hero, h_pos, flash_available, is_stuck, is_looping)
        monster_feat = self._build_monster_features(monsters, h_pos, monster_dists, monsters_sped_up, second_monster_active)
        treasure_feat = self._build_treasure_features(treasures, buffs, h_pos, nearest_monster_pos)
        map_escape_feat = self._build_map_escape_features(map_info, ray_feats, openness, is_dead_end, nearest_monster_pos, h_pos)
        status_feat = self._build_status_features(legal_act_raw, last_action, monsters_sped_up, second_monster_active, is_high_pressure)

        scalar_feature = np.array(
            hero_feat + monster_feat + treasure_feat + map_escape_feat + status_feat + move_passability,
            dtype=np.float32
        )

        # 【核心重构】多通道地图生成
        map_img = self._build_multichannel_map(map_info, h_pos, treasures, buffs, monsters)
        total_feature = np.concatenate([scalar_feature, map_img])

        total_reward = self._compute_reward(
            hero=hero,
            env_info=env_info,
            min_monster_dist=min_monster_dist,
            is_high_pressure=is_high_pressure,
            is_stuck=is_stuck,
            treasures=treasures,
            buffs=buffs,
            h_pos=h_pos,
            last_action=last_action,
            flash_available=flash_available,
            move_passability=move_passability,
            visit_age=visit_age,
            pincer_pressure=pincer_pressure,
            min_exit_monster_dist=min_exit_monster_dist,
            monsters=monsters,
        )

        self._update_state(
            h_pos=h_pos,
            min_monster_dist=min_monster_dist,
            openness=openness,
            last_action=last_action,
            hero=hero,
            env_info=env_info,
            treasures=treasures,
            buffs=buffs,
            move_passability=move_passability,
            visit_age=visit_age,
            pincer_pressure=pincer_pressure,
            min_exit_monster_dist=min_exit_monster_dist,
        )

        # 最后再更新 visited_time，避免探索奖励失效
        if 0 <= hz < 128 and 0 <= hx < 128:
            self.visited_time[hz][hx] = self.step_no

        return total_feature, [int(b) for b in legal_act_raw], total_reward

    # ================================================================
    # Feature Builders
    # ================================================================
    def _build_hero_features(self, hero, h_pos, flash_available, is_stuck, is_looping):
        buff_remaining = hero.get("buff_remaining_time", 0)
        return [
            _norm(h_pos["x"], MAP_SIZE), _norm(h_pos["z"], MAP_SIZE),
            _norm(hero.get("flash_cooldown", 0), MAX_FLASH_CD), float(flash_available),
            float(buff_remaining > 0), _norm(buff_remaining, MAX_BUFF_DURATION),
            _norm(hero.get("treasure_score", 0), 1000.0), _norm(self.step_no, self.max_step),
            float(is_stuck), float(is_looping), _norm(2.0 if buff_remaining>0 else 1.0, 2.0),
            max(0.0, 1.0 - buff_remaining / 10.0) if buff_remaining > 0 else 0.0
        ]

    def _build_monster_features(self, monsters, h_pos, monster_dists, monsters_sped_up, second_monster_active):
        feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                m_pos = m.get("pos", {"x": 0, "z": 0})
                sin_d, cos_d = _dir_sincos(h_pos, m_pos)
                feats.extend([
                    1.0, sin_d, cos_d, _norm(monster_dists[i], DIAG),
                    _norm(m.get("speed", 1), MAX_MONSTER_SPEED), _norm(m_pos["x"], MAP_SIZE),
                    _norm(m_pos["z"], MAP_SIZE), _norm(m.get("hero_l2_distance", 5), 5.0),
                ])
            else:
                feats.extend([0.0] * 8)
        
        speedup_urgency = 0.0 if monsters_sped_up else min(1.0, self.step_no / max(self.max_step * 0.4, 1))
        time_to_2nd = 0.0 if second_monster_active else (
            _norm(max(0, monsters[0].get("monster_interval", 300) - self.step_no), self.max_step) if monsters else 1.0
        )
        pincer_angle = 0.0
        if len(monsters) >= 2 and len(monster_dists) >= 2:
            v1 = (monsters[0]["pos"]["x"] - h_pos["x"], monsters[0]["pos"]["z"] - h_pos["z"])
            v2 = (monsters[1]["pos"]["x"] - h_pos["x"], monsters[1]["pos"]["z"] - h_pos["z"])
            cos_a = np.clip((v1[0]*v2[0] + v1[1]*v2[1]) / ((monster_dists[0]+1e-6)*(monster_dists[1]+1e-6)), -1, 1)
            pincer_angle = np.arccos(cos_a) / np.pi
            
        dist_trend = np.clip(_norm(min(monster_dists) if monster_dists else DIAG, DIAG) - self.last_min_monster_dist_norm, -1, 1)
        feats.extend([speedup_urgency, time_to_2nd, pincer_angle, dist_trend])
        return feats

    def _build_treasure_features(self, treasures, buffs, h_pos, nearest_monster_pos):
        feats = []
        t_sorted = sorted([(t, _compute_dist(h_pos, t["pos"])) for t in treasures], key=lambda x: x[1]) if treasures else []
        if t_sorted:
            sin_d, cos_d = _dir_sincos(h_pos, t_sorted[0][0]["pos"])
            safety = 1.0
            if nearest_monster_pos:
                t_ang = np.arctan2(t_sorted[0][0]["pos"]["z"] - h_pos["z"], t_sorted[0][0]["pos"]["x"] - h_pos["x"])
                m_ang = np.arctan2(nearest_monster_pos["z"] - h_pos["z"], nearest_monster_pos["x"] - h_pos["x"])
                safety = _angular_distance(t_ang, m_ang) / np.pi
            feats.extend([1.0, sin_d, cos_d, _norm(t_sorted[0][1], DIAG), safety])
        else:
            feats.extend([0.0] * 5)
            
        if len(t_sorted) >= 2:
            sin_d, cos_d = _dir_sincos(h_pos, t_sorted[1][0]["pos"])
            feats.extend([1.0, sin_d, cos_d, _norm(t_sorted[1][1], DIAG)])
        else:
            feats.extend([0.0] * 4)
            
        feats.append(_norm(len(treasures), 10.0))
        
        b_sorted = sorted([(b, _compute_dist(h_pos, b["pos"])) for b in buffs], key=lambda x: x[1]) if buffs else []
        if b_sorted:
            sin_d, cos_d = _dir_sincos(h_pos, b_sorted[0][0]["pos"])
            feats.extend([1.0, sin_d, cos_d, _norm(b_sorted[0][1], DIAG)])
        else:
            feats.extend([0.0] * 4)
        return feats

    def _build_map_escape_features(self, map_info, ray_feats, openness, is_dead_end, nearest_monster_pos, h_pos):
        feats = list(ray_feats)
        feats.extend([openness, _norm(sum(ray_feats), 10.0), float(is_dead_end)])
        if nearest_monster_pos:
            esc_ang = np.arctan2(h_pos["z"] - nearest_monster_pos["z"], h_pos["x"] - nearest_monster_pos["x"])
            feats.append(ray_feats[int(np.argmin([_angular_distance(esc_ang, ra) for ra in RAY_ANGLES]))])
            idx1 = int(np.argmin([_angular_distance(esc_ang + np.pi/2, ra) for ra in RAY_ANGLES]))
            idx2 = int(np.argmin([_angular_distance(esc_ang - np.pi/2, ra) for ra in RAY_ANGLES]))
            feats.append((ray_feats[idx1] + ray_feats[idx2]) / 2.0)
        else:
            feats.extend([0.5, 0.5])
            
        longest_idx = int(np.argmax(ray_feats))
        feats.extend([np.sin(RAY_ANGLES[longest_idx]), np.cos(RAY_ANGLES[longest_idx])])
        
        inner_pass = 0.5
        if map_info:
            c = len(map_info)//2
            inner_pass = sum(1 for dr,dc in [(0,1),(-1,0),(0,-1),(1,0)] if map_info[c+dr][c+dc]!=0)/4.0
        feats.append(inner_pass)
        return feats

    def _build_status_features(self, legal_act, last_action, monsters_sped_up, second_monster_active, is_high_pressure):
        n_moves = sum(1 for i in range(8) if legal_act[i])
        n_flash = sum(1 for i in range(8, 16) if legal_act[i])
        prog = self.step_no / max(self.max_step, 1)
        phase = 0.0 if prog < 0.25 else (0.33 if prog < 0.5 else (0.67 if prog < 0.75 else 1.0))
        return [
            float(is_high_pressure), _norm(max(0, last_action), 15.0), float(n_flash > 0),
            n_moves / 8.0, n_flash / 8.0, phase, float(monsters_sped_up), float(second_monster_active)
        ]

    # ================================================================
    # Multi-Channel Map Generation
    # ================================================================
    def _build_multichannel_map(self, map_info, h_pos, treasures, buffs, monsters):
        """构建 6 通道 21x21 空间感知地图"""
        channels = np.zeros((6, 21, 21), dtype=np.float32)
        if not map_info:
            return channels.flatten()
            
        # [0] 障碍物层 (1路 0墙)
        map_arr = np.array(map_info)
        channels[0] = (map_arr > 0).astype(np.float32)
        
        hz, hx = int(h_pos["z"]), int(h_pos["x"])
        
        def global_to_local(gz, gx):
            r, c = gz - hz + 10, gx - hx + 10
            if 0 <= r < 21 and 0 <= c < 21:
                return int(r), int(c)
            return None, None

        # [1] 宝箱层 & [2] Buff层
        for t in treasures:
            r, c = global_to_local(int(t["pos"]["z"]), int(t["pos"]["x"]))
            if r is not None: channels[1, r, c] = 1.0
        for b in buffs:
            r, c = global_to_local(int(b["pos"]["z"]), int(b["pos"]["x"]))
            if r is not None: channels[2, r, c] = 1.0
            
        # [3] 怪物分布
        for m in monsters:
            mz, mx = int(m.get("pos", {"z":0})["z"]), int(m.get("pos", {"x":0})["x"])
            r, c = global_to_local(mz, mx)
            if r is not None: channels[3, r, c] = 1.0
            
        # [4] 全局访问热图 & [5] 最近轨迹 Breadcrumb
        for r in range(21):
            for c in range(21):
                gz, gx = hz - 10 + r, hx - 10 + c
                if 0 <= gz < 128 and 0 <= gx < 128:
                    vt = self.visited_time[gz][gx]
                    if vt >= 0:
                        channels[4, r, c] = max(0.1, 1.0 - (self.step_no - vt)/300.0)
                        
        for i, p in enumerate(list(self.position_history)[-10:]):
            r, c = global_to_local(int(p[1]), int(p[0]))
            if r is not None: channels[5, r, c] = max(channels[5, r, c], (i+1)/10.0)

        return channels.flatten()

    def _compute_rays(self, map_info):
        ray_feats = [0.0] * 8
        if not map_info: return ray_feats
        c = len(map_info) // 2
        for idx, (dr, dc) in enumerate(RAY_DIRS):
            steps = 0
            for d in range(1, 11):
                r, c_ = c + dr * d, c + dc * d
                if 0 <= r < len(map_info) and 0 <= c_ < len(map_info[0]) and map_info[r][c_] != 0:
                    steps += 1
                else: break
            ray_feats[idx] = _norm(steps, 10.0)
        return ray_feats

    def _compute_move_passability(self, map_info, hero_speed=1):
        """完全修补的移动检测，返回 1.0(畅通), 0.5(被卡但能走一格), 0.0(死路)"""
        passability = [0.0] * 8
        if not map_info: return passability
        c = len(map_info) // 2
        speed = min(int(hero_speed), 2)
        
        for idx, (dr, dc) in enumerate(MOVE_DELTAS):
            r1, c1 = c + dr, c + dc
            if not (0 <= r1 < len(map_info) and 0 <= c1 < len(map_info[0])) or map_info[r1][c1] == 0:
                continue
            if dr != 0 and dc != 0 and map_info[c][c1] == 0 and map_info[r1][c] == 0:
                continue # 斜向约束
                
            if speed == 1:
                passability[idx] = 1.0
            else:
                r2, c2 = c + 2*dr, c + 2*dc
                if not (0 <= r2 < len(map_info) and 0 <= c2 < len(map_info[0])) or map_info[r2][c2] == 0:
                    passability[idx] = 0.5 # 只能走1格
                else:
                    if dr != 0 and dc != 0 and map_info[r1][c2] == 0 and map_info[r2][c1] == 0:
                        passability[idx] = 0.5
                    else:
                        passability[idx] = 1.0
        return passability

    # ================================================================
    # Reward Backbone Redesign
    # ================================================================

    def _compute_reward(
        self,
        hero,
        env_info,
        min_monster_dist,
        is_high_pressure,
        is_stuck,
        treasures,
        buffs,
        h_pos,
        last_action,
        flash_available,
        move_passability,
        visit_age,
        pincer_pressure,
        min_exit_monster_dist,
        monsters,
    ):
        if self.is_first_step:
            return [0.0]

        # ------------------------------------------------
        # 1) 主奖励：分数增量
        # ------------------------------------------------
        cur_score = hero.get("step_score", self.step_no * 1.5) + hero.get("treasure_score", 0)
        score_delta = cur_score - self.last_total_score
        reward = score_delta / 100.0

        flash_used = 8 <= last_action <= 15
        buff_remaining = hero.get("buff_remaining_time", 0)
        cur_collected_buff = env_info.get("collected_buff", 0)

        nearest_t = min([_compute_dist(h_pos, t["pos"]) for t in treasures]) if treasures else None
        nearest_b = min([_compute_dist(h_pos, b["pos"]) for b in buffs]) if buffs else None

        # ------------------------------------------------
        # 2) 低压下：鼓励拿宝箱，但更强地鼓励吃 buff
        # ------------------------------------------------
        if not is_high_pressure:
            if nearest_t is not None and self.last_nearest_treasure_dist is not None:
                delta_t = self.last_nearest_treasure_dist - nearest_t
                reward += 0.02 * np.clip(delta_t, -2.0, 2.0)

            if nearest_b is not None and self.last_nearest_buff_dist is not None:
                delta_b = self.last_nearest_buff_dist - nearest_b
                reward += 0.04 * np.clip(delta_b, -2.0, 2.0)

        # 吃到 buff 的瞬时奖励：必须明确拉高价值
        if cur_collected_buff > self.last_collected_buff:
            reward += 1.2

        # buff 生效期间，给一点轻微正反馈，鼓励利用 buff 去探图/脱险
        if buff_remaining > 0:
            reward += 0.01

        # ------------------------------------------------
        # 3) 高压下：鼓励脱离危险，不再贪目标
        # ------------------------------------------------
        if is_high_pressure:
            dist_diff = min_monster_dist - self.last_min_monster_raw_dist
            reward += 0.06 * np.clip(dist_diff, -2.5, 2.5)

            # 包夹压力上升要罚，下降给正反馈
            reward += 0.10 * (self.last_pincer_pressure - pincer_pressure)

            # 出口封锁缓解也给正反馈
            exit_relief = np.clip(min_exit_monster_dist - self.last_min_exit_monster_dist, -3.0, 3.0)
            reward += 0.03 * exit_relief

        # ------------------------------------------------
        # 4) 探索奖励：鼓励新路、避免死胡同隔墙等死
        # ------------------------------------------------
        # 首次访问 / 很久没来过，给正反馈
        if visit_age >= 999.0:
            reward += 0.08
        elif visit_age > 40.0:
            reward += 0.04
        elif visit_age < 6.0:
            reward -= 0.02

        # 路很窄且长期原地磨蹭，额外处罚
        open_dirs = sum(1 for x in move_passability if x > 0.0)
        if open_dirs <= 2 and is_stuck and not flash_used:
            reward -= 0.08

        # ------------------------------------------------
        # 5) 卡住惩罚：buff 期间宽松处理
        # ------------------------------------------------
        if is_stuck and not flash_used:
            if buff_remaining > 0:
                reward -= 0.03
            else:
                reward -= 0.10

        # ------------------------------------------------
        # 6) 闪现质量奖励：不再只是“低压罚一下”
        # ------------------------------------------------
        if flash_used:
            if not is_high_pressure:
                reward -= 0.35
            else:
                # 在高压下，若明显拉开距离 / 缓解包夹 / 缓解出口封锁，则奖励
                dist_gain = np.clip(min_monster_dist - self.last_min_monster_raw_dist, -5.0, 5.0)
                reward += 0.08 * max(0.0, dist_gain)
                reward += 0.15 * max(0.0, self.last_pincer_pressure - pincer_pressure)
                reward += 0.04 * max(0.0, min_exit_monster_dist - self.last_min_exit_monster_dist)

                # 高压下交闪但没改善局势，照样罚
                if dist_gain < 0.5 and (self.last_pincer_pressure - pincer_pressure) < 0.05:
                    reward -= 0.12

        return [float(reward)]
    def _update_state(
        self,
        h_pos,
        min_monster_dist,
        openness,
        last_action,
        hero,
        env_info,
        treasures,
        buffs,
        move_passability,
        visit_age,
        pincer_pressure,
        min_exit_monster_dist,
    ):
        self.last_min_monster_dist_norm = _norm(min_monster_dist, DIAG)
        self.last_min_monster_raw_dist = min_monster_dist
        self.last_total_score = hero.get("step_score", self.step_no * 1.5) + hero.get("treasure_score", 0)
        self.last_hero_pos = {"x": h_pos["x"], "z": h_pos["z"]}
        self.last_openness = openness

        self.last_nearest_treasure_dist = min([_compute_dist(h_pos, t["pos"]) for t in treasures]) if treasures else None
        self.last_nearest_buff_dist = min([_compute_dist(h_pos, b["pos"]) for b in buffs]) if buffs else None

        self.last_collected_buff = env_info.get("collected_buff", 0)
        self.last_buff_remaining = hero.get("buff_remaining_time", 0)
        self.last_flash_cd = hero.get("flash_cooldown", 0)
        self.last_visit_age = visit_age
        self.last_pincer_pressure = pincer_pressure
        self.last_min_exit_monster_dist = min_exit_monster_dist

        self.position_history.append((h_pos["x"], h_pos["z"]))
        self.last_move_passability = move_passability
        self.is_first_step = False
        self.prev_action = last_action