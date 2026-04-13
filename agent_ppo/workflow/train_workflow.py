#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase PPO with curriculum learning and train/val split.
峡谷追猎 PPO 训练工作流（课程学习 + 训练/验证分离）。

改进点:
  1. Curriculum Learning: 4 阶段课程学习，逐步增加环境难度
  2. Train/Val Split: 训练地图 1-8，验证地图 9-10，每 10 局跑 1 局验证
  3. Val 使用 greedy 策略评估，不收集训练样本
  4. 日志区分 [TRAIN] / [VAL]，监控分离
"""

import os
import time

import numpy as np
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.conf.curriculum import (
    get_curriculum_config, get_val_config, get_stage_name
)
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read base user config / 读取基础用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        base_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, base_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.base_conf = base_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_report_val_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        """Run training and validation episodes, yielding samples for training only.

        执行训练和验证对局，仅训练对局 yield 样本。
        验证对局使用 greedy 策略评估，不产生训练数据。
        """
        while True:
            self.episode_cnt += 1

            # Determine if this is a validation episode
            # 每 10 个 episode 进行 1 次验证
            is_val = (self.episode_cnt % 10 == 0) and (self.episode_cnt > 0)

            # Generate appropriate config / 生成对应配置
            if is_val:
                ep_conf = get_val_config(self.base_conf)
            else:
                ep_conf = get_curriculum_config(self.episode_cnt, self.base_conf)

            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset env with episode-specific config / 用对应配置重置环境
            env_obs = self.env.reset(ep_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            done = False
            step = 0
            total_reward = 0.0

            stage_name = "val" if is_val else get_stage_name(self.episode_cnt)
            self.logger.info(
                f"Episode {self.episode_cnt} start [{stage_name}]"
            )

            while not done:
                # Predict action / Agent 推理
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]

                # Val uses greedy, Train uses stochastic
                # 验证用 greedy，训练用随机采样
                act = self.agent.action_process(
                    act_data, is_stochastic=(not is_val)
                )

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    treasure_score = env_info.get("treasure_score", 0)

                    if terminated:
                        final_reward[0] = -10.0
                        result_str = "FAIL"
                    else:
                        final_reward[0] = 10.0
                        result_str = "WIN"

                    prefix = "[VAL]" if is_val else "[TRAIN]"
                    self.logger.info(
                        f"{prefix} episode:{self.episode_cnt} "
                        f"stage:{stage_name} steps:{step} "
                        f"result:{result_str} sim_score:{total_score:.1f} "
                        f"treasure:{treasure_score:.0f} "
                        f"total_reward:{total_reward:.3f}"
                    )

                # Build sample (training only) / 构造样本帧（仅训练）
                if not is_val:
                    frame = SampleData(
                        obs=np.array(obs_data.feature, dtype=np.float32),
                        legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                        act=np.array([act_data.action[0]], dtype=np.float32),
                        reward=reward,
                        done=np.array([float(done)], dtype=np.float32),
                        reward_sum=np.zeros(1, dtype=np.float32),
                        value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                        next_value=np.zeros(1, dtype=np.float32),
                        advantage=np.zeros(1, dtype=np.float32),
                        prob=np.array(act_data.prob, dtype=np.float32),
                    )
                    collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if not is_val:
                        # --- Training episode end ---
                        if collector:
                            collector[-1].reward = collector[-1].reward + final_reward

                        # Monitor report / 训练监控上报
                        now = time.time()
                        if now - self.last_report_monitor_time >= 60 and self.monitor:
                            env_info_final = env_obs["observation"]["env_info"]
                            monitor_data = {
                                "reward": round(total_reward + float(final_reward[0]), 4),
                                "episode_steps": step,
                                "episode_cnt": self.episode_cnt,
                                "treasure_score": round(
                                    env_info_final.get("treasure_score", 0), 1
                                ),
                                "total_score": round(
                                    env_info_final.get("total_score", 0), 1
                                ),
                            }
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                        if collector:
                            collector = sample_process(collector)
                            yield collector
                    else:
                        # --- Validation episode end ---
                        now = time.time()
                        if now - self.last_report_val_time >= 60 and self.monitor:
                            env_info_final = env_obs["observation"]["env_info"]
                            val_data = {
                                "val_reward": round(
                                    total_reward + float(final_reward[0]), 4
                                ),
                                "episode_steps": step,
                                "treasure_score": round(
                                    env_info_final.get("treasure_score", 0), 1
                                ),
                                "total_score": round(
                                    env_info_final.get("total_score", 0), 1
                                ),
                            }
                            self.monitor.put_data({os.getpid(): val_data})
                            self.last_report_val_time = now
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
