#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        # ===== Training Metrics / 训练指标 =====
        .add_panel(
            name="训练累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()
        # ===== Validation Metrics / 验证指标 =====
        .add_panel(
            name="验证累积回报",
            name_en="val_reward",
            type="line",
        )
        .add_metric(
            metrics_name="val_reward",
            expr="avg(val_reward{})",
        )
        .end_panel()
        # ===== Game Metrics / 对局指标 =====
        .add_panel(
            name="存活步数",
            name_en="episode_steps",
            type="line",
        )
        .add_metric(
            metrics_name="episode_steps",
            expr="avg(episode_steps{})",
        )
        .end_panel()
        .add_panel(
            name="宝箱得分",
            name_en="treasure_score",
            type="line",
        )
        .add_metric(
            metrics_name="treasure_score",
            expr="avg(treasure_score{})",
        )
        .end_panel()
        .add_panel(
            name="总得分",
            name_en="total_score",
            type="line",
        )
        .add_metric(
            metrics_name="total_score",
            expr="avg(total_score{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
