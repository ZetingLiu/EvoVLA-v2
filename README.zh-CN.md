# EvoVLA-v2

[English](README.md) | [简体中文](README.zh-CN.md)

## Overview

**EvoVLA-v2** 在 [RLinf](https://github.com/RLinf/RLinf) 之上重建 EvoVLA 方向：面向视觉-语言-动作（VLA）策略后训练的分布式强化学习底座。

长程操作里，失败往往按同一种故事展开。环境给出的外部反馈是**稀疏**的——在 CALVIN 上，往往只有子任务成功才有分——几十步里的信用分配因此很脆。策略在**同分布**场景里看起来不错，换到轻微偏移（我们用 **ABC → D** 压测）就掉下来。与此同时，真正想优化的东西——进度、接触质量、「是否更贴近指令」——很难写成干净的**显式奖励函数**；但现场并不缺**无标签或弱标签**的视频与语言。于是自然走到 **自监督强化学习（SSRL）**：把优化目标转成 **pretext task**，用它们造出**内部稠密奖励**，再与稀疏任务信号叠加。

具体落地上，我们计划在 π₀.₅（OpenPI）+ PPO 上加一条最小侵入旁路 `rlinf/ssrl/`：

- **\(r_{\mathrm{con}}\)**：语言锚点进度（时间对比 / InfoNCE，R3M–VIP 风格；对策略 stop-gradient）。
- **\(r_{\mathrm{cur}}\)**：腕部 latent 上的 ICM 式好奇心（可选；真机默认关闭）。

配置关闭即回退为原版 RLinf。真机迁移复用 RLinf 的 Franka 能力（含上游 async PPO），SSRL 取保守默认（冻视觉编码器、关 curiosity）。

共享基础设施仍以 RLinf 文档为准；本 README 只讲清 EvoVLA-v2 这条故事线。
