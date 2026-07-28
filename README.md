# EvoVLA-v2

[English](README.md) | [简体中文](README.zh-CN.md)

## Overview

**EvoVLA-v2** rebuilds the EvoVLA line of work on [RLinf](https://github.com/RLinf/RLinf): distributed RL infrastructure for post-training Vision-Language-Action (VLA) policies.

Long-horizon manipulation fails in a familiar pattern. External feedback is **sparse**—on CALVIN, reward arrives only when a subtask succeeds—so credit assignment over dozens of steps is brittle. Policies that look strong **in domain** often collapse under mild shift (we stress this with **ABC → D**). At the same time, the signals we care about—progress, contact quality, “getting closer to the instruction”—are hard to write as a clean **hand-crafted reward**, while large **unlabeled or weakly labeled** video and language already exist. That gap motivates **self-supervised RL**: turn the real objective into **pretext tasks** that supply **dense intrinsic rewards**, then combine them with the sparse task signal.

Concretely, we plan a minimally invasive side path `rlinf/ssrl/` on π₀.₅ (OpenPI) + PPO:

- **\(r_{\mathrm{con}}\)** — language-anchored progress (temporal contrastive / InfoNCE, R3M–VIP style; stop-grad w.r.t. the policy).
- **\(r_{\mathrm{cur}}\)** — ICM-style curiosity on wrist latents (optional; typically off on real robots).

Config off ⇒ original RLinf behavior. Real-robot transfer reuses RLinf’s Franka stack (including upstream async PPO), with a conservative SSRL default (freeze the visual encoder, curiosity off).

Upstream RLinf docs remain the reference for shared infrastructure; this README only states the EvoVLA-v2 story.
