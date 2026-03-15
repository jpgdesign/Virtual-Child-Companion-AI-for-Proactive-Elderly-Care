---
title: Virtual Child Companion AI
emoji: 👵
colorFrom: amber
colorTo: teal
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
---

# Virtual Child Companion AI

This Space demonstrates a proactive elderly-care dialogue prototype driven by reinforcement learning.

## What it shows

- `DQN` as the default policy
- Switchable `Q-learning`
- Topic deviation detection and refocusing
- Slot filling for medication, sleep, activity, and diet
- Caregiver summary generation

## Files needed in the Space repo

Upload this folder as the root of a Hugging Face Space **together with**:

- `virtual_child_rl_system.py`
- `R_data.py`
- `dueling_dqn.py`
- `tabular_q_learning.py`
- `grandma_session_20250713_185829/`
- `rl_data_20250721_142929.json`
- `artifacts/runtime_models/`

If you sync from the GitHub repository, keep the same relative paths and the app will run directly.
