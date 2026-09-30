# Unified Force

This task is a manager-based Isaac Lab reproduction of the UniFP
`b2z1_pos_force` environment. It preserves the released controller's main
training interfaces:

- 17 policy actions (12 B2 joints and the first 5 Z1 joints), with the two tool
  joints held at their default positions;
- a 15-D command containing base velocity, spherical end-effector pose,
  end-effector force, and base force;
- a 73-D actor frame stacked for 32 steps and a 149-D privileged critic frame
  stacked for 3 steps;
- a 12-D concurrent state-estimation target for base velocity, end-effector
  position, end-effector disturbance, and base disturbance;
- the UniFP history encoder, 64-D latent, estimator decoder, actor, critic,
  reward weights, force pulses, and domain randomization ranges.

Train from the repository root:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Unified-Force-v0 \
  --headless
```

For a small smoke test, override the environment and iteration counts:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Unified-Force-v0 \
  --num_envs 32 --max_iterations 2 --headless
```

Play a saved checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Unified-Force-Play-v0 \
  --num_envs 1 \
  --checkpoint /path/to/model.pt
```

The released project delays force-command and disturbance pulses until
`8000 * 24` policy steps. The training configuration keeps that schedule; the
play configuration starts the pulses immediately.
