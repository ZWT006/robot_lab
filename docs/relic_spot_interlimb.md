# ReLIC Spot interlimb loco-manipulation

This task ports the ReLIC Spot + Arm environment to robot_lab's Isaac Sim 5.0,
Isaac Lab 2.2-compatible, Python 3.11, and RSL-RL 3.x stack. The port preserves
the 84-dimensional policy observation, 12-dimensional leg action, 19 actuated
robot joints, custom Spot knee torque limits, and Phase 1–4 training curriculum.

## License

The ReLIC-derived Python implementation and Spot + Arm assets are governed by
the RAI Institute Research License, not robot_lab's Apache-2.0 license. They may
be used only for non-commercial research. The redistributed license and change
notice are stored beside the asset at
`source/robot_lab/data/Robots/boston_dynamics/relic_spot_arm/`.

## Training

Train Phase 1 from scratch:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-1-v0 \
  --headless \
  --run_name=phase_1
```

Continue each later phase from the preceding phase's RSL-RL checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-2-v0 \
  --headless \
  --resume \
  --load_run=<phase-1-run-directory> \
  --checkpoint=model_<iteration>.pt \
  --run_name=phase_2
```

Repeat with `Phase-3-v0` and `Phase-4-v0`, always resuming from the preceding
phase. All phases use `logs/rsl_rl/spot_interlimb`, so the default robot_lab
checkpoint discovery works without a ReLIC-specific training script.

## Play and export

Use the ordinary robot_lab player with the final RSL-RL checkpoint. CPU policy
inference and CPU simulation are selected together by `--device=cpu`; Isaac Sim
may still initialize the GPU renderer for the interactive window.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-ReLIC-Spot-Interlimb-Play-v0 \
  --num_envs=1 \
  --device=cpu \
  --checkpoint="$(pwd)/logs/rsl_rl/spot_interlimb/<run-directory>/model_<iteration>.pt"
```

On load, the standard player exports `exported/policy.pt` and
`exported/policy.onnx` beside the checkpoint. The upstream ReLIC exported policy
is not treated as a resumable checkpoint; this port trains and resumes with the
same default RSL-RL checkpoint format as every other robot_lab task.
