

```bash
# Tossing
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Lining-Legs-v0 \
  --headless \
  --run_name=move \
  env.commands.ee_trajectory.trajectory_file="$(pwd)/source/robot_lab/data/physical_ai/longer_pushing_trajs.pkl"

# Tossing 
  env.commands.ee_trajectory.trajectory_file="$(pwd)/source/robot_lab/data/umi_on_legs/tossing.pkl"
# Pushing
  env.commands.ee_trajectory.trajectory_file="$(pwd)/source/robot_lab/data/umi_on_legs/pushing.pkl"
# Cup rearrangement
  env.commands.ee_trajectory.trajectory_file="$(pwd)/source/robot_lab/data/umi_on_legs/umi_cup_in_wild.pkl"
```

```bash
tensorboard --logdir=logs/rsl_rl/lining_legs/ --port=6006
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Lining-Legs-Play-v0 \
  --num_envs=64 \
  --device=cpu \
  env.commands.ee_trajectory.trajectory_file="$(pwd)/source/robot_lab/data/physical_ai/shorter_pushing_trajs.pkl" \
  --checkpoint="$(pwd)/logs/rsl_rl/lining_legs/2026-09-17_22-17-35_move/model_4000.pt"
```


```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Unified-Force-v0 \
  --headless
```

```
tensorboard --logdir logs/rsl_rl/unified_force   --port 6006
```

```
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task RobotLab-Isaac-Unified-Force-Play-v0 \
  --num_envs 4 \
  --device cpu 
```

## Lining Unified (UniFP position/force on Spot + Kortex)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Lining-Unified-v0 \
  --headless
```

```bash
tensorboard --logdir logs/rsl_rl/lining_unified --port 6006
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Lining-Unified-Play-v0 \
  --num_envs 16 \
  --device cpu
```

EE-force-sensor variant (5-frame history, no state estimator):

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Lining-Unified-Sensor-v0 \
  --headless
```

```bash
tensorboard --logdir logs/rsl_rl/lining_unified_sensor --port 6006
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Lining-Unified-Sensor-Play-v0 \
  --num_envs 16 \
  --device cpu
```
