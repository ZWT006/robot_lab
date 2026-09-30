

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