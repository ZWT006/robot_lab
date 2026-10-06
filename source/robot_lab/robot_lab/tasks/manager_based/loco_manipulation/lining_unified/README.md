# Lining Unified

UniFP-style unified position/force whole-body control on the Spot + Kortex
platform (`Lining_CFG`, shared with `discard_lining_legs`). The code is a copy of
`unified_force` adapted to the new robot; the B2 + Z1 task is untouched.

## Differences from `unified_force` (B2 + Z1)

| | B2 + Z1 | Spot + Kortex |
|---|---|---|
| Policy joints | 12 legs + 5 Z1 (`wrist_rotate`, gripper held fixed) | 12 legs + 7 Kortex (no fixed joints) |
| Actor frame / critic frame | 73 / 149 | 79 / 159 (`22 + 3N` / `64 + 5N`) |
| Arm mount / sphere center | base +(0.2, 0, 0.225) / (0.2, 0, 0.8) | body +(0, 0, 0.125) / (0, 0, 0.87) |
| EE tool axis / frame offset | +x / `Rx(pi/2)` | +z / `Ry(pi/2)` |
| EE force / base force | +/-60 N / +/-50 N | +/-20 N / +/-25 N |
| Base added mass | 0-15 kg | 0-7.5 kg |
| EE reward | position | position + 0.5 * orientation bonus |

Leg kinematics (`hip_x -> hip_y -> knee` about x, y, y), sign conventions, and
FL/FR/RL/RR order are identical on both robots, so the trot reference keeps
UniFP's joint indices. `arm_joint_1` turns about world -z (opposite to
`z1_waist`); no reward or observation depends on per-arm-joint signs.

## Commands

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Lining-Unified-v0 \
  --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Lining-Unified-Play-v0 \
  --num_envs 1 \
  --checkpoint /path/to/model.pt
```

Force pulses start after `8000 * 24` policy steps during training and
immediately in the play configuration.

In play mode, the **Scene Debug Visualization** panel provides independent
checkboxes for the target-pose and force-arrow visibility as well as three
runtime command controls:

- **Mask Base Command** zeros the base velocity and virtual-force commands.
- **Refresh EE Target** advances the local target trajectory and recomputes its
  world pose around the moving base. Turning it off freezes the current target
  pose in world coordinates.
- **Apply EE External Force** enables physical force pulses at the end effector.

The target marker always shows the nominal target without its force-compliance
offset. Force arrows are anchored at the base and end-effector centers and scale
at 1 cm per newton. The same behavior can be selected before launch with Hydra:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Lining-Unified-Play-v0 \
  --num_envs 1 \
  --checkpoint /path/to/model.pt \
  env.commands.lining_unified.mask_base_command=true \
  env.commands.lining_unified.refresh_ee_target=true \
  env.commands.lining_unified.apply_ee_external_force=false
```
