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

## Training modes

| Task id | Actor input | EE force source | Runner |
|---|---|---|---|
| `Lining-Unified-v0` | 32 x 79 history | concurrent estimator (base vel, EE position, EE force, base force) | `LiningUnifiedRunner` |
| `Lining-Unified-Sensor-v0` | 5 x 82 history | measured EE force, 3-D, base yaw frame | standard `OnPolicyRunner` |

Both modes share the scene, command, reward, and critic (3 x 159). The sensor mode
logs to `logs/rsl_rl/lining_unified_sensor` and has a matching
`Lining-Unified-Sensor-Play-v0`.

The sensor (`mdp/ee_force_sensor.py`) mimics Kortex `tool_external_wrench_force`:
it reads the EE external force applied during the last policy step (not PhysX
contact forces), expresses it in the tool frame, and per episode randomizes

| Parameter | Training range | Play / deployment nominal |
|---|---|---|
| Bias | +/-1.5 N per axis | 0 |
| White-noise std | 0-0.5 N | 0 |
| Gain | 0.9-1.1 per axis | 1 |
| Delay | 0-2 policy steps | 1 step |
| First-order LPF cutoff (at 50 Hz) | 5-20 Hz | 10 Hz |

The filtered force is rotated into the base yaw frame (also the frame of the EE
force command) and scaled by 0.05. On hardware, filter the Kortex reading with
the nominal 10 Hz IIR at the policy rate, then rotate it with arm FK and the IMU
roll/pitch. The ranges come from the estimator they replace (checkpoint
2026-10-06/model_43000): about 0.25 N high-frequency noise and a slowly varying
error of about 1.2 N at zero force. The estimator's 0.55-0.6 gain shrinkage is a
regression artifact and is not reproduced.

Play draws the measured force as a green arrow at the EE next to the red
applied-force arrow.

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
