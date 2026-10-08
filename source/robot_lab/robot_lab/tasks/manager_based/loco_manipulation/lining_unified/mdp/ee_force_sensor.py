# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""End-effector force sensor model matching the Kortex ``tool_external_wrench`` interface.

Kortex Gen3 reports the external tool force (``BaseFeedback.tool_external_wrench_force_{x,y,z}``)
estimated from its joint torque sensors. Instead of reading PhysX contact forces, the model
starts from the external force actually applied at the EE during the last policy step and
corrupts it in the tool frame like a real sensor stream::

    y_t   = gain * F_t + bias + noise_t     # tool frame, per-episode gain/bias/noise std
    y_d   = y_{t - delay}                   # 0-2 policy steps
    F_obs = LPF(y_d)                        # first-order IIR with a per-episode cutoff

The filtered force is then rotated into the base yaw frame (the frame of the EE force
command), which on hardware only needs arm FK and the IMU roll/pitch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import LiningUnifiedCommand


@configclass
class EEForceSensorCfg:
    """Per-episode randomization ranges of :class:`EEForceSensorModel`.

    Defaults follow the EE-force error of the concurrent state estimator they replace
    (``Lining-Unified-v0`` checkpoint 2026-10-06/model_43000, forces always on): ~0.25 N
    high-frequency noise, ~1.2 N slowly varying error at zero force, and per-episode offsets
    up to ~0.8 N (p95). The estimator's 0.55-0.6 gain shrinkage is a regression artifact, not
    a sensor property, so the gain range stays near one.
    """

    obs_scale: float = 0.05
    """Scale applied to the yaw-frame force [N] before it enters the observation."""
    force_limit: float = 50.0
    """Symmetric clip of the reported force per axis [N]."""
    bias_range: tuple[float, float] = (-1.5, 1.5)
    """Constant per-axis offset [N]."""
    noise_std_range: tuple[float, float] = (0.0, 0.5)
    """Std of the white noise added before filtering [N]."""
    gain_range: tuple[float, float] = (0.9, 1.1)
    """Per-axis scale error."""
    delay_steps_range: tuple[int, int] = (0, 2)
    """Inclusive range of the transport delay in policy steps."""
    cutoff_hz_range: tuple[float, float] = (5.0, 20.0)
    """Cutoff of the first-order low-pass filter running at the policy rate [Hz]."""

    debug_vis: bool = False
    """Draw the measured force at the EE (green) next to the command's applied-force arrow."""
    arrow_scale: float = 0.01
    """Arrow length in meters per newton."""
    visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/lining_unified/measured_ee_force"
    )
    visualizer_cfg.markers["arrow"].scale = (1.0, 0.05, 0.05)


EE_FORCE_SENSOR_NOMINAL_CFG = EEForceSensorCfg(
    bias_range=(0.0, 0.0),
    noise_std_range=(0.0, 0.0),
    gain_range=(1.0, 1.0),
    delay_steps_range=(1, 1),
    cutoff_hz_range=(10.0, 10.0),
)
"""Deterministic sensor for play; deployment should filter with the same 10 Hz cutoff."""


class EEForceSensorModel:
    """Stateful 3-axis EE force sensor shared by all environments."""

    def __init__(self, cfg: EEForceSensorCfg, command: LiningUnifiedCommand, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.command = command
        self.robot = command.robot
        self.ee_body_id = command.ee_body_id
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self._dt = env.step_dt

        n, device = self.num_envs, self.device
        self.bias = torch.zeros(n, 3, device=device)
        self.noise_std = torch.zeros(n, 1, device=device)
        self.gain = torch.ones(n, 3, device=device)
        self.delay = torch.zeros(n, dtype=torch.long, device=device)
        self.alpha = torch.ones(n, 1, device=device)
        self._buffer = torch.zeros(n, cfg.delay_steps_range[1] + 1, 3, device=device)
        self._filtered_ee = torch.zeros(n, 3, device=device)
        self._needs_init = torch.ones(n, dtype=torch.bool, device=device)
        self._last_step = -1
        self._env_ids = torch.arange(n, device=device)
        self._visualizer = VisualizationMarkers(cfg.visualizer_cfg) if cfg.debug_vis else None
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None):
        """Resample the sensor parameters and restart the delay line and filter."""
        if env_ids is None:
            env_ids = self._env_ids
        count = len(env_ids)
        if count == 0:
            return
        cfg, device = self.cfg, self.device
        self.bias[env_ids] = torch.empty(count, 3, device=device).uniform_(*cfg.bias_range)
        self.noise_std[env_ids] = torch.empty(count, 1, device=device).uniform_(*cfg.noise_std_range)
        self.gain[env_ids] = torch.empty(count, 3, device=device).uniform_(*cfg.gain_range)
        low, high = cfg.delay_steps_range
        self.delay[env_ids] = torch.randint(low, high + 1, (count,), device=device)
        cutoff = torch.empty(count, 1, device=device).uniform_(*cfg.cutoff_hz_range)
        self.alpha[env_ids] = 1.0 - torch.exp(-2.0 * math.pi * cutoff * self._dt)
        self._needs_init[env_ids] = True

    def compute(self) -> torch.Tensor:
        """Return the measured force in the base yaw frame [N].

        The sensor advances once per environment step; repeated observation computations
        within a step (e.g. ``get_observations``) return the same reading.
        """
        ee_quat_w = self.robot.data.body_link_quat_w[:, self.ee_body_id]
        step = self._env.common_step_counter
        if step != self._last_step or self._needs_init.any():
            reading = self._corrupt(math_utils.quat_apply_inverse(ee_quat_w, self.command.applied_ee_external_force_w))
            if step != self._last_step:
                self._last_step = step
                self._buffer = torch.cat((reading.unsqueeze(1), self._buffer[:, :-1]), dim=1)
                delayed = self._buffer[self._env_ids, self.delay]
                self._filtered_ee += self.alpha * (delayed - self._filtered_ee)
            # Start a reset sensor at steady state instead of filtering up from zero.
            init_ids = self._needs_init.nonzero().flatten()
            if len(init_ids) > 0:
                self._buffer[init_ids] = reading[init_ids].unsqueeze(1)
                self._filtered_ee[init_ids] = reading[init_ids]
                self._needs_init[init_ids] = False

        limit = self.cfg.force_limit
        force_w = math_utils.quat_apply(ee_quat_w, self._filtered_ee.clamp(-limit, limit))
        force_yaw = math_utils.quat_apply_inverse(self.command.yaw_quat_w, force_w)
        if self._visualizer is not None:
            self._visualize(force_w)
        return force_yaw

    def _corrupt(self, force_ee: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(force_ee) * self.noise_std
        return self.gain * force_ee + self.bias + noise

    def _visualize(self, force_w: torch.Tensor):
        magnitudes = torch.linalg.norm(force_w, dim=-1)
        orientations = self.command._vector_to_arrow_orientation(force_w, magnitudes)
        scales = torch.ones_like(force_w)
        scales[:, 0] = magnitudes * self.cfg.arrow_scale
        scales[magnitudes <= 1.0e-6] = 0.0
        positions = self.robot.data.body_com_pos_w[:, self.ee_body_id]
        self._visualizer.visualize(positions, orientations, scales)
