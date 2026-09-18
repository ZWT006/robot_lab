# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Play-mode UI controls for Spot + Kortex trajectory visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim
import omni.kit.app

from isaaclab.envs.ui import ManagerBasedRLEnvWindow

if TYPE_CHECKING:
    import omni.ui

    from isaaclab.envs import ManagerBasedRLEnv


class DiscardLiningLegsEnvWindow(ManagerBasedRLEnvWindow):
    """Add EE target and future-trajectory controls to the standard environment window."""

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        self._ee_trajectory_command = env.command_manager.get_term("ee_trajectory")

        with self.ui_window_elements["debug_frame"]:
            with self.ui_window_elements["debug_vstack"]:
                omni.ui.Separator(height=4)
                self._create_debug_vis_ui_element("ee_targets", self._ee_trajectory_command)
                self._build_trajectory_debug_controls()

    def _build_trajectory_debug_controls(self):
        """Build controls for one selected environment's future EE target frames."""
        from omni.kit.window.extensions import SimpleCheckBox

        with omni.ui.HStack():
            omni.ui.Label(
                "EE Trajectory",
                width=isaacsim.gui.components.ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip="Show future end-effector target frames for the selected environment.",
            )
            self.ui_window_elements["ee_trajectory_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=True,
                checked=self._ee_trajectory_command.cfg.trajectory_debug_vis,
                on_checked_fn=self._ee_trajectory_command.set_trajectory_debug_vis,
            )
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()

        env_index_model = isaacsim.gui.components.ui_utils.int_builder(
            label="Trajectory Env",
            default_val=self._ee_trajectory_command.cfg.trajectory_debug_env_index + 1,
            min=1,
            max=self.env.num_envs,
            tooltip="Environment whose future EE targets are displayed. The UI index is one-based.",
        )
        env_index_model.add_value_changed_fn(self._set_trajectory_env_index)
        self.ui_window_elements["ee_trajectory_env_index"] = env_index_model

        preview_length_model = isaacsim.gui.components.ui_utils.float_builder(
            label="Preview Length [s]",
            default_val=self._ee_trajectory_command.cfg.trajectory_debug_preview_length_s,
            min=0.0,
            max=self.env.max_episode_length_s,
            step=0.1,
            format="%.1f",
            tooltip="Future time horizon displayed as a sequence of target axes.",
        )
        preview_length_model.add_value_changed_fn(self._set_trajectory_preview_length)
        self.ui_window_elements["ee_trajectory_preview_length"] = preview_length_model

    def _set_trajectory_env_index(self, model: omni.ui.AbstractValueModel):
        self._ee_trajectory_command.set_trajectory_debug_env_index(model.as_int - 1)

    def _set_trajectory_preview_length(self, model: omni.ui.AbstractValueModel):
        self._ee_trajectory_command.set_trajectory_debug_preview_length(model.as_float)
