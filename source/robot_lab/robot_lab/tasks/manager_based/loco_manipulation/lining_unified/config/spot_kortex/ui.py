# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Play-mode visualization controls for the lining-unified environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim
import omni.kit.app

from isaaclab.envs.ui import ManagerBasedRLEnvWindow

if TYPE_CHECKING:
    import omni.ui

    from isaaclab.envs import ManagerBasedRLEnv


class LiningUnifiedEnvWindow(ManagerBasedRLEnvWindow):
    """Add independent target-pose and external-force visualization controls."""

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        self._lining_unified_command = env.command_manager.get_term("lining_unified")

        with self.ui_window_elements["debug_frame"]:
            with self.ui_window_elements["debug_vstack"]:
                omni.ui.Separator(height=4)
                self._create_debug_vis_ui_element("target_ee_pose", self._lining_unified_command)
                self._build_external_force_debug_control()

    def _build_external_force_debug_control(self):
        """Build the checkbox for applied external-force arrows."""
        from omni.kit.window.extensions import SimpleCheckBox

        with omni.ui.HStack():
            omni.ui.Label(
                "External Force",
                width=isaacsim.gui.components.ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip="Show arrows for forces applied at the base and end effector; length scales with force magnitude.",
            )
            self.ui_window_elements["external_force_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=True,
                checked=self._lining_unified_command.cfg.external_force_debug_vis,
                on_checked_fn=self._lining_unified_command.set_external_force_debug_vis,
            )
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()
