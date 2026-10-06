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
    """Add visualization and runtime command controls for play mode."""

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        self._lining_unified_command = env.command_manager.get_term("lining_unified")

        with self.ui_window_elements["debug_frame"]:
            with self.ui_window_elements["debug_vstack"]:
                omni.ui.Separator(height=4)
                self._create_debug_vis_ui_element("target_ee_pose", self._lining_unified_command)
                self._build_bool_control(
                    label="Show External Force",
                    element_name="show_external_force",
                    checked=self._lining_unified_command.cfg.external_force_debug_vis,
                    callback=self._lining_unified_command.set_external_force_debug_vis,
                    tooltip="Show arrows whose direction and length represent the applied external forces.",
                )
                omni.ui.Separator(height=4)
                self._build_bool_control(
                    label="Mask Base Command",
                    element_name="mask_base_command",
                    checked=self._lining_unified_command.cfg.mask_base_command,
                    callback=self._lining_unified_command.set_mask_base_command,
                    tooltip="Set base velocity and virtual-force commands to zero for every environment.",
                )
                self._build_bool_control(
                    label="Refresh EE Target",
                    element_name="refresh_ee_target",
                    checked=self._lining_unified_command.cfg.refresh_ee_target,
                    callback=self._lining_unified_command.set_refresh_ee_target,
                    tooltip="Advance the local EE target trajectory and refresh its world pose around the moving base.",
                )
                self._build_bool_control(
                    label="Apply EE External Force",
                    element_name="apply_ee_external_force",
                    checked=self._lining_unified_command.cfg.apply_ee_external_force,
                    callback=self._lining_unified_command.set_apply_ee_external_force,
                    tooltip="Apply physical external-force pulses at the end effector.",
                )

    def _build_bool_control(self, label: str, element_name: str, checked: bool, callback, tooltip: str):
        """Build a checkbox backed by a command-term runtime setter."""
        from omni.kit.window.extensions import SimpleCheckBox

        with omni.ui.HStack():
            omni.ui.Label(
                label,
                width=isaacsim.gui.components.ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip=tooltip,
            )
            self.ui_window_elements[f"{element_name}_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=True,
                checked=checked,
                on_checked_fn=callback,
            )
            isaacsim.gui.components.ui_utils.add_line_rect_flourish()
