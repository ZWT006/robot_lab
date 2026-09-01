"""Static contracts for the two quadruped-manipulator task families."""

from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPOT_ASSET = ROOT / "source/robot_lab/data/Robots/boston_dynamics/relic_spot_arm"
UMI_URDF = ROOT / "source/robot_lab/data/Robots/unitree/go2_arx5/urdf/go2_arx5_finray_x85_z94.urdf"
TASK_ROOT = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/loco_manipulation"

ARM_JOINTS = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1", "arm_f1x"]
LEG_COMMAND_JOINTS = [
    "fl_hx",
    "fl_hy",
    "fl_kn",
    "fr_hx",
    "fr_hy",
    "fr_kn",
    "hl_hx",
    "hl_hy",
    "hl_kn",
    "hr_hx",
    "hr_hy",
    "hr_kn",
]
LEG_ACTION_JOINTS = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]


def _assignment_literal(path: Path, name: str):
    module = ast.parse(path.read_text())
    for statement in module.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(statement.value)
    raise AssertionError(f"Assignment {name!r} not found in {path}")


def test_spot_urdf_has_expected_joint_contract_and_meshes():
    root = ET.parse(SPOT_ASSET / "spot_with_arm.urdf").getroot()
    moving_joints = [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]

    assert moving_joints == LEG_COMMAND_JOINTS + ARM_JOINTS
    assert len(moving_joints) == 19

    for mesh in root.findall(".//mesh"):
        assert (SPOT_ASSET / mesh.attrib["filename"]).is_file(), mesh.attrib["filename"]


def test_spot_policy_action_order_is_explicit():
    constants = ROOT / "source/robot_lab/robot_lab/assets/relic_spot_constants.py"
    assert _assignment_literal(constants, "LEG_ACTION_JOINT_NAMES") == LEG_ACTION_JOINTS


def test_relic_license_and_modification_notice_are_kept_with_asset():
    license_text = (SPOT_ASSET / "LICENSE.relic").read_text()
    notice_text = (SPOT_ASSET / "NOTICE").read_text()

    assert "non-commercial research purposes" in license_text
    assert "modified redistribution" in notice_text


def test_task_ids_are_separate_and_include_play_configs():
    umi_registration = (TASK_ROOT / "umi_on_legs/config/go2_arx5/__init__.py").read_text()
    relic_registration = (TASK_ROOT / "relic_interlimb/config/spot/__init__.py").read_text()

    assert "RobotLab-Isaac-UMI-On-Legs-Go2-ARX5-Play-v0" in umi_registration
    for phase in range(1, 5):
        assert f"RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-{phase}-v0" in relic_registration
    assert "RobotLab-Isaac-ReLIC-Spot-Interlimb-Play-v0" in relic_registration


def test_generic_play_script_has_no_loco_manipulation_task_switches():
    play_source = (ROOT / "scripts/reinforcement_learning/rsl_rl/play.py").read_text()

    assert "UMI-On-Legs" not in play_source
    assert "ReLIC-Spot" not in play_source
    assert "map_location=agent_cfg.device" in play_source


def test_umi_urdf_has_no_unused_foot_targets():
    assert "foot_target" not in UMI_URDF.read_text()
