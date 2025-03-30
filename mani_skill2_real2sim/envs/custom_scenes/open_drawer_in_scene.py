from collections import OrderedDict
from typing import List, Optional

import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from transforms3d.euler import euler2quat

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv


class OpenDrawerInSceneEnv(CustomSceneEnv):
    drawer_ids: List[str]

    def __init__(
        self,
        light_mode: Optional[str] = None,
        camera_mode: Optional[str] = None,
        station_name: str = None,
        cabinet_joint_friction: float = 0.05,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        self.light_mode = light_mode
        self.camera_mode = camera_mode
        self.station_name = station_name
        self.cabinet_joint_friction = cabinet_joint_friction
        self.episode_stats = None
        self.drawer_id = None
        self.num = None

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())


        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True

        return ret

    # def _get_default_scene_config(self):
    #     scene_config = super()._get_default_scene_config()
    #     scene_config.enable_pcm = True
    #     return scene_config

    def _initialize_agent(self):
        init_qpos = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.7851361,
            ]
        )
        if self.camera_mode == "variant":
            init_qpos[-2] += -0.025
            init_qpos[-1] += 0.008
        self.robot_init_options.setdefault("qpos", init_qpos)
        super()._initialize_agent()

    def _setup_lighting(self):
        if self.light_mode != "simple":
            return self._setup_lighting_legacy()

        self._scene.set_ambient_light([1.0, 1.0, 1.0])
        angle = 75
        self._scene.add_directional_light(
            [-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [1.0, 1.0, 1.0]
        )

    def _setup_lighting_legacy(self):
        # self.enable_shadow = True
        # super()._setup_lighting()

        direction = [-0.2, 0, -1]
        if self.light_mode == "vertical":
            direction = [-0.1, 0, -1]

        color = [1, 1, 1]
        if self.light_mode == "darker":
            color = [0.5, 0.5, 0.5]
        elif self.light_mode == "brighter":
            color = [2, 2, 2]

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            direction, color, shadow=True, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([-1, 1, -0.05], [0.5] * 3)
        self._scene.add_directional_light([-1, -1, -0.05], [0.5] * 3)

    def _load_actors(self):
        self._load_arena_helper(add_collision=False)

    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.0, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        reconfigure = options.get("reconfigure", False)
        self.set_episode_rng(seed)
        self.drawer_id = self._episode_rng.choice(self.drawer_ids)

        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        options["reconfigure"] = reconfigure

        self._initialize_episode_stats()

        obs, info = super().reset(seed=self._episode_seed, options=options) # articulations are loaded here
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

        # setup cabinet qpos
        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        cabinet_init_qpos = obj_init_options.get("cabinet_init_qpos", None)
        if cabinet_init_qpos is not None:
            if isinstance(cabinet_init_qpos, float):
                # set qpos for target cabinet joint
                tmp = [0.0] * self.art_obj.dof
                tmp[self.joint_idx] = cabinet_init_qpos
                cabinet_init_qpos = tmp
            self.art_obj.set_qpos(cabinet_init_qpos)
        else:
            self.art_obj.set_qpos([0.0] * self.art_obj.dof) # ensure that the drawer is closed
        obs = self.get_obs()

        info.update(
            {
                "drawer_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.drawer_obj.pose,
                "cabinet_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.art_obj.pose,
                "station_name": self.station_name,
                "light_mode": self.light_mode,
            }
        )
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged evaluation configs under visual matching setup
        overlay_ids = ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]
        rgb_overlay_paths = [
            str(ASSET_DIR / f"real_inpainting/open_drawer_{i}.png") for i in overlay_ids
        ]
        robot_init_xs = [0.644, 0.765, 0.889, 0.652, 0.752, 0.851, 0.665, 0.765, 0.865]
        robot_init_ys = [
            -0.179,
            -0.182,
            -0.203,
            0.009,
            0.009,
            0.035,
            0.224,
            0.222,
            0.222,
        ]
        robot_init_rotzs = [-0.03, -0.02, -0.06, 0, 0, 0, 0, -0.025, -0.025]
        idx_chosen = self._episode_rng.choice(len(overlay_ids))

        options["robot_init_options"] = {
            "init_xy": [robot_init_xs[idx_chosen], robot_init_ys[idx_chosen]],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, robot_init_rotzs[idx_chosen]))
                * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        self.rgb_overlay_path = rgb_overlay_paths[idx_chosen]
        self.rgb_overlay_img = (
            cv2.cvtColor(cv2.imread(rgb_overlay_paths[idx_chosen]), cv2.COLOR_BGR2RGB)
            / 255
        )
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(qpos=0.0)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(success = qpos >= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self, **kwargs):
        return f"open {self.drawer_id} drawer"


#------------------------------------------------------------------------------------
#デフォルトクラス
#------------------------------------------------------------------------------------
@register_env("OpenDrawerCustomInScene-v0", max_episode_steps=200)
class OpenDrawerCustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","middle","bottom"]
    
@register_env("OpenAndCloseDrawerCustomInScene-v0", max_episode_steps=200)
class OpenAndCloseDrawerCustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","middle","bottom"]
    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        if qpos >= 0.2 and self.num == None: #qposが0.2以上かつopenのタスク中だったら
            self.num = 1 #openできたら1にする
        return dict(success = self.num == 1 and qpos <= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self, **kwargs):
        if self.num == None:
            print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
            print(f"open {self.drawer_id} drawer")
            return f"open {self.drawer_id} drawer"
        print(f"close {self.drawer_id} drawer")
        return f"close {self.drawer_id} drawer"

@register_env("OpenTopDrawerCustomInScene-v0", max_episode_steps=113)
class OpenTopDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

@register_env("OpenMiddleDrawerCustomInScene-v0", max_episode_steps=113)
class OpenMiddleDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

@register_env("OpenBottomDrawerCustomInScene-v0", max_episode_steps=113)
class OpenBottomDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]

class CloseDrawerInSceneEnv(OpenDrawerInSceneEnv):

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        if "obj_init_options" not in options:
            options["obj_init_options"] = dict()
        if "cabinet_init_qpos" not in options["obj_init_options"]:
            options["obj_init_options"]["cabinet_init_qpos"] = 0.2
        return super().reset(seed=seed, options=options)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(success=qpos <= 0.05, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self):
        return f"close {self.drawer_id} drawer"

@register_env("CloseDrawerCustomInScene-v0", max_episode_steps=113)
class CloseDrawerCustomInSceneEnv(CloseDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

@register_env("CloseTopDrawerCustomInScene-v0", max_episode_steps=113)
class CloseTopDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

@register_env("CloseMiddleDrawerCustomInScene-v0", max_episode_steps=113)
class CloseMiddleDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

@register_env("CloseBottomDrawerCustomInScene-v0", max_episode_steps=113)
class CloseBottomDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]

#------------------------------------------------------------------------------------
#実験用の27クラス
#------------------------------------------------------------------------------------
@register_env("Color111CustomInScene-v0", max_episode_steps=200)
class Color111CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","middle","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret

@register_env("Color211CustomInScene-v0", max_episode_steps=200)
class Color211CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","middle","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    
@register_env("Color311CustomInScene-v0", max_episode_steps=200)
class Color311CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","middle","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    
@register_env("Color112CustomInScene-v0", max_episode_steps=200)
class Color112CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret

@register_env("Color212CustomInScene-v0", max_episode_steps=200)
class Color212CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    
@register_env("Color312CustomInScene-v0", max_episode_steps=200)
class Color312CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret

@register_env("Color113CustomInScene-v0", max_episode_steps=200)
class Color113CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret

@register_env("Color213CustomInScene-v0", max_episode_steps=200)
class Color213CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    
@register_env("Color313CustomInScene-v0", max_episode_steps=200)
class Color313CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret

@register_env("Color121CustomInScene-v0", max_episode_steps=200)
class Color121CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["middle","middle","middle"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color131CustomInScene-v0", max_episode_steps=200)
class Color131CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["middle","middle","middle"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color122CustomInScene-v0", max_episode_steps=200)
class Color122CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color132CustomInScene-v0", max_episode_steps=200)
class Color132CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color123CustomInScene-v0", max_episode_steps=200)
class Color123CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color133CustomInScene-v0", max_episode_steps=200)
class Color133CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_1"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")






@register_env("Color221CustomInScene-v0", max_episode_steps=200)
class Color221CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["middle","middle","middle"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color321CustomInScene-v0", max_episode_steps=200)
class Color321CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["middle","middle","middle"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color231CustomInScene-v0", max_episode_steps=200)
class Color231CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["middle","middle","middle"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color331CustomInScene-v0", max_episode_steps=200)
class Color331CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["middle","middle","middle"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color222CustomInScene-v0", max_episode_steps=200)
class Color222CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color322CustomInScene-v0", max_episode_steps=200)
class Color322CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color232CustomInScene-v0", max_episode_steps=200)
class Color232CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color332CustomInScene-v0", max_episode_steps=200)
class Color332CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","top","top"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")





@register_env("Color223CustomInScene-v0", max_episode_steps=200)
class Color223CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color233CustomInScene-v0", max_episode_steps=200)
class Color233CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_2"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color323CustomInScene-v0", max_episode_steps=200)
class Color323CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, 0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")

@register_env("Color333CustomInScene-v0", max_episode_steps=200)
class Color333CustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["bottom","bottom","bottom"]
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "dummy_drawer"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/open_drawer_a0.png"
        )  # dummy path; to be replaced later
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        ret["shader_dir"] = "rt"
        self.station_name = "color_3"
        self.light_mode = "simple"
        ret["disable_bad_material"] = True
        return ret
    def _load_articulations(self):

        filename = str(self.asset_root / f"{self.station_name}.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.345, -0.1, 0.017], [1, 0, 0, 0]))
        print("cabinet loaded1")
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(self.cabinet_joint_friction)
            joint.set_drive_property(stiffness=0, damping=1)

        self.drawer_obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = self.joint_names.index(f"{self.drawer_id}_drawer_joint")