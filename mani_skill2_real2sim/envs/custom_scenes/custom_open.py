from collections import OrderedDict
from typing import List, Optional
import json
import numpy as np
import cv2
import sapien.core as sapien
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import get_entity_by_name
from mani_skill2_real2sim.utils.common import random_choice
from transforms3d.euler import euler2quat
from mani_skill2_real2sim.utils.sapien_utils import (
    get_pairwise_contacts,
    compute_total_impulse,
)

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv


class OpenDrawerInSceneEnv(CustomSceneEnv):
    drawer_ids: List[str]

    def __init__(
        self,
        light_mode: Optional[str] = None,
        force_advance_subtask_time_steps: int = 100,
        camera_mode: Optional[str] = None,
        station_name: str = None,
        cabinet_joint_friction: float = 0.05,
        prepackaged_config: bool = None,
        drawer_pos: List[float] = [-0.295, 0.0, 0.017],
        drawer_quat: List[float] = [0, 0, 0], #クォータニオン
        drawer_rot: List[float] = [0, 0, 0], #オイラー角
        episode_id: int = 0,
        **kwargs,
    ):
        self.light_mode = light_mode
        self.camera_mode = camera_mode
        self.station_name = station_name
        self.cabinet_joint_friction = cabinet_joint_friction
        self.episode_stats = None
        self.drawer_id = None
        self.num = None
        # self.drawer_pos = drawer_pos
        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None
        self.obj = None
        self.obj_init_options = {}
        self.prepackaged_config = prepackaged_config
        self.light_mode = light_mode
        self.camera_mode = camera_mode
        self.station_name = station_name
        self.cabinet_joint_friction = cabinet_joint_friction
        self.episode_stats = None
        self.drawer_id = None
        self.num = None
        self.drawer_pos = drawer_pos
        self.drawer_quat = drawer_quat
        self.drawer_rot = drawer_rot
        self.init_drawer_pos = None
        self.episode_id = episode_id
        print(kwargs["model_ids"])
        # 例
        with open("distractors_pos.json", "r") as f:
            episode_list = json.load(f)

        # たとえば id=3 のエピソードをやりたいなら
        self.current_episode_data = episode_list[self.episode_id-1]

        self.force_advance_subtask_time_steps = force_advance_subtask_time_steps

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
        # filename = str(self.asset_root / "color_2.urdf")

        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        self.art_obj.name = 'cabinet'
        print("cabinet loaded")
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose(self.drawer_pos, self.drawer_quat))
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
        # idx_chosen = self._episode_rng.choice(len(overlay_ids))
        idx_chosen = 3


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
        # print(f"qpos: {qpos}") 
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(success = qpos >= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    # def get_language_instruction(self, **kwargs):
    #     return f"open {self.drawer_id} drawer"

    # def get_language_instruction(self, **kwargs):
    #     if self.station_name == "color_1":
    #         return f"open brown {self.drawer_id} drawer"
    #     elif self.station_name == "color_2":
    #         return f"open pink {self.drawer_id} drawer"
    #     elif self.station_name == "color_3":
    #         return f"open beige {self.drawer_id} drawer"
    
    # def get_language_instruction(self, **kwargs):
    #     if self.station_name == "color_1":
    #         return f"open brown {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_2":
    #         return f"open pink {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_3":
    #         return f"open beige {self.drawer_id} drawer with a small handle"

    def get_language_instruction(self, **kwargs):
        if self.station_name == "color_1":
            if self.drawer_pos[1] == 0:
                return f"open brown {self.drawer_id} drawer in the center"
            elif self.drawer_pos[1] == 0.1:
                return f"open brown {self.drawer_id} drawer on the right"
            else:
                return f"open brown {self.drawer_id} drawer on the left"
        elif self.station_name == "color_2":
            if self.drawer_pos[1] == 0:
                return f"open pink {self.drawer_id} drawer in the center"
            elif self.drawer_pos[1] == 0.1:
                return f"open pink {self.drawer_id} drawer on the right"
            else:
                return f"open pink {self.drawer_id} drawer on the left"
        elif self.station_name == "color_3":
            if self.drawer_pos[1] == 0:
                return f"open beige {self.drawer_id} drawer in the center"
            elif self.drawer_pos[1] == 0.1:
                return f"open beige {self.drawer_id} drawer on the right"
            else:
                return f"open beige {self.drawer_id} drawer on the left"


#------------------------------------------------------------------------------------
#Openクラス
#------------------------------------------------------------------------------------
@register_env("OpenDrawerCustomInScene-v0", max_episode_steps=300)
class OpenDrawerCustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top","middle","bottom"]

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        if "obj_init_options" not in options:
            options["obj_init_options"] = dict()
        if "cabinet_init_qpos" not in options["obj_init_options"]:
            #キャビネットの初期位置をランダムに設定
            scale = np.random.choice([0.0,0.0,0.0,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.024,0.26,0.28])
            print(scale)
            options["obj_init_options"]["cabinet_init_qpos"] = 0.2 * scale
        return super().reset(seed=seed, options=options)

@register_env("OpenTopDrawerInScene-v0", max_episode_steps=300)
class OpenTopDrawerInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

@register_env("OpenMiddleDrawerInScene-v0", max_episode_steps=300)
class OpenMiddleDrawerInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

@register_env("OpenBottomDrawerInScene-v0", max_episode_steps=300)
class OpenBottomDrawerInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]

#-----------------------------------------------------------------------------------
#distractorを使うクラス
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#Openwithdistクラス
#-----------------------------------------------------------------------------------
class OpenDrawerWithDistractorsSceneEnv(OpenDrawerCustomInSceneEnv, CustomOtherObjectsInSceneEnv):

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = (
            0.005
        )  # avoid "false-positive" collisions with other objects
        return scene_config
    
    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly from self.model_ids."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)

        self.obj = self._build_actor_helper(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        print(self.obj.name)
    
    def _load_actors(self):
        super()._load_actors()
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _initialize_actors(self):
        """子クラスでオーバーライドする必要がある"""
        pass

    def local_to_world_2d(self, local_xy, drawer_pos, drawer_quat):
        """
        引き出しの重心から見たローカル座標 (local_xy) をワールド座標に変換する
        :param local_xy: 引き出し基準での相対位置 (x, y)
        :param drawer_pos: 引き出しの重心のワールド位置 [x, y, z]
        :param drawer_quat: オイラー角 [x, y, z_deg]（クォータニオンじゃなくオイラー角として使う前提）
        """
        drawer_xy = np.array(drawer_pos[:2])
        theta_rad = np.deg2rad(drawer_quat[2])

        # 正しい 2D 回転行列
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ])

        # 回転 + 平行移動
        world_xy = R @ local_xy + drawer_xy
        return world_xy
    
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
        # idx_chosen = self._episode_rng.choice(len(overlay_ids))
        idx_chosen = 3


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

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        self.set_episode_rng(seed)
        # set objects
        print(options)
        self.obj_init_options = options.get("obj_init_options", {})
        print(self.obj_init_options)
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure

        obs, info = super().reset(seed=self._episode_seed, options=options)
        self.drawer_link: sapien.Link = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        self.drawer_collision = self.drawer_link.get_collision_shapes()[2]

        return obs, info

    def _initialize_episode_stats(self):
        self.cur_subtask_id = 0 # 0: open drawer, 1: place object into drawer
        self.episode_stats = OrderedDict(
            qpos=0.0, is_drawer_open=False, has_contact=0
        )

    def advance_to_next_subtask(self):
        self.cur_subtask_id = 1

    def step(self, action):
        if self._elapsed_steps >= self.force_advance_subtask_time_steps:
            # force advance to the next subtask
            self.advance_to_next_subtask()
        return super().step(action)

@register_env("OpenTopDrawerWithDistractorsScene-v0", max_episode_steps=300)
class OpenTopDrawerWithDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["top"]

    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            self._main_seed = None
            self.set_main_rng(None)
            self.set_episode_rng(None)
            obj_init_xy = self._episode_rng.uniform([0.15, -0.2], [0.23, 0.2], [2])
            obj_init_xy = [0.15,-0.2]
            obj_init_xy = self.local_to_world_2d(obj_init_xy, self.drawer_pos, self.drawer_rot)
            print(obj_init_xy)
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z - 0.1  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
            init_rand_axis_rot_range := self.obj_init_options.get(
                "init_rand_axis_rot_range", 0.0
            )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

@register_env("CustomSomeDistractorsScene-v0", max_episode_steps=300)
class CustomSomeDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["top","middle","bottom"]

    def _initialize_actors(self):
        """
        top/middle/bottom すべての drawer に同時に物体を落下配置する
        """

        assert hasattr(self, "objects"), "self.objects が必要です！"
        assert hasattr(self, "current_episode_data"), "self.current_episode_data が必要です！"

        distractors = self.current_episode_data["distractors"]
        assert len(distractors) == len(self.objects), "distractors数とobjects数が一致していません！"

        for obj, distractor_info in zip(self.objects, distractors):
            x = distractor_info["x"]
            y = distractor_info["y"]
            drawer = distractor_info["drawer"]

            # 各引き出しに応じた z 座標を設定
            if drawer == "top":
                obj_init_z = self.scene_table_height - 0.1
            elif drawer == "middle":
                obj_init_z = self.scene_table_height - 0.25
            elif drawer == "bottom":
                obj_init_z = self.scene_table_height - 0.35
            else:
                raise ValueError(f"Unknown drawer: {drawer}")

            # 回転を設定
            rot_degree = distractor_info["rot"]
            rot_radian = np.deg2rad(rot_degree)
            obj_init_rot_quat = euler2quat(0, 0, rot_radian)

            # pose を設定
            p = np.array([x, y, obj_init_z])
            q = obj_init_rot_quat
            obj.set_pose(sapien.Pose(p, q))

            # 落下のためにxy回転をロック
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # ロボットを遠ざける
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # 一斉に落とす
        self._settle(0.5)

        # ロック解除
        for obj in self.objects:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        self._settle(0.5)

        # 完全停止確認
        for obj in self.objects:
            lin_vel = np.linalg.norm(obj.velocity)
            ang_vel = np.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(1.5)

        # 最終高さ記録
        self.objects_height_after_settle = [obj.pose.p[2] for obj in self.objects]
    
    def _load_model(self):
        """
        3個のオブジェクトをロードして、self.objectsに保存し、さらに1個目をself.objに設定する
        """
        self.objects = []
        model_ids = [d["name"] for d in self.current_episode_data["distractors"]]

        for model_id in model_ids:
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=self.model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            print(f"Loaded object: {obj.name}")

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする

@register_env("OpenMiddleDrawerWithDistractorsScene-v0", max_episode_steps=300)
class OpenMiddleDrawerWithDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["middle"]
    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            self._main_seed = None
            self.set_main_rng(None)
            self.set_episode_rng(None)
            obj_init_xy = self._episode_rng.uniform([0.15, -0.2], [0.22, 0.2], [2])
            # obj_init_xy = [0.15,0.15]
            obj_init_xy = self.local_to_world_2d(obj_init_xy, self.drawer_pos, self.drawer_rot)
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z - 0.25  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
            init_rand_axis_rot_range := self.obj_init_options.get(
                "init_rand_axis_rot_range", 0.0
            )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]
    

@register_env("OpenMiddleDrawerWithSomeDistractorsScene-v0", max_episode_steps=300)
class OpenMiddleDrawerWithSomeDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["middle"]

    def _initialize_actors(self):
        """
        self.current_episode_data['distractors'] に従って物体を正確に配置する
        """

        assert hasattr(self, "objects"), "self.objects が必要です！"
        assert hasattr(self, "current_episode_data"), "self.current_episode_data が必要です！"

        distractors = self.current_episode_data["distractors"]
        assert len(distractors) == len(self.objects), "distractors数とobjects数が一致していません！"

        for obj, distractor_info in zip(self.objects, distractors):
            # 位置を取得
            x = distractor_info["x"]
            y = distractor_info["y"]
            print(x)
            print(y)

            # world座標系への変換（drawer_posとdrawer_rot考慮）
            local_pos = np.array([x, y])
            world_xy = self.local_to_world_2d(local_pos, self.drawer_pos, self.drawer_rot)

            # 高さを設定
            obj_init_z = self.scene_table_height - 0.25  # 少し高めから落下

            # 回転を設定
            rot_degree = distractor_info["rot"]
            rot_radian = np.deg2rad(rot_degree)
            obj_init_rot_quat = euler2quat(0, 0, rot_radian)

            # ポーズをまとめる
            p = np.hstack([world_xy, obj_init_z])
            q = obj_init_rot_quat

            # 物体を配置
            obj.set_pose(sapien.Pose(p, q))

            # 落下のためにxy回転をロック
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # ロボットを遠ざける
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # 全部一緒に落とす
        self._settle(0.5)

        # ロック解除
        for obj in self.objects:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        self._settle(0.5)

        # 完全停止確認
        for obj in self.objects:
            lin_vel = np.linalg.norm(obj.velocity)
            ang_vel = np.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(1.5)

        # 最後に高さを記録
        self.objects_height_after_settle = [obj.pose.p[2] for obj in self.objects]
    
    def _load_model(self):
        """
        3個のオブジェクトをロードして、self.objectsに保存し、さらに1個目をself.objに設定する
        """
        self.objects = []
        model_ids = [d["name"] for d in self.current_episode_data["distractors"]]

        for model_id in model_ids:
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=self.model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            print(f"Loaded object: {obj.name}")

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする

@register_env("OpenBottomDrawerWithDistractorsScene-v0", max_episode_steps=300)
class OpenBottomDrawerWithDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["bottom"]
    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            self._main_seed = None
            self.set_main_rng(None)
            self.set_episode_rng(None)
            obj_init_xy = self._episode_rng.uniform([0.15, -0.2], [0.22, 0.2], [2])
            # obj_init_xy = [0.15,0.15]
            obj_init_xy = self.local_to_world_2d(obj_init_xy, self.drawer_pos, self.drawer_rot)
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z - 0.5  # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", [1, 0, 0, 0])
        p = np.hstack([obj_init_xy, obj_init_z])
        q = obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (
            init_rand_axis_rot_range := self.obj_init_options.get(
                "init_rand_axis_rot_range", 0.0
            )
        ) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # Lock rotation around x and y to let the target object fall onto the table
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record the object height after it settles
        self.obj_height_after_settle = self.obj.pose.p[2]

@register_env("OpenBottomDrawerWithSomeDistractorsScene-v0", max_episode_steps=300)
class OpenBottomDrawerWithSomeDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["bottom"]

    def _initialize_actors(self):
        """
        self.current_episode_data['distractors'] に従って物体を正確に配置する
        """

        assert hasattr(self, "objects"), "self.objects が必要です！"
        assert hasattr(self, "current_episode_data"), "self.current_episode_data が必要です！"

        distractors = self.current_episode_data["distractors"]
        assert len(distractors) == len(self.objects), "distractors数とobjects数が一致していません！"

        for obj, distractor_info in zip(self.objects, distractors):
            # 位置を取得
            x = distractor_info["x"]
            y = distractor_info["y"]
            print(x)
            print(y)

            # world座標系への変換（drawer_posとdrawer_rot考慮）
            local_pos = np.array([x, y])
            world_xy = self.local_to_world_2d(local_pos, self.drawer_pos, self.drawer_rot)

            # 高さを設定
            obj_init_z = self.scene_table_height - 0.5  # 少し高めから落下

            # 回転を設定
            rot_degree = distractor_info["rot"]
            rot_radian = np.deg2rad(rot_degree)
            obj_init_rot_quat = euler2quat(0, 0, rot_radian)

            # ポーズをまとめる
            p = np.hstack([world_xy, obj_init_z])
            q = obj_init_rot_quat

            # 物体を配置
            obj.set_pose(sapien.Pose(p, q))

            # 落下のためにxy回転をロック
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # ロボットを遠ざける
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # 全部一緒に落とす
        self._settle(0.5)

        # ロック解除
        for obj in self.objects:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        self._settle(0.5)

        # 完全停止確認
        for obj in self.objects:
            lin_vel = np.linalg.norm(obj.velocity)
            ang_vel = np.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(1.5)

        # 最後に高さを記録
        self.objects_height_after_settle = [obj.pose.p[2] for obj in self.objects]
    def _load_model(self):
        """
        3個のオブジェクトをロードして、self.objectsに保存し、さらに1個目をself.objに設定する
        """
        self.objects = []
        model_ids = [d["name"] for d in self.current_episode_data["distractors"]]

        for model_id in model_ids:
            density = self.model_db[model_id].get("density", 1000)

            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=self.model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            print(f"Loaded object: {obj.name}")

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする