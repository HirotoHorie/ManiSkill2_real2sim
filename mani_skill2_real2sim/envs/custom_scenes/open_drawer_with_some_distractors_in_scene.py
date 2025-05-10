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
from .open_drawer_with_distractors_in_scene import OpenDrawerWithDistractorsSceneEnv

@register_env("OpenTopDrawerWithSomeDistractorsScene-v0", max_episode_steps=300)
class OpenTopDrawerWithSomeDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["top"]

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

            # world座標系への変換（drawer_posとdrawer_rot考慮）
            local_pos = np.array([x, y])
            world_xy = self.local_to_world_2d(local_pos, self.drawer_pos, self.drawer_rot)

            # 高さを設定
            obj_init_z = self.scene_table_height - 0.1  # 少し高めから落下

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

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする
    
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

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする

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

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする