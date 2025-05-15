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

#-----------------------------------------------------------------------------------
class CustomSomeDistractorsSceneEnv(OpenDrawerWithDistractorsSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        print("qpos:", qpos)
        print("self.num:", self.num)
        if qpos >= 0.2 and self.num == None: #qposが0.2以上かつopenのタスク中だったら
            self.num = 1 #openできたら1にする
        return dict(success = self.num == 1 and qpos >= 0.2, qpos=qpos, episode_stats=self.episode_stats)
    
    def get_task_progress(self):
        if self.num == None:
            return 0.0
        elif self.num == 1:
            return 1.0
        else:
            return 0.5

    def _initialize_actors(self):
        assert hasattr(self, "objects"), "self.objects が必要です！"
        assert hasattr(self, "current_episode_data"), "self.current_episode_data が必要です！"

        distractors = self.current_episode_data["distractors"]
        assert len(distractors) == len(self.objects), "distractors数とobjects数が一致していません！"

        # オブジェクトと設定情報をペアにして処理
        for obj, distractor_info in zip(self.objects, distractors):
            x = distractor_info["x"]
            y = distractor_info["y"]
            drawer = distractor_info["drawer"]

            # 各引き出しごとのZ座標設定
            if drawer == "top":
                obj_init_z = self.scene_table_height - 0.1
            elif drawer == "middle":
                obj_init_z = self.scene_table_height - 0.25
            elif drawer == "bottom":
                obj_init_z = self.scene_table_height - 0.5
            elif drawer == "tabletop":
                obj_init_z = self.scene_table_height + 0.5
            else:
                raise ValueError(f"Unknown drawer: {drawer}")

            # ワールド座標への変換
            local_pos = np.array([x, y])
            world_xy = self.local_to_world_2d(local_pos, self.drawer_pos, self.drawer_rot)

            # 回転設定
            rot_degree = distractor_info["rot"]
            rot_radian = np.deg2rad(rot_degree)
            obj_init_rot_quat = euler2quat(0, 0, rot_radian)

            # ポーズ設定
            p = np.hstack([world_xy, obj_init_z])
            q = obj_init_rot_quat
            obj.set_pose(sapien.Pose(p, q))

            # 落下のためにxy回転をロック
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # ロボットを遠ざける
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        # 物体を一斉に落とす
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

            self.objects.append(obj)

        # 🛠️ ターゲットオブジェクトをself.objにも設定（必須！）
        self.obj = self.objects[0]  # 最初の1個を代表にする

@register_env("OpenTopDrawerCustomWithDistractorsInScene-v0", max_episode_steps=300)
class OpenTopDrawerCustomWithDistractorsInSceneEnv(CustomSomeDistractorsSceneEnv):
    drawer_ids = ["top"]

@register_env("OpenMiddleDrawerCustomWithDistractorsInScene-v0", max_episode_steps=300)
class OpenTopDrawerCustomWithDistractorsInSceneEnv(CustomSomeDistractorsSceneEnv):
    drawer_ids = ["middle"]

@register_env("OpenBottomDrawerCustomWithDistractorsInScene-v0", max_episode_steps=300)
class OpenTopDrawerCustomWithDistractorsInSceneEnv(CustomSomeDistractorsSceneEnv):
    drawer_ids = ["bottom"]