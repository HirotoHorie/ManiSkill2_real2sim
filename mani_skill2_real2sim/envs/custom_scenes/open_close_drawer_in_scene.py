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
from .open_drawer_custom_in_scene import OpenDrawerCustomInSceneEnv


#-----------------------------------------------------------------------------------
#OpenCloseクラス
#-----------------------------------------------------------------------------------
class OpenCloseDrawerInSceneEnv(OpenDrawerCustomInSceneEnv, CustomOtherObjectsInSceneEnv):
    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        if qpos >= 0.2 and self.num == None: #qposが0.2以上かつopenのタスク中だったら
            self.num = 1 #openできたら1にする
        return dict(success = self.num == 1 and qpos <= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self, **kwargs):
        if self.num == None:
            # print(f"open {self.drawer_id} drawer")
            return f"open {self.drawer_id} drawer"
        # print(f"close {self.drawer_id} drawer")
        return f"close {self.drawer_id} drawer"
        # return "move the arm away from the drawer"
    
    def get_task_progress(self):
        if self.num == None:
            return 0.0
        elif self.num == 1:
            return 1.0
        else:
            return 0.5

@register_env("OpenCloseTopDrawerInScene-v0", max_episode_steps=300)
class OpenCloseTopDrawerInSceneEnv(OpenCloseDrawerInSceneEnv):
    drawer_ids = ["top"]

@register_env("OpenCloseMiddleDrawerInScene-v0", max_episode_steps=300)
class OpenCloseMiddleDrawerInSceneEnv(OpenCloseDrawerInSceneEnv):
    drawer_ids = ["middle"]

@register_env("OpenCloseBottomDrawerInScene-v0", max_episode_steps=300)
class OpenCloseBottomDrawerInSceneEnv(OpenCloseDrawerInSceneEnv):
    drawer_ids = ["bottom"]