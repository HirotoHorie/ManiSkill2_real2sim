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
from .open_drawer_in_scene import OpenDrawerInSceneEnv

#-----------------------------------------------------------------------------------
#Closeクラス
#-----------------------------------------------------------------------------------

class CloseDrawerInSceneEnv(OpenDrawerInSceneEnv):

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        if "obj_init_options" not in options:
            options["obj_init_options"] = dict()
        if "cabinet_init_qpos" not in options["obj_init_options"]:
            scale = np.random.choice([1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1,1.11,1.12])
            options["obj_init_options"]["cabinet_init_qpos"] = 0.2 * scale
            self.init_drawer_pos = 0.2 * scale
        return super().reset(seed=seed, options=options)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        # print(f"qpos: {qpos}")
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        return dict(success=qpos <= 0.05, qpos=qpos, episode_stats=self.episode_stats)

    def get_language_instruction(self):
        return f"close {self.drawer_id} drawer"

@register_env("CloseDrawerCustomInScene-v0", max_episode_steps=300)
class CloseDrawerCustomInSceneEnv(CloseDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    drawer_ids = ["top", "middle", "bottom"]

@register_env("CloseTopDrawerCustomInScene-v0", max_episode_steps=300)
class CloseTopDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["top"]

@register_env("CloseMiddleDrawerCustomInScene-v0", max_episode_steps=300)
class CloseMiddleDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["middle"]

@register_env("CloseBottomDrawerCustomInScene-v0", max_episode_steps=300)
class CloseBottomDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_ids = ["bottom"]