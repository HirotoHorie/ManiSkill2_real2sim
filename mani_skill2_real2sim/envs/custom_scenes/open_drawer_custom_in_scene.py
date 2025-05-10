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
            scale = np.random.choice([0.0,0.0,0.0,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.024,0.26,0.28])
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