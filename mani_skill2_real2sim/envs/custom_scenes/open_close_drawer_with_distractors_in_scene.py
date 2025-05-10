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
from .open_drawer_with_distractors_in_scene import OpenTopDrawerWithDistractorsSceneEnv, OpenMiddleDrawerWithDistractorsSceneEnv, OpenBottomDrawerWithDistractorsSceneEnv

#-----------------------------------------------------------------------------------
#OpenClosewithdistクラス
#-----------------------------------------------------------------------------------
@register_env("OpenCloseTopDrawerWithDistractorsScene-v0", max_episode_steps=300)
class OpenCloseTopDrawerWithDistractorsSceneEnv(OpenTopDrawerWithDistractorsSceneEnv):
    drawer_ids = ["top"]
    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        if qpos >= 0.2 and self.num == None: #qposが0.2以上かつopenのタスク中だったら
            self.num = 1 #openできたら1にする
        return dict(success = self.num == 1 and qpos <= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         # print(f"open {self.drawer_id} drawer")
    #         return f"open {self.drawer_id} drawer"
    #     # print(f"close {self.drawer_id} drawer")
    #     return f"close {self.drawer_id} drawer"

    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             return f"open brown {self.drawer_id} drawer"
    #         elif self.station_name == "color_2":
    #             return f"open pink {self.drawer_id} drawer"
    #         elif self.station_name == "color_3":
    #             return f"open beige {self.drawer_id} drawer"
    #     if self.station_name == "color_1":
    #         return f"close brown {self.drawer_id} drawer"
    #     elif self.station_name == "color_2":
    #         return f"close pink {self.drawer_id} drawer"
    #     elif self.station_name == "color_3":
    #         return f"close beige {self.drawer_id} drawer"
        
    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             return f"open brown {self.drawer_id} drawer with a small handle"
    #         elif self.station_name == "color_2":
    #             return f"open pink {self.drawer_id} drawer with a small handle"
    #         elif self.station_name == "color_3":
    #             return f"open beige {self.drawer_id} drawer with a small handle"
    #     if self.station_name == "color_1":
    #         return f"close brown {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_2":
    #         return f"close pink {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_3":
    #         return f"close beige {self.drawer_id} drawer with a small handle"
        
    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open brown {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open brown {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open brown {self.drawer_id} drawer on the left"
    #         elif self.station_name == "color_2":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open pink {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open pink {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open pink {self.drawer_id} drawer on the left"
    #         elif self.station_name == "color_3":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open beige {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open beige {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open beige {self.drawer_id} drawer on the left"
    #     if self.station_name == "color_1":
    #         if self.drawer_pos[1] == 0:
    #             return f"close brown {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close brown {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close brown {self.drawer_id} drawer on the left"
    #     elif self.station_name == "color_2":
    #         if self.drawer_pos[1] == 0:
    #             return f"close pink {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close pink {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close pink {self.drawer_id} drawer on the left"
    #     elif self.station_name == "color_3":
    #         if self.drawer_pos[1] == 0:
    #             return f"close beige {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close beige {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close beige {self.drawer_id} drawer on the left"

    def get_language_instruction(self, **kwargs):
        if self.num == None:
            if self.drawer_pos[1] == 0:
                return f"open {self.drawer_id} drawer in the center"
            elif self.drawer_pos[1] == 0.1:
                return f"open {self.drawer_id} drawer on the right"
            else:
                return f"open {self.drawer_id} drawer on the left"
        if self.drawer_pos[1] == 0:
            return f"close {self.drawer_id} drawer in the center"
        elif self.drawer_pos[1] == 0.1:
            return f"close {self.drawer_id} drawer on the right"
        else:
            return f"close {self.drawer_id} drawer on the left"
    
    def get_task_progress(self):
        if self.num == None:
            return 0.0
        elif self.num == 1:
            return 1.0
        else:
            return 0.5
    
@register_env("OpenCloseMiddleDrawerWithDistractorsScene-v0", max_episode_steps=300)
class OpenCloseMiddleDrawerWithDistractorsSceneEnv(OpenMiddleDrawerWithDistractorsSceneEnv):
    drawer_ids = ["middle"]
    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        if qpos >= 0.2 and self.num == None: #qposが0.2以上かつopenのタスク中だったら
            self.num = 1 #openできたら1にする
        return dict(success = self.num == 1 and qpos <= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         # print(f"open {self.drawer_id} drawer")
    #         return f"open {self.drawer_id} drawer"
    #     # print(f"close {self.drawer_id} drawer")
    #     return f"close {self.drawer_id} drawer"

    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             return f"open brown {self.drawer_id} drawer"
    #         elif self.station_name == "color_2":
    #             return f"open pink {self.drawer_id} drawer"
    #         elif self.station_name == "color_3":
    #             return f"open beige {self.drawer_id} drawer"
    #     if self.station_name == "color_1":
    #         return f"close brown {self.drawer_id} drawer"
    #     elif self.station_name == "color_2":
    #         return f"close pink {self.drawer_id} drawer"
    #     elif self.station_name == "color_3":
    #         return f"close beige {self.drawer_id} drawer"
    
    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             return f"open brown {self.drawer_id} drawer with a small handle"
    #         elif self.station_name == "color_2":
    #             return f"open pink {self.drawer_id} drawer with a small handle"
    #         elif self.station_name == "color_3":
    #             return f"open beige {self.drawer_id} drawer with a small handle"
    #     if self.station_name == "color_1":
    #         return f"close brown {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_2":
    #         return f"close pink {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_3":
    #         return f"close beige {self.drawer_id} drawer with a small handle"
    
    def get_task_progress(self):
        if self.num == None:
            return 0.0
        elif self.num == 1:
            return 1.0
        else:
            return 0.5
        
    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open brown {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open brown {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open brown {self.drawer_id} drawer on the left"
    #         elif self.station_name == "color_2":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open pink {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open pink {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open pink {self.drawer_id} drawer on the left"
    #         elif self.station_name == "color_3":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open beige {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open beige {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open beige {self.drawer_id} drawer on the left"
    #     if self.station_name == "color_1":
    #         if self.drawer_pos[1] == 0:
    #             return f"close brown {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close brown {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close brown {self.drawer_id} drawer on the left"
    #     elif self.station_name == "color_2":
    #         if self.drawer_pos[1] == 0:
    #             return f"close pink {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close pink {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close pink {self.drawer_id} drawer on the left"
    #     elif self.station_name == "color_3":
    #         if self.drawer_pos[1] == 0:
    #             return f"close beige {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close beige {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close beige {self.drawer_id} drawer on the left"
            
    def get_language_instruction(self, **kwargs):
        if self.num == None:
            if self.drawer_pos[1] == 0:
                return f"open {self.drawer_id} drawer in the center"
            elif self.drawer_pos[1] == 0.1:
                return f"open {self.drawer_id} drawer on the right"
            else:
                return f"open {self.drawer_id} drawer on the left"
        if self.drawer_pos[1] == 0:
            return f"close {self.drawer_id} drawer in the center"
        elif self.drawer_pos[1] == 0.1:
            return f"close {self.drawer_id} drawer on the right"
        else:
            return f"close {self.drawer_id} drawer on the left"
    
@register_env("OpenCloseBottomDrawerWithDistractorsScene-v0", max_episode_steps=300)
class OpenCloseBottomDrawerWithDistractorsSceneEnv(OpenBottomDrawerWithDistractorsSceneEnv):
    drawer_ids = ["bottom"]
    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        self.episode_stats["qpos"] = "{:.3f}".format(qpos)
        if qpos >= 0.2 and self.num == None: #qposが0.2以上かつopenのタスク中だったら
            self.num = 1 #openできたら1にする
        return dict(success = self.num == 1 and qpos <= 0.2, qpos=qpos, episode_stats=self.episode_stats)

    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         # print(f"open {self.drawer_id} drawer")
    #         return f"open {self.drawer_id} drawer"
    #     # print(f"close {self.drawer_id} drawer")
    #     return f"close {self.drawer_id} drawer"

    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             return f"open brown {self.drawer_id} drawer"
    #         elif self.station_name == "color_2":
    #             return f"open pink {self.drawer_id} drawer"
    #         elif self.station_name == "color_3":
    #             return f"open beige {self.drawer_id} drawer"
    #     if self.station_name == "color_1":
    #         return f"close brown {self.drawer_id} drawer"
    #     elif self.station_name == "color_2":
    #         return f"close pink {self.drawer_id} drawer"
    #     elif self.station_name == "color_3":
    #         return f"close beige {self.drawer_id} drawer"
        
    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             return f"open brown {self.drawer_id} drawer with a small handle"
    #         elif self.station_name == "color_2":
    #             return f"open pink {self.drawer_id} drawer with a small handle"
    #         elif self.station_name == "color_3":
    #             return f"open beige {self.drawer_id} drawer with a small handle"
    #     if self.station_name == "color_1":
    #         return f"close brown {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_2":
    #         return f"close pink {self.drawer_id} drawer with a small handle"
    #     elif self.station_name == "color_3":
    #         return f"close beige {self.drawer_id} drawer with a small handle"

    def get_task_progress(self):
        if self.num == None:
            return 0.0
        elif self.num == 1:
            return 1.0
        else:
            return 0.5
        
    # def get_language_instruction(self, **kwargs):
    #     if self.num == None:
    #         # print(f"{self.art_obj.get_qpos()[self.joint_idx]}")#qposを出力するように修正
    #         if self.station_name == "color_1":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open brown {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open brown {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open brown {self.drawer_id} drawer on the left"
    #         elif self.station_name == "color_2":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open pink {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open pink {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open pink {self.drawer_id} drawer on the left"
    #         elif self.station_name == "color_3":
    #             if self.drawer_pos[1] == 0:
    #                 return f"open beige {self.drawer_id} drawer in the center"
    #             elif self.drawer_pos[1] == 0.1:
    #                 return f"open beige {self.drawer_id} drawer on the right"
    #             else:
    #                 return f"open beige {self.drawer_id} drawer on the left"
    #     if self.station_name == "color_1":
    #         if self.drawer_pos[1] == 0:
    #             return f"close brown {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close brown {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close brown {self.drawer_id} drawer on the left"
    #     elif self.station_name == "color_2":
    #         if self.drawer_pos[1] == 0:
    #             return f"close pink {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close pink {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close pink {self.drawer_id} drawer on the left"
    #     elif self.station_name == "color_3":
    #         if self.drawer_pos[1] == 0:
    #             return f"close beige {self.drawer_id} drawer in the center"
    #         elif self.drawer_pos[1] == 0.1:
    #             return f"close beige {self.drawer_id} drawer on the right"
    #         else:
    #             return f"close beige {self.drawer_id} drawer on the left"

    def get_language_instruction(self, **kwargs):
        if self.num == None:
            if self.drawer_pos[1] == 0:
                return f"open {self.drawer_id} drawer in the center"
            elif self.drawer_pos[1] == 0.1:
                return f"open {self.drawer_id} drawer on the right"
            else:
                return f"open {self.drawer_id} drawer on the left"
        if self.drawer_pos[1] == 0:
            return f"close {self.drawer_id} drawer in the center"
        elif self.drawer_pos[1] == 0.1:
            return f"close {self.drawer_id} drawer on the right"
        else:
            return f"close {self.drawer_id} drawer on the left"