'''
Description: 
Author: notplus
Date: 2022-01-07 14:19:47
LastEditors: notplus
LastEditTime: 2022-01-07 16:31:47
FilePath: /model/efficient_pose.py

Copyright (c) 2022 notplus
'''

import tensorflow as tf

from layers import create_efficient_pose_rt_lite

def create_efficient_pose(arch='EfficientPoseRTLite', input_size=224):
    # TODO: support EfficientPoseRT
    assert arch in ['EfficientPoseRTLite']
    
    if arch == 'EfficientPoseRTLite':
        return create_efficient_pose_rt_lite(input_size=input_size)