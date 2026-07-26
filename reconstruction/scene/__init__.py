#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.flexible_deform_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset

class Scene:
    """
    场景类，用于加载和管理场景数据，包括相机信息和点云数据。
    """
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None):
        """
        初始化场景。
        参数：
        - args: 模型参数
        - gaussians: 高斯模型
        - load_iteration: 加载的迭代次数
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info = sceneLoadTypeCallbacks["endonerf"](
                args.source_path, args.dataset_type
            )
            print("Found poses_bounds.npy and an EndoNeRF-compatible scene")
        elif os.path.exists(os.path.join(args.source_path, "point_cloud.obj")) or os.path.exists(os.path.join(args.source_path, "left_point_cloud.obj")):
            scene_info = sceneLoadTypeCallbacks["scared"](args.source_path, args.white_background, args.eval)
            print("Found point_cloud.obj, assuming SCARED data!")
        else:
            assert False, "Could not recognize scene type!"

        self.maxtime = scene_info.maxtime
        self.cameras_extent = (
            args.camera_extent
            if args.camera_extent is not None
            else scene_info.nerf_normalization["radius"]
        )
        print("self.cameras_extent is ", self.cameras_extent)

        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras
        print("Loading Video Cameras")
        self.video_camera =  scene_info.video_cameras

        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        # self.gaussians._deformation.deformation_net.grid.set_aabb(xyz_max,xyz_min)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, self.cameras_extent, self.maxtime
            )

    def save(self, iteration, stage):
        """
        保存当前迭代的点云数据。
        参数：
        - iteration: 当前迭代次数
        - stage: 保存阶段（粗略或精细）
        """
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # self.gaussians.save_deformation(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        """
        获取训练相机信息。
        参数：
        - scale: 缩放比例
        返回：
        - 训练相机信息列表
        """
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        """
        获取测试相机信息。
        参数：
        - scale: 缩放比例
        返回：
        - 测试相机信息列表
        """
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        """
        获取视频相机信息。
        参数：
        - scale: 缩放比例
        返回：
        - 视频相机信息列表
        """
        return self.video_camera
