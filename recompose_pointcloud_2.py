#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:51:34 2019

@author: Matthieu Zins
"""

import os
import glob
import numpy as np
from Pose_Manipulation import pose_interpolation, triaxe
import cv2


def depthmap_to_pointcloud(depth, K, color_image=None):
    y, x = np.mgrid[:depth.shape[0], :depth.shape[1]]
    d = depth.flatten()
    xyz = np.vstack((x.flatten(), y.flatten(), np.ones(depth.size).flatten()))
    if color_image is not None:
        rgb = color_image.reshape((-1, 3))
    K_inv = np.linalg.inv(K)
    pc = (K_inv @ xyz).T
    pc[:, 0] *= d
    pc[:, 1] *= d
    pc[:, 2] *= d
    
    
    good_points = np.where(np.logical_and(pc[:, 2] > 0.1, pc[:, 2] < 10))[0]
#    pc = pc[pc[:, 2] > 0.1]
#    pc = pc[pc[:, 2] < 10]
    pc = pc[good_points, :]
    if color_image is not None:
#        rgb = rgb[pc[:, 2] > 0.1, :]
#        rgb = rgb[pc[:, 2] < 10, :]
        rgb = rgb[good_points, :]
        rgb = rgb[:, ::-1]
        return pc, rgb
    else:
        return pc

    



#dataset_folder = "rgbd_dataset_freiburg1_xyz"
#dataset_folder = "rgbd_dataset_freiburg1_360"
#dataset_folder = "rgbd_dataset_freiburg2_pioneer_360"
dataset_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk"
#dataset_folder = "rgbd_dataset_freiburg3_long_office_household"

output_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/recomposed_pointcloud"
depth_folder = "depth"
gt_poses_filename = os.path.join(dataset_folder, "groundtruth.txt")
use_pose_interpolation = False


images_filenames = sorted(glob.glob(os.path.join(dataset_folder, "depth", "*.png")))
images_names = [os.path.basename(f) for f in images_filenames]
images_timestamps = [float(os.path.splitext(f)[0]) for f in images_names]

rgb_filenames = sorted(glob.glob(os.path.join(dataset_folder, "rgb", "*.png")))
rgb_names = [os.path.basename(f) for f in rgb_filenames]
rgb_timestamps = np.array([float(os.path.splitext(f)[0]) for f in rgb_names])



poses_data = np.loadtxt(gt_poses_filename)
poses_timestamps = poses_data[:, 0]
poses = np.hstack((poses_data[:, 4:], poses_data[:, 1:4]))




N = 10

K_depth = np.array([[525.0, 0.0, 319.5],
                    [0.0, 525.0, 239.5],
                    [0.0, 0.0, 1.0]])


samples = np.linspace(0, len(images_timestamps)-1, N, dtype=int)

scene = []
scene_colors = []
for i in samples:
    t = images_timestamps[i]
    
    depth = cv2.imread(images_filenames[i], cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float) / 5000
    
    
    
    d_rgb = np.abs(rgb_timestamps - t)
    min_index_rgb = np.argmin(d_rgb)
    rgb = cv2.imread(rgb_filenames[min_index_rgb], cv2.IMREAD_UNCHANGED)
    
    
    pc, colors = depthmap_to_pointcloud(depth, K_depth, rgb)
    
    
    d = np.abs(poses_timestamps - t)
    min_index = np.argmin(d)

    unsure_pose = False
    if d[min_index] > 0.02:
        print("Warning pose is very unsure")
        unsure_pose = True
        
    
    

    if use_pose_interpolation:
        if t > poses_timestamps[min_index]:
            t1 = poses_timestamps[min_index]
            t2 = poses_timestamps[min(min_index+1, len(poses_timestamps)-1)]
            pose1 = poses[min_index]
            pose2 = poses[min(min_index+1, len(poses_timestamps)-1)]
        else:
            t1 = poses_timestamps[max(0, min_index-1)]
            t2 = poses_timestamps[min_index]
            pose1 = poses[max(0, min_index-1)]
            pose2 = poses[min_index]

        if t < t1 or t > t2:        # if at the poses boundaries, we cannot interpolate
            t = t1
            inter_pose = pose1
        else:
            interp_pose = pose_interpolation.interpolate_pose(t, t1 ,pose1, t2, pose2)
        pose = pose_interpolation.pose_quat_to_matrix(interp_pose)
    else:
        pose = pose_interpolation.pose_quat_to_matrix(poses[min_index, :])
        
    pc_w = pose_interpolation.apply_pose(pose, pc)
    triaxe.write_pointcloud_PLY(os.path.join(output_folder, "pc_%d.ply" % t), pc_w, colors)
    if not unsure_pose:
        scene.append(pc_w)
        scene_colors.append(colors)
    
# save the total scene pointcloud
if len(scene) > 0:
    scene = np.vstack(scene)
    scene_colors = np.vstack(scene_colors)
    triaxe.write_pointcloud_PLY(os.path.join(output_folder, "scene.ply"), scene, scene_colors)
else:
    print("empty scene")