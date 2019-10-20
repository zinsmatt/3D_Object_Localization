#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:09:12 2019

@author: Matthieu Zins
"""

import os
import glob
import numpy as np
from Pose_Manipulation import pose_interpolation, triaxe
import cv2

#dataset_folder = "rgbd_dataset_freiburg1_xyz"
#dataset_folder = "rgbd_dataset_freiburg1_360"
#dataset_folder = "rgbd_dataset_freiburg2_pioneer_360"
dataset_folder = "rgbd_dataset_freiburg3_long_office_household"

output_folder = "output"
gt_poses_filename = os.path.join(dataset_folder, "groundtruth.txt")
use_pose_interpolation = False


images_filenames = sorted(glob.glob(os.path.join(dataset_folder, "rgb", "*.png")))
images_names = [os.path.basename(f) for f in images_filenames]
images_timestamps = [float(os.path.splitext(f)[0]) for f in images_names]

poses_data = np.loadtxt(gt_poses_filename)
poses_timestamps = poses_data[:, 0]
poses = np.hstack((poses_data[:, 4:], poses_data[:, 1:4]))

triaxe_model, color = triaxe.create_triaxe(0.5, 30, True)


for i, t in enumerate(images_timestamps):
    img = cv2.imread(images_filenames[i])
    
    d = np.abs(poses_timestamps - t)
    min_index = np.argmin(d)

    if d[min_index] > 0.02:
        print("Warning pose is very unsure")
        
    pose = poses[min_index, :]
    triaxe_w = pose_interpolation.apply_pose(pose, triaxe_model)
    triaxe.write_pointcloud_PLY(os.path.join(output_folder, "pose_%d.ply" % i), triaxe_w, color)
