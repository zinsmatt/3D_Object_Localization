#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:33:55 2019

@author: Matthieu Zins
"""

import os
import glob
import numpy as np
from Pose_Manipulation import pose_interpolation
import cv2

#dataset_folder = "rgbd_dataset_freiburg1_xyz"
#dataset_folder = "rgbd_dataset_freiburg1_360"
#dataset_folder = "rgbd_dataset_freiburg2_pioneer_360"
dataset_folder = "rgbd_dataset_freiburg2_desk"
#dataset_folder = "rgbd_dataset_freiburg3_long_office_household"

output_folder = "output"
gt_poses_filename = os.path.join(dataset_folder, "groundtruth.txt")
use_pose_interpolation = True


images_filenames = sorted(glob.glob(os.path.join(dataset_folder, "rgb", "*.png")))
images_names = [os.path.basename(f) for f in images_filenames]
images_timestamps = [float(os.path.splitext(f)[0]) for f in images_names]

poses_data = np.loadtxt(gt_poses_filename)
poses_timestamps = poses_data[:, 0]
poses = np.hstack((poses_data[:, 4:], poses_data[:, 1:4]))


#%% Create the object (cube)

pose0 = pose_interpolation.pose_quat_to_matrix(poses[0, :])
pos = pose0[:3, 3]
v = pose0[:3, 2]
pos = pos + 3 * v
pos[2] -= 1
pos = np.array([0.0, 0.0, 0.0], dtype=np.float)


# place in Freiburg 1 360
#pos = np.array([2.07383, -0.731926, 0.479844], dtype=np.float)
#cube_size = 0.1


# place in Freiburg 2 desk (poses are quite bad on this dataset)
#pos = np.array([1.27407, -1.11093, 0.989057], dtype=np.float)
#pos = np.array([(0.9798+0.865237)/2, (-1.81454-1.81922)/2, (0.761877+0.706918)/2], dtype=np.float)
#cube_size = 0.05


# place in Freiburg 3
#pos = np.array([-1.65707, -1.15254, 0.88349], dtype=np.float)
#pos = np.array([-1.85281, -0.645324, 0.718654], dtype=np.float)
#cube_size = 0.03


cube = []
for dx in [-cube_size, cube_size]:
    for dy in [-cube_size, cube_size]:
        for dz in [-cube_size, cube_size]:
            cube.append([pos[0] + dx, pos[1] + dy, pos[2] + dz])
cube = np.vstack(cube)
edges = [(0, 1), (0, 2), (0, 4),
         (1, 3), (1, 5), (3, 2), (3, 7),
         (2, 6), (4, 5), (4, 6), (5, 7), (6, 7)]
edges = np.vstack(edges)

cubes_set = [cube]
# create sevral cubes
#radius = 4.0
#for i, angle in enumerate([0]):#np.linspace(0, 2*3.14156, 10)):
#    x = radius * np.cos(angle)
#    y = radius * np.sin(angle)
#    for z in [0]:#-1.0, 0.0, 1.0, 2.0]:
#        cubes_set.append(cube + np.array([x, y, z]))
#
#    


#%% Reproject object in image
N = 800
START = 0

# dataset freiburg 1
#K = np.array([[517.3, 0.0, 318.6],
#              [0.0, 516.5, 255.3],
#              [0.0, 0.0, 1.0]])

# dataset freiburg 2
K = np.array([[520.9, 0.0, 325.1],
              [0.0, 521.0, 249.7],
              [0.0, 0.0, 1.0]])

# dataset freiburg 3
#K = np.array([[535.4, 0.0, 320.1],
#              [0.0, 539.2, 247.6],
#              [0.0, 0.0, 1.0]])

time_differences = []
for i, t in enumerate(images_timestamps[:START+N]):
    if i < START:
        continue
    img = cv2.imread(images_filenames[i])
    
    d = np.abs(poses_timestamps - t)
    min_index = np.argmin(d)
    time_differences.append(abs(d[min_index]))

    if d[min_index] > 0.02:
        print("Warning pose is very unsure")

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
            interp_pose = pose1
        else:
            interp_pose = pose_interpolation.interpolate_pose(t, t1 ,pose1, t2, pose2)
        pose = pose_interpolation.pose_quat_to_matrix(interp_pose)
    else:
        pose = pose_interpolation.pose_quat_to_matrix(poses[min_index, :])
        
    Rt = np.linalg.inv(pose)
    
    for cube in cubes_set:
        cube_cam = pose_interpolation.apply_pose(Rt, cube)
        if np.all(cube_cam[:, 2] > 0):
            projection = K @ cube_cam.T
            projection /= projection[2, :]
            projection = np.round(projection).astype(int).T
            
            for v1, v2 in edges:
                cv2.line(img, (projection[v1, 0], projection[v1, 1]),
                         (projection[v2, 0], projection[v2, 1]), 
                         (0, 255, 0), 3)

        
    cv2.imwrite(os.path.join(output_folder, images_names[i]), img)