#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:41:26 2019

@author: mzins
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import os
import glob
import cv2

#%% Load pose file
#filename = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/groundtruth.txt"
#images_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/rgb"

filename = "/home/mzins/Dataset/rgbd_dataset_freiburg1_xyz/groundtruth.txt"
images_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg1_xyz/rgb"

#output_folder = "/home/mzins/dev/Tools/"

images_filenames = sorted(glob.glob(os.path.join(images_folder, "*.png")))[:100]
images = [cv2.imread(f) for f in images_filenames]
images_timestamps = [float(os.path.splitext(os.path.basename(t))[0]) for t in images_filenames]

# Read all poses available
all_poses = []
poses_timestamps = []
with open(filename, "r") as fin:
    lines = fin.readlines()
for l in lines:
    if len(l) > 0 and l[0] != '#':
        t, x, y, z, qx, qy, qz, qw = list(map(float, l.split()))
        all_poses.append([qx, qy, qz, qw, x, y, z])
        poses_timestamps.append(t)
all_poses = np.asarray(all_poses)
poses_timestamps = np.vstack(poses_timestamps)

# Extract only the poses corresponding to an images (or at least the closest)
corresp_timestamps = []
for t in images_timestamps:
    d = np.abs(poses_timestamps - t)
    index = np.argmin(d)
    corresp_timestamps.append(index)
poses = all_poses[corresp_timestamps, :]


# Create a box in the scene   
positions = poses[:, 4:]
v = Rot.from_quat(poses[0, :4]).as_dcm()[:, 2]
v /= np.linalg.norm(v)
center = positions[0] + v * 2
center[2] -= 0.4
center[0] += 0.3
#center = np.mean(positions, 0)
#center[2] -= 1.0

cube = []
size = 0.1
for dx in [-size, size]:
    for dy in [-size, size]:
        for dz in [-size, size]:
            cube.append([center[0]+dx, center[1]+dy, center[2]+dz])
cube = np.vstack(cube)


#np.savetxt("test_cube.txt", cube)

#%%
# Project the box in each camera
K = np.array([[520.9, 0, 325.1],
              [0, 521.0, 249.7],
              [0, 0, 1]], dtype=np.float)
    
output_folder = "projections/"

for i, p in enumerate(poses):
    orientation = Rot.from_quat(p[:4]).as_dcm()
    position = p[4:]
    R = orientation.T
    t = (-R.dot(position)).reshape((-1, 1))
    pts = K @ (R @ cube.T + t)
    pts /= pts[2, :]
    pts = np.round(pts).astype(int).T
    
    edges = [[0, 1], [0, 2], [0, 4],
             [1, 3], [1, 5],
             [2, 3], [2, 6],
             [3, 7],
             [4, 5], [4, 6],[5, 7], [6, 7]]
    for a, b in edges:
        cv2.line(images[i], (pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1]), (0, 255, 0), 3)
    
#    figure = plt.figure()
#    axes = figure.add_subplot(111)
#    axes.scatter(pts[0, :].T, pts[1, :].T)
#    axes.set_xlim(0, 800)
#    axes.set_ylim(0, 600)
#    
#    plt.savefig(os.path.join(output_folder, str(i) + ".png"))
#    plt.close("all")
    cv2.imwrite(os.path.join(output_folder, "%.6f" % images_timestamps[i] + ".png"), images[i])
    if i > 100:
        break
    