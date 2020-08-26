#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import os
from Pose_Manipulation import triaxe


quadrics_file = "/home/matt/dev/3D_Object_Localization/ellipsoids_python.txt"
output_file = "/home/matt/dev/3D_Object_Localization/ellipsoids.obj"

Qs = np.loadtxt(quadrics_file)

n_objects = Qs.shape[0] // 4

pts = []
faces = []
with open("unit_sphere.obj", "r") as fin:
    lines = fin.readlines()
    for l in lines:
        if len(l) > 0 and l[0] == 'v':
            pts.append(list(map(float, l[1:].split())))
        elif len(l) > 0 and l[0] == 'f':
            faces.append(list(map(int, l[1:].split())))
pts = np.vstack(pts)
faces = np.vstack(faces)


all_pts = []
all_faces = []
for i in range(n_objects):
    
    Q = Qs[i*4:i*4+4, :]
    Q /= -Q[3, 3]
    
    center = -Q[:3, -1]
    T = np.eye(4)
    T[:3, -1] = -center
    
    C_center = T @ Q @ T.T
    
    # force symetry
    C_center = 0.5 * (C_center + C_center.T)
    

    D, V = np.linalg.eig(C_center[:3, :3])
    ax = np.sqrt(np.abs(D))
    scaling = np.array([[ax[0], 0.0, 0.0],
                        [0.0, ax[1], 0.0],
                        [0.0, 0.0, ax[2]]])
    
    pts_w = (V @ scaling @ pts.T) + center.reshape((-1, 1))
    faces_w = faces + i * pts.shape[0]
    
    all_pts.append(pts_w.T)
    all_faces.append(faces_w)
    
all_pts = np.vstack(all_pts)
all_faces = np.vstack(all_faces)

with open(output_file, "w") as fout:
    for p in all_pts:
        fout.write("v " + " ".join(p.astype(str)) + "\n")
        
    for f in all_faces:
        fout.write("f " + " ".join(f.astype(str)) + "\n")
    
    
    
    

