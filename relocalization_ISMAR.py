#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np


# =============================================================================
# Compute camera position from one ellipse-ellipsoid correspondence 
# (assuming the orientaiton is known)
# =============================================================================


# Ellipsoid paramters
C = np.array([0.0, 0.0, 0.0])
a = 0.1
b = 0.2
c = 0.1
A = np.diag([1.0/a**2, 1.0/b**2, 1.0/c**2])


# Camera parameters
orientation = np.eye(3, dtype=float)
position = np.array([0.0, 0.0, -2.0])
pose = np.hstack((orientation, position.reshape((-1, 1))))
Rt = np.hstack((orientation.T, -orientation.T.dot(position.reshape((-1, 1)))))

K = np.array([[250.0, 0.0, 400.0],
              [0.0, 250.0, 300.0],
              [0.0, 0.0, 1.0]])

P = K @ Rt



# transform ellipsoid into homogeneous representation
Ah = np.eye(4)
Ah[:3, :3] = A
Ah[3, 3] = -1.0
t = np.eye(4, dtype=float)
t[:3, 3] = -C
Ah = t.T @ Ah @ t
Ah_dual = np.linalg.inv(Ah)


proj_dual = P @ Ah_dual @ P.T
proj_dual /= -proj_dual[2, 2]

center = -proj_dual[:2, 2]
t = np.eye(3, dtype=float)
t[:2, 2] = -center
proj_dual_center = t @ proj_dual @ t.T
proj_dual_center = (proj_dual_center + proj_dual_center.T) * 0.5
S, V = np.linalg.eig(proj_dual_center[:2, :2])
ax, ay = np.sqrt(np.abs(S))
R = V

