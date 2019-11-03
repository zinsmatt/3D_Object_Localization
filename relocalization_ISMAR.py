#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import scipy.linalg

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



# copmpute backprojection cone
center = np.hstack((center, 1.0))
Kc = np.linalg.inv(K) @ center
Kc = Kc.reshape((-1, 1))
U = np.hstack((R[:, 0], 0.0)).reshape((-1, 1))
V = np.hstack((R[:, 1], 0.0)).reshape((-1, 1))
N = np.array([0.0, 0.0, 1.0]).reshape((-1, 1))
E = np.zeros(3).reshape((-1, 1))

M = U @ U.T / a**2 + V @ V.T / b**2

W = N / (N.T.dot(Kc - E))
P = np.eye(3, dtype=float) - (Kc - E) @ W.T
Q = W @ W.T
B_ = P.T @ M @ P - Q


# estimate of the rotation
Rw_c = np.eye(3, dtype=float)


# compute position from ellipsoid + backprojection cone

Ac = Rw_c.T @ A @ Rw_c
P, D = scipy.linalg.eig(Ac, B_)
Preal = np.real(P)
diffs = [abs(Preal[i]-Preal[(i+1) % 3]) for i in range(3)]
i_min = np.argmin(diffs)
if i_min == 0:
    s1 = Preal[2]
    s2 = (Preal[0]+Preal[1])/2
    delta = D[:, 2]
elif i_min == 1:
    s1 = Preal[0]
    s2 = (Preal[1] + Preal[2]) / 2
    delta = D[0]
else:
    s1 = Preal[1]
    s2 = (Preal[0] + Preal[2]) / 2
    delta = D[1]
delta = delta.reshape((-1, 1))

B = s1 * B_
num_K = (B - Ac)
denom_K = (Ac @ delta @ delta.T @ Ac - (delta.T @ Ac @ delta) * Ac)
K = np.divide(num_K, denom_K)
K = np.nan_to_num(K)

K_vec = np.diag(K)
k = np.sign(delta[2]) * np.sqrt(np.mean(K_vec))

Delta = k * delta
Ew = Rw_c @ Delta + C.reshape((-1, 1))
print("Ew = ", Ew)
