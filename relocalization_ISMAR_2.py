#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import os
import glob
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as Rot


K = np.array([[520.9, 0.0, 325.1],
              [0.0, 521.0, 249.7],
              [0.0, 0.0, 1.0]])
K_inv = np.linalg.inv(K)


def project_ellipsoids(quadrics, Rt):
    """
        This function projects the ellipsoids using the projection matrix P
        Parameters:
            - quadrics: list of 4x4 matrices representing the ellipsoids in dual form
            - Rt: [3x4] extrinsic matrix of the projection
    """
    global K
    P = K @ Rt
    res = []
    for q in quadrics:
        ell = P @ q @ P.T
        ell /= -ell[2, 2]
        res.append(ell)
    return res        
    

def oriented_bbox_to_ellipse(obb):
    """
        Return the dual-form matrix of an ellipse given its oriented bbox
        Parameters:
            - obb: [x, y, w, h, angle]  (angle in degrees)
        Returns:
            - ellipse in dual-form as a 3x3 matrix
    """
    x, y, w, h, angle = obb
    C = np.diag([1.0/(w/2)**2, 1.0/(h/2)**2, -1.0])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a), 0.0],
                  [np.sin(a), np.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    T = np.array([[1.0, 0.0, x],
                  [0.0, 1.0, y],
                  [0.0, 0.0, 1.0]])
    Tinv = np.linalg.inv(T)
    Rinv = R.T
    C = Tinv.T @ Rinv.T @ C @ Rinv @ Tinv
#    C /= C[2, 2]
    C_dual = np.linalg.inv(C)
    C_dual /= -C_dual[2, 2]
    return C_dual


def compute_backprojection_cone(ell):
    """
        This function compute the 3x3 matrix (Euclidena representation) of 
        the backprojection cone.
        Parameters:
            - ell: ellipse [x, y, w, h, angle, label]
        Returns:
            - cone as a 3x3 matrix
    """
    x, y, w, h, angle, label = ell
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)],
                   [np.sin(a), np.cos(a)],
                   [0.0, 0.0]])
    a = w / 2
    b = h / 2
    Ec = np.zeros(3, dtype=float)
    global K_inv
    Kc = K_inv.dot(np.array([x, y, 1.0]))
    
    Uc = R[:, 0].reshape((-1, 1))
    Vc = R[:, 1].reshape((-1, 1))
    Nc = np.array([0.0, 0.0, 1.0])
    M = (Uc @ Uc.T) / a**2 + (Vc @ Vc.T) / b**2
    W = Nc.reshape((-1, 1)) / Nc.dot(Kc - Ec)
    P = np.eye(3, dtype=float) - (Kc - Ec).reshape((-1, 1)) @ W.T
    Q = W @ W.T
    B = P.T @ M @ P - Q
    return B


def decompose_ellipsoid(Q):
    """ 
        This function decomposes a dual-form ellipsoid into center, axes and 
        rotation
        Parameters:
            - Q: 4x4 dual-form matrix
        Returns:
            [axes, rot, center]
    """
    Q /= -Q[3, 3]
    
    center = -Q[:3, -1]
    T = np.eye(4, dtype=float)
    T[:3, -1] = -center
    Q_center = T @ Q @ T.T
    Q_center = (Q_center + Q_center.T) * 0.5
    D, R = np.linalg.eig(Q_center[:3, :3])
    axes = np.sqrt(np.abs(D))
    if np.linalg.det(R) < 0:
        R *= -1
    return [axes, R, center]
    
    
    

def get_euclidean_form_ellipsoid(Q):
    """
        This function computes the Euclidean form of the ellipsoid
        Parameters:
            - Q: 4x4 dual-form matrix representing the ellipsoid
        Return:
            [Q3x3, center]: the Euclidean form of the ellipse
    """
    axes, R, center = decompose_ellipsoid(Q)
    # Euclidean-form ellipsoid (=> 3x3 matrix + center)
    Q3 = R @ np.diag([1.0/axes[0]**2, 1.0/axes[1]**2, 1.0/axes[2]**2]) @ R.T
    return Q3, center
        

def get_camera_pose(timestamp, tolerance):
    """
        This function find the closest camera pose.
        Parameters:
            - timestamp of the image
            - tolerance: (in s) if the closest time is farther than this
                         tolerance, the pose is considered unreliable
        Returns:
            - the pose [3x4] matrix [R t]
            - the reliability (True or False)
    """
    global gt_poses
    d = np.abs(timestamp - gt_poses[:, 0])
    idx = np.argmin(d)
    print("idx = ", idx)
    reliable = True
    if abs(timestamp - gt_poses[idx, 0]) > tolerance:
        reliable = False
    R = Rot.from_quat(gt_poses[idx, 4:]).as_dcm()
    t = gt_poses[idx, 1:4].reshape((-1, 1))
    return np.hstack((R, t)), reliable


def compute_position(Rw_c, ellipsoid, B):
    """
        This function compute the position of the camera from an estimate of 
        its orientation (Rw_c), and ellipsoid (Q) and a backprojection cone
        obtained from an ellipse
        Parameters:
            - Rw_c: [3x3] estimate of the camera orientation
            - ellipsoid: [Q3, center] the Euclidean form of the ellipsoid
            - B: [3x3] matrix of the backprojection cone
        Returns:
            - an estimate of the camera position
    """
    Q3, C = ellipsoid
    Ac = Rw_c.T @ Q3 @ Rw_c
    P, D = scipy.linalg.eig(Ac, B)
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
    B = s1 * B
    num_k = (B - Ac)
    denom_k = (Ac @ delta @ delta.T @ Ac - (delta.T @ Ac @ delta) * Ac)
    k = np.divide(num_k, denom_k)
    k = np.nan_to_num(k)
    
    k_vec = np.diag(k)
    k = np.sign(delta[2]) * np.sqrt(np.mean(k_vec[np.nonzero(k_vec)]))
    
    Delta = k * delta
    Ew = Rw_c @ Delta + C.reshape((-1, 1))
    return Ew


    
# camera ground-truth poses
camera_gt_poses_file = "/home/matt/dev/3D_Object_Localization/Dataset/rgbd_dataset_freiburg2_desk/groundtruth.txt"
gt_poses = np.loadtxt(camera_gt_poses_file)



    
# Load 3D ellipsoids map
ellipsoids_file = "/home/matt/dev/3D_Object_Localization/ellipsoids_python.txt"


# dual-form matrices
ellipsoids = np.loadtxt(ellipsoids_file)
n_objects = ellipsoids.shape[0] // 4
ellipsoids = np.split(ellipsoids, n_objects, axis=0)

ellipsoids_labels = [""] # tofill


# load objects detections
detections_folder = "/home/matt/dev/3D_Object_Localization/Dataset/rgbd_dataset_freiburg2_desk/MaskRCNN_Detections/"
detections_files = sorted(glob.glob(os.path.join(detections_folder, "*.txt")))

detection_file = detections_files[0]
detections = np.loadtxt(detection_file)
timestamp = float(".".join(os.path.splitext(os.path.basename(detection_file))[0].split('.')[:-1]))
pose, reliable = get_camera_pose(timestamp, 0.05)





ellipses = detections
ellipses_matrices = [oriented_bbox_to_ellipse(d) for d in detections[:, :-1]]


backproj_cones = [compute_backprojection_cone(ell) for ell in ellipses]

Rw_c = pose[:3, :3]


position = compute_position(pose[:3, :3], get_euclidean_form_ellipsoid(ellipsoids[0]), backproj_cones[0])
print("Computed position = ", position.T)
print("Ground-truth position = ", pose[:, -1].T)