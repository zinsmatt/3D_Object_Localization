#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from Pose_Manipulation import pose_interpolation


def bbox_to_ellipse(obb):
    """
        Return an ellipse from an oriented bbox/
        obb: [x, y, w, h, angle]  (angle in degrees)
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
    C /= C[2, 2]
    return C


def test_bbox_to_ellipse():
    bbox = [5.0, 1.0, 8.0, 4.0, 10.0]
    ellipse = bbox_to_ellipse(bbox)

    xmin, xmax = -100, 100
    ymin, ymax = -100, 100
    scale = 10

    y, x = np.mgrid[ymin:ymax+1, xmin:ymax+1]
    pts = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten()))).T
    pts = pts.astype(np.float)
    pts[:, :2] /= scale

    d = [p.dot(ellipse).dot(p) for p in pts]
    indices = np.where(np.abs(d) < 0.01)[0]
    good_points = pts[indices, :]

    axes = plt.gca()
    axes.set_xlim([xmin/scale,xmax/scale])
    axes.set_ylim([ymin/scale,ymax/scale])
    plt.scatter(good_points[:, 0], good_points[:, 1])
    plt.axes().set_aspect('equal', 'datalim')



def compute_B(P):
    p11, p12, p13, p14 = P[0, :]
    p21, p22, p23, p24 = P[1, :]
    p31, p32, p33, p34 = P[2, :]
    B = np.array([[p11**2, 2*p12*p11, 2*p13*p11, 2*p14*p11, p12**2, 2*p13*p12, 2*p14*12, p13**2, 2*p13*p14, p14**2],
                  [p21*p11, p21*p12+p22*p11, p23*p11+p21*p13, p24*p11+p21*p14, p22*p12, p22*p13+p23*p12, p22*p14+p24*p12, p23*p13, p23*p14+p24*p13, p24*p14],
                  [p31*p11, p31*p12+p32*p11, p33*p11+p31*p13, p34*p11+p31*p14, p32*p12, p32*p13+p33*p12, p32*p14+p34*p12, p33*p13, p33*p14+p34*p13, p34*p14],
                  [p21**2, 2*p22*p21, 2*p23*p21, 2*p24*p21, p22**2, 2*p23*p22, 2*p24*p22, p23**2, 2*p23*p24, p24**2],
                  [p31*p21, p31*p22+p32*p21, p33*p21+p31*p23, p34*p21+p31*p24, p32*p22, p32*p23+p33*p22, p32*p24+p34*p22, p33*p23, p33*p24+p34*p23, p34*p24],
                  [p31**2, 2*p32*p31, 2*p33*p31, 2*p34*p31, p32**2, 2*p33*p32, 2*p34*p32, p33**2, 2*p33*p34, p34**2]])
    return B

def compute_ellipsoids_overlap(Q1, Q2, samples=100):
    """
        Comute the overlapping between two ellipsoids.
        Q1 and Q2 are the matrix associated to the dual quadrics.
    """
    Q1 = -Q1 / Q1[3, 3]
#    Q1 = -Q1 / np.sign(Q1[0, 0])
    Q2 = -Q2 / Q2[3, 3]
#    Q2 = -Q2 / np.sign(Q2[0, 0])

    bbx1 = ellipsoid_to_bbox(Q1)
    bbx2 = ellipsoid_to_bbox(Q2)

    bbx_envelope = bbx1
    bbx_envelope[:, 0] = np.minimum(bbx1[:, 0], bbx2[:, 0])
    bbx_envelope[:, 1] = np.minimum(bbx1[:, 1], bbx2[:, 1])

    x, y, z = np.mgrid[:samples+1, :samples+1, :samples+1]
    pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).T.astype(np.float)
    pts /= samples
    pts[:, 0] *= (bbx_envelope[0, 1] - bbx_envelope[0, 0]) + bbx_envelope[0, 0]
    pts[:, 1] *= (bbx_envelope[1, 1] - bbx_envelope[1, 0]) + bbx_envelope[1, 0]
    pts[:, 2] *= (bbx_envelope[2, 1] - bbx_envelope[2, 0]) + bbx_envelope[2, 0]
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))

    Q1_primal = np.linalg.inv(Q1)      # primal quadrics
    Q2_primal = np.linalg.inv(Q2)

    d1 = np.asarray([p @ Q1_primal @ p.T for p in pts])
    d2 = np.asarray([p @ Q2_primal @ p.T for p in pts])

    in_Q1 = d1 <= 0
    in_Q2 = d2 <= 0
    in_Q1_and_Q2 = np.logical_and(in_Q1, in_Q2)

    intersection = float(np.sum(in_Q1_and_Q2))
    union = float(np.sum(in_Q1) + np.sum(in_Q2)- intersection)
    return intersection / union



def ellipsoid_to_bbox(Q):
    """
        Return the axis aligned bbox of an ellipsoid
        [xmin, xmax]
        [ymin, ymax]
        [zmin, zmax]
    """
    x0 = (Q[3, 0] + np.sqrt(Q[3, 0]**2 - Q[3,3] * Q[0, 0])) / Q[3, 3]
    x1 = -(-Q[3, 0] + np.sqrt(Q[3, 0]**2 - Q[3,3] * Q[0, 0])) / Q[3, 3]

    y0 = (Q[3, 1] + np.sqrt(Q[3, 1]**2 - Q[3,3] * Q[1, 1])) / Q[3, 3]
    y1 = -(-Q[3, 1] + np.sqrt(Q[3, 1]**2 - Q[3,3] * Q[1, 1])) / Q[3, 3]

    z0 = (Q[3, 2] + np.sqrt(Q[3, 2]**2 - Q[3,3] * Q[2, 2])) / Q[3, 3]
    z1 = -(-Q[3, 2] + np.sqrt(Q[3, 2]**2 - Q[3,3] * Q[2, 2])) / Q[3, 3]

    bbox = [[min(x0, x1), max(x0, x1)],
            [min(y0, y1), max(y0, y1)],
            [min(z0, z1), max(z0, z1)]]
    return np.asarray(bbox)



def test_compute_ellipsoids_overlap():
    q1 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, -1.0]])
    T = np.eye(4)
    T[0, 3] = 1.0
    Tinv = np.linalg.inv(T)

    q1 = Tinv.T @ q1 @ Tinv

    q2 = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, -1.0]])

    iou = compute_ellipsoids_overlap(np.linalg.inv(q1), np.linalg.inv(q2), 50)
    print("IoU = ", iou)


def reconstruct_objects(proj_matrices, ellipses, obj_appeareances, method, x_est):
    """ This function reconstruct the ellipsoids from ellipses observations
        in images and camera motion matrices
        x_est: estimated quadric center (this enables a better preconditioning)
              [3 x n_objects]
    """

    # Only objects visible at least 3 times can be reconstructed
    nb_views = np.sum(obj_appeareances, 0)
    reconstructible_indices = np.where(nb_views > 2)[0]

    quadrics = []
    for i in reconstructible_indices:
        visibility = obj_appeareances[:, i]
        n_views = np.sum(visibility)
        selector = np.kron(visibility, np.array([1, 1, 1]).T).astype(bool)
        print(selector)
        P = proj_matrices[selector, :]
        C = ellipses[selector, i*3:i*3+3]
        print("P shape ", P.shape)
        print("C shape ", C.shape)
        print(P)
        print(C)
    

        T = np.eye(4)
        T[:3, 3] = x_est[:, i]

        P_new = np.zeros_like(P)
        for vi in range(n_views):
            P_new[vi*3:vi*3+3, :] = (T.T @  P[vi*3:vi*3+3, :].T).T

        if method == "SVD":
            Qadj = reconstruct_ellipsoid(P_new, C)
        else:
            print("Unknown method")

        print("Qadj = ", Qadj)

        Qadj = T @ Qadj @ T.T
        Qadj = 0.5 * (Qadj + Qadj.T)
        print("Qadj = ", Qadj)
        Qadj /= -Qadj[3, 3]

        quadrics.append(Qadj)
    return quadrics


def decompose_ellipse(C):
    """
        Decompose an ellipse into 2 half-axis,
        an orientation and a center. (returned in this order)
    """
    print("CCC ", C)
    if C[2, 2] > 0:
        C = -C / C[2, 2]
    center = -C[:2, 2]
    T = np.array([[1.0, 0.0, -center[0]],
                  [0.0, 1.0, -center[1]],
                  [0.0, 0.0, 1.0]])
    C_center = T @ C @ T.T
    # force symetry
    C_center = 0.5 * (C_center + C_center.T)

    D, V = np.linalg.eig(C_center[:2, :2])
    ax = np.sqrt(np.abs(D))

    print("ax = ", ax)
    print("V = ", V)
    print("center = ", center)
    return ax, V, center

def sym2vec(M):
    """
        Return the lower triangular part of a symetric matrix
    """
    res = []
    N = M.shape[0]
    for i in range(N):
        res.extend(M[i:, i])
    return np.asarray(res)

def vec2sym(v):
    """
        Return a symetric matrix from a lower triangular part.
    """
    print(v)
    x = 1
    n = 1
    while n < v.size:
        x += 1
        n += x
    print(x)
    M = np.zeros((x, x), dtype=np.float)
    a = 0
    for i in range(x):
        M[i:, i] = v[a:a+(x-i)]
        a += (x-i)
    return M


def reconstruct_ellipsoid(Ps, Cs):
    """
        This functions reconstruct an ellipsoid from multiview ellipses
        and return the matrix of its dual representation
    """
    print("Cs = ", Cs)
    

    n_views = Cs.shape[0] // 3
    M = np.zeros((6 * n_views, 10 + n_views))
    for i in range(n_views):
        C = Cs[i*3:i*3+3, :]
        
        print(C)

        ax, R, center = decompose_ellipse(C)

        h = np.sqrt(np.sum(ax**2))
        H = np.array([[h, 0.0, center[0]],
                      [0.0, h, center[1]],
                      [0.0, 0.0, 1.0]])
        print("H= ", H)
        H_inv = np.linalg.inv(H)

        P = Ps[i*3:i*3+3, :]
        new_P = (P.T @ H_inv.T).T

        print("new_P ",  new_P)
        B = compute_B(new_P)
        
        print("C ", C)

        # normalize and center ellipsoid
        C_nc = H @ C @ H.T
        print("C_nc = ", C_nc)
        C_nc_vec = sym2vec(C_nc)
        C_nc_vec /= -C_nc_vec[-1]


        M[6*i:6*i+6, :10] = B
        M[6*i:6*i+6, 10+i] = -C_nc_vec

    print("Reconstruction method: SVD")
    print(M[:, -1])
    U, S, Vt = np.linalg.svd(M)
    w = Vt[-1, :]
    Qadj_vec = w[:10]
    Qadj = vec2sym(Qadj_vec)

    return Qadj


dataset_folder = "/home/matt/dev/3D_Object_Localization/Dataset/rgbd_dataset_freiburg2_desk/"
gt_poses_filename = os.path.join(dataset_folder, "groundtruth.txt")
images_folder = os.path.join(dataset_folder, "rgb")


detection_associations_file = "/home/matt/dev/3D_Object_Localization/ellipses_association/Yolo_c/assoc_yolo_c.txt"
tmp, ext = os.path.splitext(detection_associations_file)
images_to_use_file = tmp + ".used_images" + ext

# Load images to use
images_filenames = []
images_timestamps = []
with open(images_to_use_file, "r") as fin:
    lines = fin.readlines()
    images_filenames = [os.path.splitext(f)[0]+".png" for f in lines]
    images_timestamps = [float(os.path.splitext(os.path.basename(f))[0]) for f in lines]
    
# Load gt poses
poses_data = np.loadtxt(gt_poses_filename)
poses_timestamps = poses_data[:, 0]
poses = np.hstack((poses_data[:, 4:], poses_data[:, 1:4]))


K = np.array([[520.9, 0.0, 325.1],
              [0.0, 521.0, 249.7],
              [0.0, 0.0, 1.0]])

P = []
for t, img_f in zip(images_timestamps, images_filenames):
    d = np.abs(poses_timestamps - t)
    
    min_index = np.argmin(d)
    
    if d[min_index] > 0.02:
        print("Warning: unsure pose")
    
    M = pose_interpolation.pose_quat_to_matrix(poses[min_index])
    Rt = np.linalg.inv(M)
    
    Proj = K @ Rt[:3, :]
    P.append(Proj)
P = np.vstack(P)
    

# Load ellipses
C = np.loadtxt(detection_associations_file)
C = C[:, :3]

# Compute visibility
n_objects = C.shape[1] // 3
n_images = C.shape[0] // 3
visibility = np.ones((n_images, n_objects), dtype=int)
for i in range(n_images):
    for j in range(n_objects):
        if np.abs(C[3*i+2, 3*j+2]) < 1e-5:
            visibility[i, j] = 0

quad_centers = np.zeros((3, n_objects), dtype=np.float)
quadrics = reconstruct_objects(P, C, visibility, "SVD", quad_centers)


output_file = "/home/matt/dev/3D_Object_Localization/quadrics.txt"
quadrics = np.vstack(quadrics)
np.savetxt(output_file, quadrics)