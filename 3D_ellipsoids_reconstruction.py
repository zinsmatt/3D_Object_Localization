#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2


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
    x = 1
    n = 1
    while n < v.size:
        x += 1
        n += x
    M = np.zeros((x, x), dtype=np.float)
    a = 0
    for i in range(x):
        M[i:, i] = v[a:a+(x-i)]
        M[i, i:] = v[a:a+(x-i)]
        a += (x-i)
    return M


def compute_B(P):
    p11, p12, p13, p14 = P[0, :]
    p21, p22, p23, p24 = P[1, :]
    p31, p32, p33, p34 = P[2, :]
    B = np.array([[p11**2, 2*p12*p11, 2*p13*p11, 2*p14*p11, p12**2, 2*p13*p12, 2*p14*p12, p13**2, 2*p13*p14, p14**2],
                  [p21*p11, p21*p12+p22*p11, p23*p11+p21*p13, p24*p11+p21*p14, p22*p12, p22*p13+p23*p12, p22*p14+p24*p12, p23*p13, p23*p14+p24*p13, p24*p14],
                  [p31*p11, p31*p12+p32*p11, p33*p11+p31*p13, p34*p11+p31*p14, p32*p12, p32*p13+p33*p12, p32*p14+p34*p12, p33*p13, p33*p14+p34*p13, p34*p14],
                  [p21**2, 2*p22*p21, 2*p23*p21, 2*p24*p21, p22**2, 2*p23*p22, 2*p24*p22, p23**2, 2*p23*p24, p24**2],
                  [p31*p21, p31*p22+p32*p21, p33*p21+p31*p23, p34*p21+p31*p24, p32*p22, p32*p23+p33*p22, p32*p24+p34*p22, p33*p23, p33*p24+p34*p23, p34*p24],
                  [p31**2, 2*p32*p31, 2*p33*p31, 2*p34*p31, p32**2, 2*p33*p32, 2*p34*p32, p33**2, 2*p33*p34, p34**2]])
    return B


def reconstruct_objects(proj_matrices, ellipses, obj_appeareances, method, centers_est):
    """ 
        This function reconstructs the ellipsoids from ellipses observations
        in images and camera motion matrices
        
        Parameters:
         - proj_matrices: [N_views*3 x 4] stacked projection matrices of the cameras
         - ellipses: [N_views*3 x N_objects*3] matrix where each block contains
                     the an ellipse matrix (dual form) of only zeros if the 
                     object is not observed in this view
         - obj_appearances: [N_views x N_objects] matrix indicating if the 
                            jth object is visible in the jth view 
         - method: the name of the method (use "SVD")
         - centers_est: [3 x N_objects] estimated ellipsoids centers (this
                        enables a better preconditioning)
        Returns:
            a list of 4x4 matrices representing the ellipsoids
    """
    # Only objects visible at least 3 times can be reconstructed
    nb_views = np.sum(obj_appeareances, 0)
    reconstructible_indices = np.where(nb_views > 2)[0]

    quadrics = [np.eye(4, dtype=float) for i in range(len(nb_views))]
    for i in reconstructible_indices:
        visibility = obj_appeareances[:, i]
        n_views = np.sum(visibility)

        selector = np.kron(visibility, [1, 1, 1]).T.astype(bool)
        P = proj_matrices[selector, :]
        C = ellipses[selector, i*3:i*3+3]

        T = np.eye(4)
        T[:3, 3] = -centers_est[:, i]

        P_new = np.zeros_like(P)
        for vi in range(n_views):
            P_new[vi*3:vi*3+3, :] = (T.T @  P[vi*3:vi*3+3, :].T).T

        if method == "SVD":
            Qadj = reconstruct_ellipsoid(P_new, C)
        else:
            print("Unknown method")

        Qadj = T @ Qadj @ T.T
        Qadj = 0.5 * (Qadj + Qadj.T)
        Qadj /= -Qadj[3, 3]
        quadrics[i] = Qadj

    return quadrics


def decompose_ellipse(C):
    """
        Decompose an ellipse into: half-axes, rotation and center
        Parameters:
            - C: adjointe matrix of the ellipse (dual form)
        Returns:
            half-axes, rotation matrix, center        
    """
    C = -C / C[2, 2]
    center = -C[:2, 2]
    T = np.array([[1.0, 0.0, -center[0]],
                  [0.0, 1.0, -center[1]],
                  [0.0, 0.0, 1.0]])
    C_center = T @ C @ T.T
    C_center = 0.5 * (C_center + C_center.T)
    D, R = np.linalg.eig(C_center[:2, :2])
    ax = np.sqrt(np.abs(D))
    return ax, R, center




def reconstruct_ellipsoid(Ps, Cs):
    """
        This functions reconstruct an ellipsoid from multiview ellipses
        and return the matrix of its dual representation
        Parameters:
            - Ps: [N_views*3 x 4] stacked projection matrices of the views
            - Cs: [N_views*3 x 3] stacked ellipses dual matrices
    """
    n_views = Cs.shape[0] // 3
    M = np.zeros((6 * n_views, 10 + n_views))
    for i in range(n_views):
        C = Cs[i*3:i*3+3, :]
        ax, R, center = decompose_ellipse(C)

        h = np.sqrt(np.sum(ax**2))
        H = np.array([[h, 0.0, center[0]],
                      [0.0, h, center[1]],
                      [0.0, 0.0, 1.0]])
        H_inv = np.linalg.inv(H)

        P = Ps[i*3:i*3+3, :]
        new_P = P.T @ H_inv.T

        B = compute_B(new_P.T)

        # normalize and center ellipsoid
        C_nc = H_inv @ C @ H_inv.T
        C_nc_vec = sym2vec(C_nc)
        C_nc_vec /= -C_nc_vec[-1]

        M[6*i:6*i+6, :10] = B
        M[6*i:6*i+6, 10+i] = -C_nc_vec


    print("Reconstruction method: SVD")
    U, S, Vt = np.linalg.svd(M)
    w = Vt[-1, :]
    Qadj_vec = w[:10]
    Qadj = vec2sym(Qadj_vec)

    return Qadj

def draw_ellipse(image, ellipse, color=(0, 255, 0)):
    """
        Draw the ellipses on image
        Parameters:
            - image: the image on which we draw
            - ellipses: [3 x 3] dual matrix representing the ellipse
    """
    if np.abs(ellipse[2, 2]) > 1e-3:
        axes, rot, center = decompose_ellipse(ellipse)
        cv2.ellipse(image, (int(round(center[0])), int(round(center[1]))),
                    (int(round(axes[0])), int(round(axes[1]))),
                    int(np.arctan2(rot[1, 0], rot[0, 0])),
                    0, 360, color, 2)
    return image


def draw_ellipses(image, ellipses, color=(0, 255, 0)):
    """
        Tool function to draw several ellipses. It just calls draw_ellipse
        several time
    """
    for ell in ellipses:
        image = draw_ellipse(image, ell, color)
    return image


def project_ellipsoids(quadrics, P):
    """
        This function projects the ellipsoids using the projection matrix P
        Parameters:
            - quadrics: list of 4x4 matrices representing the ellipsoids in dual form
            - P: [3x4] projection matrix
    """
    res = []
    for q in quadrics:
        ell = P @ q @ P.T
        ell /= -ell[2, 2]
        res.append(ell)
    return res        
    


# ----------------------------------------------------------------------------
# inputs:
#   - camera poses: a text file where each line contains: timestamp, pos, quat
#   - images RGB
#   - detection_association: a n_views*3 x n_objects*3 matrix where each block
#     is the matrix representing the ellipse of the detection of object j 
#     in image i
#   - images_to_use: a matrix n_views x n_objects indicating the visibility of 
#      each object in each view

dataset_folder = "/home/matt/dev/3D_Object_Localization/Dataset/rgbd_dataset_freiburg2_desk/"
gt_poses_file = os.path.join(dataset_folder, "groundtruth.txt")
rgb_images_folder = os.path.join(dataset_folder, "rgb")

# these two files are generated with the Detections Association Tool
detection_associations = "/home/matt/dev/3D_Object_Localization/ellipses_association/more_objects/maskrcnn.txt"
images_to_use_file = "/home/matt/dev/3D_Object_Localization/ellipses_association/more_objects/maskrcnn.used_images.txt"

C = np.loadtxt(detection_associations)
K = np.array([[520.9, 0.0, 325.1],
              [0.0, 521.0, 249.7],
              [0.0, 0.0, 1.0]])
poses = np.loadtxt(gt_poses_file)
timestamps = poses[:, 0]
positions = poses[:, 1:4]
orientations = poses[:, 4:]
images_to_use = np.genfromtxt(images_to_use_file, dtype=str)


images = []
projections = []
for f in images_to_use:
    images.append(cv2.imread(os.path.join(rgb_images_folder, f))[:, :, ::-1].copy())

    time = float(os.path.splitext(f)[0])

    d = np.abs(timestamps - time)
    index = np.argmin(d)

    rotation = R.from_quat(orientations[index, :])
    Rc_w = rotation.as_dcm().T
    Tc_w = -Rc_w.dot(positions[index, :])

    Rt = np.hstack((Rc_w, Tc_w.reshape((-1, 1))))
    projections.append(K @ Rt)
projections = np.vstack(projections)

n_images = C.shape[0] // 3
n_objects = C.shape[1] // 3
print(n_images, " images")
print(n_objects, " objects")

to_use = np.zeros((n_images, n_objects), dtype=int)
for i in range(n_images):
    for j in range(n_objects):
        if abs(C[i*3+2, j*3+2]) > 1e-3:
            to_use[i, j] = 1


centers = np.zeros((3, n_objects), dtype=float)

quadrics = reconstruct_objects(projections, C, to_use, "SVD", centers)

# update the estimated centers of the ellipsoids
for i in range(n_objects):
    Q = quadrics[i]
    centers[:, i] = -Q[:3, -1]

# recompute the quadrics with better approximation of the ellipsoids centers
quadrics = reconstruct_objects(projections, C, to_use, "SVD", centers)


np.savetxt("ellipsoids_python.txt", np.vstack(quadrics))


#%% VISUALIZATION

for i in range(6, n_images, 10):
    visible_quadrics = [quadrics[j] for j in range(n_objects) if to_use[i, j]]
    projected_ellipsoids = project_ellipsoids(visible_quadrics, projections[i*3:i*3+3, :])
    im = images[i]
    im = draw_ellipses(im, projected_ellipsoids, color=(0, 255, 0))
    plt.figure(i)
    im = draw_ellipses(im, np.split(C[i*3:i*3+3, :], n_objects, axis=1), color=(255, 0, 0))
    plt.imshow(im)
    plt.show()
    plt.waitforbuttonpress(1)
    plt.close("all")
