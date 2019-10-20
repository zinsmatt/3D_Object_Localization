#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import matplotlib.pyplot as plt


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
