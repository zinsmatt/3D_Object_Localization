#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import numpy as np
import matplotlib.pyplot as plt


def OBB_to_ellipse(obb):
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


def test_OBB_to_ellipse():
    bbox = [5.0, 1.0, 8.0, 4.0, 10.0]
    ellipse = OBB_to_ellipse(bbox)
    
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

    

    
    
    
    