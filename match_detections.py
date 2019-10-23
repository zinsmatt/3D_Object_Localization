#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:29:39 2019

@author: mzins
"""
#%%

import numpy as np
import glob
import os
import cv2

detections_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/MaskRCNN_detections/"
N = 1000

detections_files = sorted(glob.glob(os.path.join(detections_folder, "*.txt")))[:N]

id_counter = 0

class Tracklet:
    def __init__(self, classe):
        self.classe = classe
        global id_counter
        self.id = id_counter
        id_counter += 1
        self.detections = None
    
    def addObservation(self, time, ellipse):
        nouv = np.hstack([time, ellipse]).reshape((1, -1))
        if self.detections is not None:
            self.detections = np.vstack((self.detections, nouv))
        else:
            self.detections = nouv.reshape((1, -1))
        
    def computeSimilarity(self, time, ellipse, classe):
        if classe != self.classe:
            return 10000000
        dist_to_t = np.abs(self.detections[:, 0] - time)
        min_index = np.argmin(dist_to_t)
        val = dist_to_t[min_index]
        if val > 1:
            return 10000000            
        x, y, w, h, a = self.detections[min_index, 1:]
        a = np.deg2rad(a)
#        r = np.array([[np.cos(a), -np.sin(a)],
#                      [np.sin(a), np.cos(a)]])
#        p1 = r @ np.array([-w/2, -h/2]).reshape((-1, 1)) + np.array([x, y]).reshape((-1, 1))
#        p2 = r @ np.array([w/2, h/2]).reshape((-1, 1)) + np.array([x, y]).reshape((-1, 1))
        dist = np.sqrt((x - ellipse[0])**2 + (y - ellipse[1])**2)
        return dist
    
    def getNbObservations(self):
        if self.detections is not None:
            return self.detections.shape[0]
        else:
            return 0
        
    def getObservation(self, time):
        if self.detections is not None:
            dist = np.abs(self.detections[:, 0] - time)
            min_index = np.argmin(dist)
            if dist[min_index] < 0.001:
                return True, self.detections[min_index, 1:]
        return False, []
    
    def getFirstObservationTimestamp(self):
        if self.detections is not None:
            return self.detections[0, 0]
        else:
            return -1
    
tracklets = []


map_time_to_index = {}
images_timestamp = []

for k, f in enumerate(detections_files):
    name = os.path.splitext(os.path.basename(f))[0]
    ts, tms, ss = name.split('.')
    time_str = "%s.%s" % (ts, tms)
    timestamp = float(time_str)

    images_timestamp.append(timestamp)
    map_time_to_index[timestamp] = k
        
    detections = np.loadtxt(f).reshape((-1, 6))
    classes = detections[:, -1]
    ellipses = detections[:, :-1]
    
    
    
    for i, ell in enumerate(ellipses):
        similarities = [t.computeSimilarity(timestamp,  ell, classes[i]) for t in tracklets]
        if len(similarities) > 0:
            min_index = np.argmin(similarities)
            min_sim = similarities[min_index]
            if min_sim < 40:
                tracklets[min_index].addObservation(timestamp, ell)
                continue
        nouv_tracklet = Tracklet(classes[i])
        nouv_tracklet.addObservation(timestamp, ell)
        tracklets.append(nouv_tracklet)
    
        
#%%
        
# filter Tracks
filtered_tracks = [t for t in tracklets if t.getNbObservations() > 40]

images_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/rgb"
output_folder = "/home/mzins/dev/3D-Object-Localisation/tracks"

images_filenames = sorted(glob.glob(os.path.join(images_folder, "*.png")))[:N]

for f in images_filenames:
    image = cv2.imread(f)
    name = os.path.splitext(os.path.basename(f))[0]
    timestamp = float(name)
    for t in filtered_tracks:
        ret, ell = t.getObservation(timestamp)
        if ret:
            cv2.ellipse(image, (int(ell[0]), int(ell[1])), (int(ell[2]/2), int(ell[3]/2)), ell[4], 0, 360, (0, 255, 0), 4)
            cv2.putText(image, str(t.id), (int(ell[0]), int(ell[1])), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 245, 0))
    cv2.imwrite(os.path.join(output_folder, name+".png"), image)



#%% format detections in  a matrix
#   of size NB_images * 3 x NB_objects * 3
    
def createDualEllipseMatrixFromOBB(ellipse):
    x, y, w, h, a = ellipse
    C = np.diag((1.0/(w/2)**2, 1.0/(h/2)**2, -1))
    a = np.deg2rad(a)
    R = np.array([[np.cos(a), -np.sin(a), 0.0],
                  [np.sin(a),  np.cos(a), 0.0],
                  [0.0, 0.0, 1.0]])
    T = np.array([[1.0, 0.0, x],
                  [0.0, 1.0, y],
                  [0.0, 0.0, 1.0]])
    inv_T = np.linalg.inv(T)
    mat = inv_T.T @ R @ C @ R.T @ inv_T
    mat = np.linalg.inv(mat)    # we want the dual
    mat /= mat[2, 2]
    return mat
    
C = np.zeros((3*N, 3*len(filtered_tracks)), dtype=np.float)
to_use = np.zeros((N, len(filtered_tracks)))
for i, t in enumerate(filtered_tracks):
    if t.getFirstObservationTimestamp() > 0 and t.getFirstObservationTimestamp() < images_timestamp[500]:
        times = t.detections[:, 0]
        ellipses = t.detections[:, 1:]
        for time, ell in zip(times, ellipses):
            index = map_time_to_index[time]
            mat = createDualEllipseMatrixFromOBB(ell)
            C[index*3:index*3+3, i*3:i*3+3] = mat
            to_use[index, i] = 1

np.savetxt("C.txt", C)
np.savetxt("to_use.txt", to_use)
with open("images_to_use.txt", "w") as fout:
    for f in images_filenames:
        fout.write(f + "\n")
    
