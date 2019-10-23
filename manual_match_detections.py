#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""
#%%

import numpy as np
import glob
import os
import cv2

#detections_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/MaskRCNN_detections/"
detections_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/Yolo_detections/"

detections_files = sorted(glob.glob(os.path.join(detections_folder, "*.txt")))

images_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/rgb"
output_folder = "/home/mzins/dev/3D-Object-Localisation/tracks"

images_filenames = sorted(glob.glob(os.path.join(images_folder, "*.png")))
N = len(images_filenames)

tracklets = []


map_time_to_index = {}
images_timestamp = []

images_indices = [310, 391, 595, 726, 924, 2142, 2376, 2590, 2648, 2847]
images_indices = list(map(lambda x: x-1, images_indices))


id_counter = 0

tracks = []
used_images = []
for idx in images_indices:
    f = detections_files[idx]
        
    detections = np.loadtxt(f).reshape((-1, 6))
    classes = detections[:, -1]
    detections = detections[:, :-1]
    
    ell = detections[classes == 63, ]
    if ell.shape[0] >= 1:
        ell = ell[0, :]
    if len(ell) > 0:
        tracks.append(ell)
        used_images.append(idx)




for i, idx in enumerate(used_images):
    f = images_filenames[idx]
    image = cv2.imread(f)
    name = os.path.splitext(os.path.basename(f))[0]
    
    ell = tracks[i]
    cv2.ellipse(image, (int(ell[0]), int(ell[1])), (int(ell[2]/2), int(ell[3]/2)), ell[4], 0, 360, (0, 255, 0), 4)
    #cv2.putText(image, str(t.id), (int(ell[0]), int(ell[1])), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 245, 0))
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
    
C = np.zeros((3*N, 3), dtype=np.float)
to_use = np.zeros((N, 1))

for idx, ell in zip(used_images, tracks):
    mat = createDualEllipseMatrixFromOBB(ell)
    to_use[idx] = 1
    C[idx*3:idx*3+3, :] = mat

np.savetxt("C.txt", C)
np.savetxt("to_use.txt", to_use, header="need a header for MATLAB")
with open("images_to_use.txt", "w") as fout:
    for f in images_filenames:
        fout.write(f + "\n")
    
