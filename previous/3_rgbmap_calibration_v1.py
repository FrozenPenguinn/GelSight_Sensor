# from gradmap build rgbmap
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from math import sqrt
import pickle

# reading saved gradmap
gradmap = np.load( "gradmap.npy" )

# initialize rgbmap dictionary
rgbmap_raw = {}

# loop gradmap to map rgb to grad
for i in range(0, 601):
    for j in range(0, 601):
        R = int(gradmap[i][j][0][0])
        G = int(gradmap[i][j][1][0])
        B = int(gradmap[i][j][2][0])
        J = np.matrix([[gradmap[i,j,0,1],gradmap[i,j,0,2]],
                       [gradmap[i,j,1,1],gradmap[i,j,1,2]],
                       [gradmap[i,j,2,1],gradmap[i,j,2,2]]])
        p = (i - 300) / 100
        q = (j - 300) / 100
        if (R, G, B) in rgbmap_raw:
            rgbmap_raw[R,G,B].append((p,q,J))
        else:
            rgbmap_raw[R,G,B] = [(p,q,J)]

# perform mean-shift clustering for every existing bin, save clusters into new dict
rgbmap_clustered = {}
for key, item in rgbmap_raw.items():
    temp = item
    p_arr = np.asarray([value[0] for value in temp])
    q_arr = np.asarray([value[1] for value in temp])
    J_arr = np.asarray([value[2] for value in temp])
    # container for MeanShift data
    X = np.zeros([p_arr.size,2], dtype = float)
    X[:,0] = p_arr
    X[:,1] = q_arr
    J = np.mean(J_arr, axis = 0)
    bandwidth = 0.05 * sqrt(np.mean(pow(p_arr,2) + pow(q_arr,2)))
    if (bandwidth == 0):
        bandwidth = 0.01
    ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    if key in rgbmap_clustered:
        for n in range(0, n_clusters_):
            rgbmap_clustered[key].append((cluster_centers[n,0], cluster_centers[n,1], J))
    else:
        rgbmap_clustered[key] = (cluster_centers[0,0], cluster_centers[0,1], J)

# saving
file = open(r'./rgbmap_clustered.pkl', 'wb')
pickle.dump(rgbmap_clustered, file)
file.close()
print("rgbmap_clustered saved")
