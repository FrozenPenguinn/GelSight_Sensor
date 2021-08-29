# from gradmap build rgbmap
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from math import sqrt
import pickle
from itertools import cycle

# reading saved gradmap
gradmap = np.load("gradmap.npy")

# initialize rgbmap dictionary
rgbmap_raw = {}

print("1")

# loop gradmap to map rgb to grad
for i in range(0, 401):
    for j in range(0, 401):
        R = int(gradmap[i][j][0][0])
        G = int(gradmap[i][j][1][0])
        B = int(gradmap[i][j][2][0])
        J = np.matrix([[gradmap[i,j,0,1],gradmap[i,j,0,2]],
                       [gradmap[i,j,1,1],gradmap[i,j,1,2]],
                       [gradmap[i,j,2,1],gradmap[i,j,2,2]]]) * 100
        p = (i - 200) / 100
        q = (j - 200) / 100
        if (R, G, B) in rgbmap_raw:
            rgbmap_raw[R,G,B].append((p,q,J))
        else:
            rgbmap_raw[R,G,B] = [(p,q,J)]

print("2")

# perform mean-shift clustering for every existing bin, save clusters into new dict
count = 0
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
    bandwidth = 0.3 * sqrt(np.mean(pow(p_arr,2) + pow(q_arr,2)))
    #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    if (bandwidth == 0):
        bandwidth = 0.01
    ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    if (n_clusters_ > 1): # count collisions
        count += 1
        '''
        # visualize certain bin
        plt.figure(1)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k # first determine whether is despired label
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt_name = "./clustered_plots/" + str(key) + ".png"
        plt.savefig(plt_name, bbox_inches='tight')
        #plt.show()
        '''
    for n in range(0, n_clusters_):
        if (n == 0):
            my_members = labels == n
            J_sub = J_arr[my_members]
            J = np.mean(J_sub, axis = 0)
            rgbmap_clustered[key] = [(cluster_centers[0,0], cluster_centers[0,1], J)]
        else:
            my_members = labels == n
            J_sub = J_arr[my_members]
            J = np.mean(J_sub, axis = 0)
            rgbmap_clustered[key].append((cluster_centers[n,0], cluster_centers[n,1], J))

print("collision count: " + str(count))
#print(rgbmap_clustered[(-8, 3, 0)])
# saving
file = open(r'./rgbmap_clustered.pkl', 'wb')
pickle.dump(rgbmap_clustered, file)
file.close()
print("rgbmap_clustered saved")
