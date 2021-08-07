from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import cKDTree
import pickle
from numpy.linalg import inv

# data structure is needed when performing KNN
class KDDict(dict):
    def __init__(self, ndims, regenOnAdd=False):
        super(KDDict, self).__init__()
        self.ndims = ndims
        self.regenOnAdd = regenOnAdd
        self.__keys = []
        self.__tree = None
        self.__stale = False

    # Enforce dimensionality
    def __setitem__(self, key, val):
        if not isinstance(key, tuple): key = (key,)
        if len(key) != self.ndims: raise KeyError("key must be %d dimensions" % self.ndims)
        self.__keys.append(key)
        self.__stale = True
        if self.regenOnAdd: self.regenTree()
        super(KDDict, self).__setitem__(key, val)

    def regenTree(self):
        self.__tree = cKDTree(self.__keys)
        self.__stale = False

    # Helper method and might be of use
    def nearest_key(self, key):
        if not isinstance(key, tuple): key = (key,)
        if self.__stale: self.regenTree()
        _, idx = self.__tree.query(key, 1)
        return self.__keys[idx]

    def __missing__(self, key):
        if not isinstance(key, tuple): key = (key,)
        if len(key) != self.ndims: raise KeyError("key must be %d dimensions" % self.ndims)
        return self[self.nearest_key(key)]

# load the rgbmap
file = open(r'./rgbmap_clustered.pkl', 'rb')
rgbmap_clustered = pickle.load(file)
file.close()

count = 0
# transform rgbmap from Dict to KDDict
rgbmap_kd = KDDict(3)
for key, item in rgbmap_clustered.items():
    if key in rgbmap_kd:
        for n in range(0, n_clusters_):
            rgbmap_kd[key].append(item)
    else:
        #print(key)
        #print(item)
        rgbmap_kd[key] = item
        if (len(item) > 1):
            count += 1
            #print("multi")
#print("collision count: " + str(count))

# using another image for testing
raw_img = cv2.imread("./testing/raw_img.jpg")
raw_img = cv2.GaussianBlur(raw_img,(3,3),0)
bg_img = cv2.imread("./testing/bg_img.jpg")
bg_img = cv2.GaussianBlur(bg_img,(3,3),0)
height, width = raw_img.shape[0:2]
print(height, width)

# get difference in seperate color channels
diff_img = np.zeros_like(raw_img)
diff_img = (raw_img.astype(int) - bg_img.astype(int))

'''
display = diff_img[:,:,2]
plt.imshow(display)
plt.colorbar()
plt.show()
'''

# gradient map of the testing img
grad = np.zeros([width, height, 2], dtype = float)
# change of color near target pixels
dR_img = np.gradient(diff_img[:,:,2]) #dR[0][:,:] gives p, dR[1][:,:] gives q
dG_img = np.gradient(diff_img[:,:,1])
dB_img = np.gradient(diff_img[:,:,0])
# determine gradient of img pixel wise
count_same = 0
count_multi_clu = 0
count_dif = 0
count_dif_multi = 0
for y in range(0, height):
    print(str(y)+"/"+str(height))
    for x in range(0, width):
        R = diff_img[y][x][2]
        G = diff_img[y][x][1]
        B = diff_img[y][x][0]
        if (R,G,B) in rgbmap_kd: # key exists in rgbmap
            if(len(rgbmap_kd[R,G,B]) == 1): # if there is only one cluster in the bin
                count_same += 1
                grad[x][y][0] = rgbmap_kd[R,G,B][0][0]
                grad[x][y][1] = rgbmap_kd[R,G,B][0][1]
            else:
                count_multi_clu += 1
                min = 1e+10
                dI_img = np.asarray([dR_img[0][y][x] + dR_img[1][y][x],
                                     dG_img[0][y][x] + dG_img[1][y][x],
                                     dB_img[0][y][x] + dB_img[1][y][x]])
                # choose most suitable if more than one cluster in bin
                for index in range(0, len(rgbmap_kd[R,G,B])):
                    p = rgbmap_kd[R,G,B][index][0]
                    q = rgbmap_kd[R,G,B][index][1]
                    J_pre = rgbmap_kd[R,G,B][index][2]
                    #print("this is J_pre: ")
                    #print(J_pre)
                    dI_pre = J_pre.dot(np.asarray([p,q]).reshape((2,1)))
                    #print("this is dI_img: ")
                    #print(dI_img)
                    #print("this is dI_pre: ")
                    #print(dI_pre)
                    norm = np.linalg.norm(dI_pre - dI_img)
                    if (norm < min):
                        min = norm
                        grad[x][y][0] = p
                        grad[x][y][1] = q
        else: # key doesn't exist in rgbmap
            # approximation by first order Taylor
            closest_key = rgbmap_kd.nearest_key((R,G,B))
            #print("original key: ")
            #print(R,G,B)
            #print(closest_key)
            if(len(rgbmap_kd[closest_key]) == 1): # only one cluster in closest key
                #count_dif += 1
                grad[x][y][0] = rgbmap_kd[R,G,B][0][0]
                grad[x][y][1] = rgbmap_kd[R,G,B][0][1]
                #print("diff key")
                J = rgbmap_kd[closest_key][0][2]
                #print(J)
                J_tik = inv(J.T.dot(J) + 0.05 * np.identity(2)).dot(J.T)
                #print(J_tik)
                diff = J_tik.dot((np.asmatrix([R,G,B]).T - np.asmatrix(closest_key).T))
                #print(np.asmatrix([R,G,B]).T - np.asmatrix(closest_key).T)
                #print(diff)
                grad[x][y][0] = grad[x][y][0] + diff[0]
                grad[x][y][1] = grad[x][y][1] + diff[1]
            else: # more than one cluster in closest key
                #print("This is original: " + str(R) + " " + str(R) + " " + str(R))
                #print("This is closest: " + str(closest_key))
                count_dif_multi += 1
                min = 1e+10
                dI_img = np.asarray([dR_img[0][y][x] + dR_img[1][y][x],
                                     dG_img[0][y][x] + dG_img[1][y][x],
                                     dB_img[0][y][x] + dB_img[1][y][x]])
                # choose most suitable if more than one cluster in bin
                #print("multi")
                for index in range(0, len(rgbmap_kd[R,G,B])):
                    #print("new")
                    #print(rgbmap_kd[R,G,B])
                    p = rgbmap_kd[R,G,B][index][0]
                    q = rgbmap_kd[R,G,B][index][1]
                    J_pre = rgbmap_kd[R,G,B][index][2]
                    dI_pre = J_pre.dot(np.asarray([p,q]).reshape((2,1)))
                    #print("dI_img: ")
                    #print(dI_img)
                    #print("dI_pre: ")
                    #print(dI_pre)
                    norm = np.linalg.norm(dI_pre - dI_img)
                    if (norm < min):
                        min = norm
                        grad[x][y][0] = p
                        grad[x][y][1] = q
                #print("not in bin and multi-clusters")

print("This is count same: " + str(count_same))
print("This is count multi clu: " + str(count_multi_clu))
print("This is count diff: " + str(count_dif))
print("This is count dif multi: " + str(count_dif_multi))
map1 = np.flip(np.transpose(grad[:,:,0]), 0)
map2 = np.flip(np.transpose(grad[:,:,1]), 0)
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x, y = np.meshgrid(x, y)
# graph
matplotlib.rc('figure', figsize=(10, 5))
fig, axes = plt.subplots(1, 2)
# difference img
dif = np.flip(diff_img[:,:,2],0)
im1 = axes[0].imshow(dif, origin='lower', extent=[0, width, 0, height])
# extracted gradients
im2 = axes[1].imshow(map1, origin='lower', extent=[0, width, 0, height])
# set labels
plt.setp(axes[:], xlabel='x')
plt.setp(axes[:], ylabel='y')
axes[0].set_title("Red difference")
axes[1].set_title("Gradient in X-direction")
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
print(raw_img.shape)
plt.show()

# saving the gradients
np.save("grad.npy", grad)
print("grad saved")
