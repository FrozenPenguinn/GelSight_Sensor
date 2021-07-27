import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# multi-image calibration

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter, median_filter
import scipy.linalg

# import training/calibration image
img = cv2.imread("./ok/p3.jpg")
height, width = img.shape[0:2]
center_h = int(height / 2)
center_w = int(height / 2)
radius = int(width / 2)
count = 0

print(1)
# initialization gradient map and polulate
gradmap = np.zeros([width, height, 2], dtype = float) # 0:dz/dx; 1:dz/dy
hmap = np.zeros([width + 1, height + 1], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        if (pow(x - center_w, 2) + pow(y - center_h, 2) < pow(radius, 2)):
            gradmap[x][y][0] = - (x - center_w) * pow((pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2)), -0.5)
            gradmap[x][y][1] = - (y - center_h) * pow((pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2)), -0.5)
            # hmap[x][y] = sqrt(pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2))

print(2)

# size of lookup table 600*600, max is \pm arctan3
# initialize reflectance function map in RGB seperately (inspired by MIT's paper) # only for universality, not accuracy
red = np.zeros([601,601], dtype = float)
green = np.zeros([601,601], dtype = float)
blue = np.zeros([601,601], dtype = float)
countmap = np.zeros([601,601], dtype = int)
for x in range(0, width):
    for y in range(0, height):
        dx = floor(gradmap[x][y][0] * 100) + 300
        dy = floor(gradmap[x][y][1] * 100) + 300
        # limiting max and min of gradient
        if (dx > 600):
            dx = 600
        if (dx < 0):
            dx = 0
        if (dy > 600):
            dy = 600
        if (dy < 0):
            dy = 0
        red[dx][dy] = (red[dx][dy] * countmap[dx][dy] + img[y][x][2]) / (countmap[dx][dy] + 1)
        green[dx][dy] = (green[dx][dy] * countmap[dx][dy] + img[y][x][1]) / (countmap[dx][dy] + 1)
        blue[dx][dy] = (blue[dx][dy] * countmap[dx][dy] + img[y][x][0]) / (countmap[dx][dy] + 1)
        countmap[dx][dy] += 1

# interpolate lookup table above
for i in range(0,3):
    if (i == 0):
        array = red
    elif (i == 1):
        array = green
    elif (i == 2):
        array = blue
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    #mask invalid values
    array = np.ma.masked_equal(array, 0)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    # interpolate
    griddata = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='linear')
    if (i == 0):
        red_new = griddata
    elif (i == 1):
        green_new = griddata
    elif (i == 2):
        blue_new = griddata


# round and convert to int
red_new = np.round(red).astype(int)
green_new = np.round(green).astype(int)
blue_new = np.round(blue).astype(int)
'''
# some 3-dim points
mean = np.array([0.0,0.0,0.0])
cov = np.array([[1.0,-0.2,0.8], [-0.2,1.1,0.0], [0.8,0.0,1.0]])
data = np.random.multivariate_normal(mean, cov, 90)
data[:,1] = data[:,1]**2

# regular grid covering the domain of the data
r=4
X,Y = np.meshgrid(np.arange(-r, r, 0.5), np.arange(-r, r*2, 0.5))
XX = X.flatten()
YY = Y.flatten()
'''

x = np.arange(-3, 3.01, 0.01)
y = np.arange(-3, 3.01, 0.01)
X, Y = np.meshgrid(x, y)
XX = X.flatten()
YY = Y.flatten()
data[:,0] = x
data[:,1] = y
data[:,2] = red_new

order = 1   # 1: linear, 2: quadratic, 3: cubic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients

    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]

    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

elif order == 3:
    # best-fit cubic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2, data[:,:2]**3]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2, XX**3, YY**3], C).reshape(X.shape)


# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.contourf(X, Y, Z, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
#ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
#ax.axis('equal')
#ax.axis('tight')
plt.show()
