from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

# import training/calibration image
img = cv2.imread("./img/gauss.jpg")
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
            hmap[x][y] = sqrt(pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2))

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
    griddata = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='cubic')
    if (i == 0):
        red = griddata
    elif (i == 1):
        green = griddata
    elif (i == 2):
        blue = griddata

# round and convert to int
red = np.round(red).astype(int)
green = np.round(green).astype(int)
blue = np.round(blue).astype(int)

'''
# blur and smooth
red = gaussian_filter(red, sigma = 1)
green = gaussian_filter(green, sigma = 1)
blue = gaussian_filter(blue, sigma = 1)
'''

'''
# visualize gradient-intensity table (similar to MIT Fig.3)
fig = plt.figure(figsize = (8,8), dpi = 80)
ax = Axes3D(fig, auto_add_to_figure = False)
x = np.arange(0, 601, 1)
y = np.arange(0, 601, 1)
x, y = np.meshgrid(x, y)
ax = fig.add_subplot(1, 1, 1, projection='3d') # first plot
fig.add_axes(ax)
ax.plot_surface(x, y, red, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
# ax.contourf(x, y, red, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.set_xlim3d(0, width + 1)
#ax.set_ylim3d(0, height + 1)
#ax.set_zlim3d(0, height + 1)
ax.set_title('gradient-intensity map')
plt.show()
'''

# initialize rgbmap
countmap = np.zeros([255,255,255], dtype = int)
rgbmap = np.zeros([255, 255, 255, 2], dtype = float)
for x in range(0, 601):
    for y in range(0, 601):
        R = red[x][y]
        G = green[x][y]
        B = blue[x][y]
        # dz/dx
        rgbmap[R][G][B][0] = (rgbmap[R][G][B][0] * countmap[R][G][B] + ((x - 300) / 100)) / (countmap[R][G][B] + 1)
        # dz/dy
        rgbmap[R][G][B][1] = (rgbmap[R][G][B][1] * countmap[R][G][B] + ((y - 300) / 100)) / (countmap[R][G][B] + 1)
        # countmap
        countmap[R][G][B] += 1

'''
# check error of gradient
error_sum = 0
error_map = np.zeros([width, height], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        if (pow(x - center_w, 2) + pow(y - center_h, 2) <  pow(radius - 20, 2)):
            R = img[y][x][2]
            G = img[y][x][1]
            B = img[y][x][0]
            error_map[x][y] = pow(rgbmap[R][G][B][0] - gradmap[x][y][0], 2) + pow(rgbmap[R][G][B][1] - gradmap[x][y][1], 2)
            # error_map[x][y] = 0
            count += 1
            error_sum += error_map[x][y]
print("error count: " + str(count))
print("error sum: " + str(round(error_sum, 2)))
'''
print(3)

'''
# populate rgbmap
dx_array = rgbmap[:,:,:,0]
dy_array = rgbmap[:,:,:,1]
x = np.arange(0, 255)
y = np.arange(0, 255)
z = np.arange(0, 255)
# mask invalid values
dx_array = np.ma.masked_equal(dx_array, 0)
dy_array = np.ma.masked_equal(dy_array, 0)
xx, yy, zz = np.meshgrid(x, y, z)
# get only the valid values
px1 = xx[~dx_array.mask]
py1 = yy[~dx_array.mask]
pz1 = zz[~dx_array.mask]
dx_newarr = dx_array[~dx_array.mask]
qx1 = xx[~dy_array.mask]
qy1 = yy[~dy_array.mask]
qz1 = zz[~dy_array.mask]
dy_newarr = dy_array[~dy_array.mask]
# interpolate
print("stage 1 start")
dx_griddata = interpolate.griddata((px1, py1, pz1), dx_newarr.ravel(),(xx, yy, zz),method='nearest')
print("stage 2 start")
dy_griddata = interpolate.griddata((qx1, qy1, qz1), dy_newarr.ravel(),(xx, yy, zz),method='nearest')
rgbmap[:,:,:,0] = dx_griddata
rgbmap[:,:,:,1] = dy_griddata
'''
'''
# saving the rgbmap
np.save("rgbmap.npy", rgbmap)
print("saved!!!")
'''
# using another image for testing
img = cv2.imread("./img/gauss.jpg")
height, width = img.shape[0:2]

# reconstruct heightmap from gradmap from different directions
zmap_l = np.zeros([width + 1, height + 1], dtype = float) # left
zmap_r = np.zeros([width + 1, height + 1], dtype = float) # right
zmap_u = np.zeros([width + 1, height + 1], dtype = float) # up
zmap_d = np.zeros([width + 1, height + 1], dtype = float) # down
zmap = np.zeros([width + 1, height + 1], dtype = float) # final heightmap
for x in range(0, width):
    for y in range(0, height):
        R = img[y][x][2]
        G = img[y][x][1]
        B = img[y][x][0]
        zmap_l[x + 1][y] = zmap_l[x][y] + rgbmap[R][G][B][0]
        zmap_r[width - 1 - x][y] = zmap_r[width - x][y] + rgbmap[R][G][B][0]
        zmap_u[x][y + 1] = zmap_u[x][y] + rgbmap[R][G][B][1]
        zmap_d[x][height - 1 - y] = zmap_d[x][height - y] + rgbmap[R][G][B][1]

print(4)

# summing by weight
for y in range(0, height):
    for x in range(0, width):
        zmap[x][y] = (zmap_l[x][y] * (width - x) + zmap_r[x][y] * x + zmap_u[x][y] * (height - y) + zmap_d[x][y] * y) / (width + height) # in 4 directions
        # zmap[x][y] = (zmap_l[x][y] * (404 - x) + zmap_r[x][y] * x) / 404 # in 2 directions
print(5)


# visualize 3D height map
fig = plt.figure(figsize = (8,8), dpi = 80)
ax = Axes3D(fig, auto_add_to_figure = False)
ax = fig.add_subplot(1, 2, 1, projection='3d') # first plot
fig.add_axes(ax)
x = np.arange(0, width + 1, 1)
y = np.arange(0, height + 1, 1)
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, zmap, rstride = 5, cstride = 5, cmap = plt.get_cmap('rainbow'))
# ax.contourf(x, y, zmap, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0, width + 1)
ax.set_ylim3d(0, height + 1)
ax.set_zlim3d(0, height + 1)
ax.set_title('reconstructed surface')

ax = fig.add_subplot(1, 2, 2, projection='3d') # second plot
ax.plot_surface(x, y, hmap, rstride = 5, cstride = 5, cmap = plt.get_cmap('rainbow'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0, width + 1)
ax.set_ylim3d(0, height + 1)
ax.set_zlim3d(0, height + 1)
ax.set_title('testing surface')
plt.show()

'''
ax = fig.add_subplot(2, 2, 3, projection='3d') # third plot
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, error_map, rstride = 5, cstride = 5, cmap = plt.get_cmap('rainbow'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0, width)
ax.set_ylim3d(0, height)
ax.set_zlim3d(0, 10)
ax.set_title('error source')
print(6)
'''
