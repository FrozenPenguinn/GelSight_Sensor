from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# import training/calibration image
img = cv2.imread("./img/1.jpg")
height, width = img.shape[0:2]
center_h = height / 2
center_w = height / 2
radius = width / 2
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

'''
# initialize reflectance function map in RGB seperately (inspired by MIT's paper) # only for universality, not accuracy
rmap = np.zeros([404,404], dtype = int)
gmap = np.zeros([404,404], dtype = int)
bmap = np.zeros([404,404], dtype = int)
for x in range(0, width):
    for y in range(0, height):
        rmap[x][y] = img[y][x][2]
        gmap[x][y] = img[y][x][1]
        bmap[x][y] = img[y][x][0]
'''

# initialize RGB map and polulate
countmap = np.zeros([255,255,255], dtype = int)
rgbmap = np.zeros([255, 255, 255, 2], dtype = float)
count = 0
for x in range(0, width):
    for y in range(0, height):
        R = img[y][x][2]
        G = img[y][x][1]
        B = img[y][x][0]
        # dz/dx
        rgbmap[R][G][B][0] = (rgbmap[R][G][B][0] * countmap[R][G][B] + gradmap[x][y][0]) / (countmap[R][G][B] + 1)
        # dz/dy
        rgbmap[R][G][B][1] = (rgbmap[R][G][B][1] * countmap[R][G][B] + gradmap[x][y][1]) / (countmap[R][G][B] + 1)
        # countmap
        countmap[R][G][B] += 1
R = img[0][0][2]
G = img[0][0][1]
B = img[0][0][0]
#print(rgbmap[R][G][B][0])
#print(countmap[R][G][B])

# check error of gradient
error_sum = 0
error_map = np.zeros([width, height], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        R = img[y][x][2]
        G = img[y][x][1]
        B = img[y][x][0]
        error_map[x][y] = pow(rgbmap[R][G][B][0] - gradmap[x][y][0], 2) + pow(rgbmap[R][G][B][1] - gradmap[x][y][1], 2)
        if (error_map[x][y] > 1):
            if (count == 2000):
                print(x, y)
                print("test dx: " + str(rgbmap[R][G][B][0]))
                print("theory dx: " + str(gradmap[x][y][0]))
                print("test dy: " + str(rgbmap[R][G][B][1]))
                print("theory dy: " + str(gradmap[x][y][1]))
            count += 1
        error_sum += error_map[x][y]
print("error count: " + str(count))
print("error sum: " + str(round(error_sum, 2)))
print(3)

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

# visualize heightmap 3D
'''
red = np.zeros([405,405], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        red[x][y] = img[y][x][0]
'''

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
print(6)
plt.show()

'''
# visualize heightmap 2D
x = np.arange(1,width + 1)
y = zmap[202]
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show()
'''
