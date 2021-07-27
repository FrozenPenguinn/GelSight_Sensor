
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import cv2
import numpy as np
import matplotlib.pyplot as plt

# import training/calibration image
img = cv2.imread("./img/3.jpg")
height, width = img.shape[0:2]
center_h = height / 2
center_w = height / 2
radius = width / 2
count = 0

print(1)
# initialization gradient map and polulate
gradmap = np.zeros([404, 404, 2], dtype = float) # 0:dz/dx; 1:dz/dy
for x in range(0, width):
    for y in range(0, height):
        if (pow(x - center_w, 2) + pow(y - center_h, 2) < pow(radius, 2)):
            gradmap[x][y][0] = - (x - center_w) * pow((pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2)), -0.5)
            gradmap[x][y][1] = - (y - center_h) * pow((pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2)), -0.5)

print(2)
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

print(3)
# reconstruct heightmap from gradmap from different directions
zmap_l = np.zeros([405,405], dtype = float) # left
zmap_r = np.zeros([405,405], dtype = float) # right
zmap_u = np.zeros([405,405], dtype = float) # up
zmap_d = np.zeros([405,405], dtype = float) # down
zmap = np.zeros([405,405], dtype = float) # final heightmap
for x in range(0, width):
    for y in range(0, height):
        R = img[y][x][2]
        G = img[y][x][1]
        B = img[y][x][0]
        zmap_l[x + 1][y] = zmap_l[x][y] + rgbmap[R][G][B][0]
        zmap_r[403 - x][y] = zmap_r[404 - x][y] + rgbmap[R][G][B][0]
        zmap_u[x][y + 1] = zmap_u[x][y] + rgbmap[R][G][B][1]
        zmap_d[x][403 - y] = zmap_d[x][404 - y] + rgbmap[R][G][B][1]
print(4)

# check error of gradient
error_sum = 0
for x in range(0, width):
    for y in range(0, height):
        R = img[y][x][2]
        G = img[y][x][1]
        B = img[y][x][0]
        error_sum += pow(rgbmap[R][G][B][0] - gradmap[x][y][0], 2) + pow(rgbmap[R][G][B][1] - gradmap[x][y][1], 2)

print("error sum: " + str(round(error_sum, 2)))

# summing by weight
for y in range(0, height):
    for x in range(0, width):
        zmap[x][y] = (zmap_l[x][y] * (404 - x) + zmap_r[x][y] * x + zmap_u[x][y] * (404 - y) + zmap_d[x][y] * y) / 808 # in 4 directions
        # zmap[x][y] = (zmap_l[x][y] * (404 - x) + zmap_r[x][y] * x) / 404 # in 2 directions
print(5)

# visualize heightmap 3D
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
x = np.arange(0, 405, 1)
y = np.arange(0, 405, 1)
x, y = np.meshgrid(x, y)
# ax.plot_surface(x, y, zmap, rstride = 5, cstride = 5, cmap = plt.get_cmap('rainbow'))
ax.contourf(x, y, zmap, zdir = 'z', offset = -2, cmap = 'rainbow') # draw isoheight
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D surface reconstruction')
print(6)
plt.show()

'''
# visualize heightmap 2D
x = np.arange(1,405)
y = zmap[202]
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.plot(x,y)
plt.show()
'''
