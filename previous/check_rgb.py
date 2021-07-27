from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import cv2
import numpy as np
import matplotlib.pyplot as plt

# import training image
img = cv2.imread("./img/2.jpg")
height, width = img.shape[0:2]

red = np.zeros([width, height], dtype = int)
green = np.zeros([width, height], dtype = int)
blue = np.zeros([width, height], dtype = int)

for x in range(0,width):
    for y in range(0,height):
        red[x][y] = int(img[y,x][2])
        green[x][y] = int(img[y,x][1])
        blue[x][y] = int(img[y,x][0])

# 3D plotting


fig = plt.figure(figsize = (9,3), dpi = 120)
ax = Axes3D(fig, auto_add_to_figure = False)
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x, y = np.meshgrid(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_surface(x, y, red, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
ax.set_title('Red')

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot_surface(x, y, green, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
ax.set_title('Green')

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(x, y, blue, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
ax.set_title('Blue')

plt.show()
