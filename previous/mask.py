import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter

img = cv2.imread('./img/ball_light.jpg')
img = cv2.GaussianBlur(img,(7,7),0)
bg_img = cv2.imread('./img/background_light.jpg')
bg_img = cv2.GaussianBlur(bg_img,(7,7),0)
height, width = img.shape[0:2]
'''
dot_threshold = 1.5
mask = np.greater(np.sum(bg_img, axis = 2), dot_threshold).astype(float)
kernel = np.ones((5,5),np.uint8)
mask = cv2.erode(mask, kernel, iterations = 2)
'''
'''
plt.subplot(121),plt.imshow(img),plt.title('original')
plt.xticks([]), plt.yticks([])
'''
diff = np.zeros([width, height, 3],dtype=int)
for x in range(0, width):
    for y in range(0, height):
        diff[x][y][0] = int(img[y][x][0]) - int(bg_img[y][x][0])
        #if (diff[x][y][0] < 0):
            #diff[x][y][0] = 0
        diff[x][y][1] = int(img[y][x][1]) - int(bg_img[y][x][1])
        if (diff[x][y][1] < 0):
            diff[x][y][1] = 0
        diff[x][y][2] = int(img[y][x][2]) - int(bg_img[y][x][2])
        if (diff[x][y][2] < 0):
            diff[x][y][2] = 0
diff_r = diff[:,:,0]
diff_t = np.zeros([width, height],dtype=int)
for x in range(0, width):
    for y in range(0, height):
        diff_t[x][y] = diff[x][y][0] + diff[x][y][1] + diff[x][y][2]

diff_t = gaussian_filter(diff_t, sigma = 5)

mask1 = (diff_t > 5).astype('uint8')

fig = plt.figure(figsize = (8,8), dpi = 80)
ax = Axes3D(fig, auto_add_to_figure = False)

# print(diff.shape[0:2])

x = np.arange(0, height, 1)
y = np.arange(0, width, 1)
x, y = np.meshgrid(x, y)
#ax = fig.add_subplot(1, 1, 1, projection='3d') # first plot
#fig.add_axes(ax)
# ax.plot_surface(x, y, diff_t, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
#ax.contourf(x, y, diff_r, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
plt.xlabel('x')
plt.ylabel('y')
plt.title("relative intensity (Red)")
plt.imshow(diff_r, origin='lower', extent=[0, width, 0, height])
plt.colorbar()
plt.show()
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('diff')
#ax.set_xlim3d(0, width + 1)
#ax.set_ylim3d(0, height + 1)
#ax.set_zlim3d(0, height + 1)
#ax.set_title('difference from background')
#plt.show()
'''
# mask = np.logical_and(diff_R, diff_G, diff_B).astype(float)
mask = np.expand_dims(mask, 2)
mask = np.tile(mask, [1, 1, 3])
mask = img * mask
plt.subplot(122),plt.imshow(diff_R),plt.title('modified')
plt.colorbar()
plt.xticks([]), plt.yticks([])
plt.show()
'''
