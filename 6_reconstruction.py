import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import dst, idst

# import gradients
grad = np.load( "grad.npy" )
gx = grad[:,:,0]
gy = grad[:,:,1]
ydim, xdim, dim = grad.shape

# create smallest enclosing rectangle
max_dim = max(xdim, ydim)
min_dim = min(xdim, ydim)

print(1)

'''
# Integration based on line integrals
zmap_l = np.zeros([width + 1, height + 1], dtype = float) # left
zmap_r = np.zeros([width + 1, height + 1], dtype = float) # right
zmap_u = np.zeros([width + 1, height + 1], dtype = float) # up
zmap_d = np.zeros([width + 1, height + 1], dtype = float) # down
zmap = np.zeros([width + 1, height + 1], dtype = float) # final heightmap
for x in range(0, width):
    for y in range(0, height):
        zmap_l[x + 1][y] = zmap_l[x][y] + grad[x][y][0]
        zmap_r[width - 1 - x][y] = zmap_r[width - x][y] + grad[x][y][0]
        zmap_u[x][y + 1] = zmap_u[x][y] + grad[x][y][1]
        zmap_d[x][height - 1 - y] = zmap_d[x][height - y] + grad[x][y][1]
# summing by weight
for x in range(0, width):
    for y in range(0, height):
        zmap[x][y] = (zmap_l[x][y] * (width - x) + zmap_r[x][y] * x + zmap_u[x][y] * (height - y) + zmap_d[x][y] * y) / (width + height) # in 4 directions
        # zmap[x][y] = (zmap_l[x][y] * (404 - x) + zmap_r[x][y] * x) / 404 # in 2 directions
# smoothing the heightmap
zmap = gaussian_filter(zmap, sigma = 3)
'''

# Integration based on DFT method
gxx = np.zeros((ydim, xdim))
gyy = np.zeros((ydim, xdim))
f = np.zeros((ydim, xdim))
gyy[1:,:-1] = gy[1:,:-1] - gy[:-1,:-1] # dG/dy
gxx[:-1,1:] = gx[:-1,1:] - gx[:-1,:-1] # dF/dx
f = gxx + gyy # sum of Laplacian
height, width = f.shape[:2]
f2 = f[1 : height - 1, 1: width - 1]
tt = dst(f2.T, type=1).T /2
f2sin = (dst(tt, type=1)/2)
x, y = np.meshgrid(np.arange(1, xdim-1), np.arange(1, ydim-1))
denom = (2*np.cos(np.pi * x/(xdim-1))-2) + (2*np.cos(np.pi*y/(ydim-1)) - 2)
f3 = f2sin/denom
tt = np.real(idst(f3, type=1, axis=0))/(f3.shape[0]+1)
img_tt = (np.real(idst(tt.T, type=1, axis=0))/(tt.T.shape[0]+1)).T
img_direct = np.zeros((ydim, xdim))
height, width = img_direct.shape[:2]
img_direct[1: height - 1, 1: width - 1] = img_tt
zmap = img_direct
zmap[zmap < 0] = 0

# TFLI

# reconstruction based on MG iterator

# zmap expansion
hmap = np.zeros([max_dim, max_dim], dtype = float)
#print(max_dim)
#print(min_dim)
lower = int((max_dim - min_dim) / 2)
upper = int(lower + min_dim)
#print(upper)
#print(lower)
if (max_dim == width):
    #print("yes")
    hmap[lower:upper,:] = zmap
else:
    #print("no")
    hmap[:,lower:upper] = zmap

print(2)
'''
# check error
p = np.zeros([width, height], dtype = float)
q = np.zeros([width, height], dtype = float)
e = np.zeros([width, height], dtype = float)
for x in range(0, ydim - 1):
    for y in range(0, xdim - 1):
        p[x][y] = zmap[x + 1][y] - zmap[x][y]
        q[x][y] = zmap[x][y + 1] - zmap[x][y]
        e[x][y] = pow(p[x][y] - grad[x][y][0], 2) + pow(q[x][y] - grad[x][y][1], 2)
print("sum of error is: " + str(np.sum(e)))
'''
# visual augmentation
zmap = hmap * 3

# visualize 3D height map
fig = plt.figure(figsize = (8,8), dpi = 80)
ax = Axes3D(fig, auto_add_to_figure = False)
ax = fig.add_subplot(1, 1, 1, projection='3d') # first plot
fig.add_axes(ax)
x = np.arange(0, max_dim, 1)
y = np.arange(0, max_dim, 1)
x, y = np.meshgrid(x, y)
#ax.plot_surface(x, y, zmap, rstride = 5, cstride = 5, cmap = plt.get_cmap('rainbow'))
ax.plot_surface(x, y, zmap, rstride = 5, cstride = 5)
#ax.contourf(x, y, e, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim3d(0, max_dim)
ax.set_ylim3d(0, max_dim)
ax.set_zlim3d(0, max_dim)
ax.set_title('surface reconstructed')

'''
ax = fig.add_subplot(1, 2, 2, projection='3d') # second plot
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, e, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
# ax.contourf(x, y, zmap, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('e')
ax.set_xlim3d(0, width)
ax.set_ylim3d(0, height)
ax.set_zlim3d(0, max_e)
ax.set_title('Integration errors')
'''

plt.show()
