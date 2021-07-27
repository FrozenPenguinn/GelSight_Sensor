# spherical harmonic model implementation

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, pi
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter, median_filter
import scipy.linalg
from scipy.optimize import leastsq

# import training/calibration image
img = cv2.imread("./ok/p3.jpg")
height, width = img.shape[0:2]
center_h = int(height / 2)
center_w = int(height / 2)
radius = int(width / 2)
r = 120 # 60 pix/mm

print(1)
# gradient table corresponding to calibration image
calgrads = np.zeros([width, height, 2], dtype = float) # 0:dz/dx; 1:dz/dy
hmap = np.zeros([width + 1, height + 1], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        if (pow(x - center_w, 2) + pow(y - center_h, 2) < pow(radius, 2)):
            calgrads[x][y][0] = - (x - center_w) * pow((pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2)), -0.5)
            calgrads[x][y][1] = - (y - center_h) * pow((pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2)), -0.5)
            # hmap[x][y] = sqrt(pow(radius, 2) - pow((x - center_w), 2) - pow((y - center_h), 2))

gx = calgrads[:,:,0]
gy = calgrads[:,:,1]

print(2)

# record color channels
colors = np.zeros([width, height, 3], dtype = int)
colors[:,:,0] = img[:,:,2]
colors[:,:,1] = img[:,:,1]
colors[:,:,2] = img[:,:,0]

# Grad-GRB map
gradmap = np.zeros([601,601,3], dtype = float)
countmap = np.zeros([601,601], dtype = int)
for x in range(0, width):
    for y in range(0, height):
        dx = np.clip(floor(gx[x][y] * 100) + 300, 0, 600) # limit dx to within 0 and 600
        dy = np.clip(floor(gy[x][y] * 100) + 300, 0, 600)
        gradmap[dx][dy][0] = (gradmap[dx][dy][0] * countmap[dx][dy] + colors[y][x][0]) / (countmap[dx][dy] + 1)
        gradmap[dx][dy][1] = (gradmap[dx][dy][1] * countmap[dx][dy] + colors[y][x][1]) / (countmap[dx][dy] + 1)
        gradmap[dx][dy][2] = (gradmap[dx][dy][2] * countmap[dx][dy] + colors[y][x][2]) / (countmap[dx][dy] + 1)
        countmap[dx][dy] += 1

# saving for later comparison
old_gradmap = gradmap.copy()

# using spherical harmonics model to populate gradmap
harmonics = np.zeros([width, height, 9], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        if (pow(x - center_w, 2) + pow(y - center_h, 2) < pow(r, 2)):
            z = sqrt(pow(r, 2) - pow((x - center_w), 2) - pow((y - center_h), 2))
            harmonics[x][y][0] = sqrt(1/(4*pi)) # Y00
            harmonics[x][y][1] = sqrt(3/(4*pi)) * (y/r) # Y1-1
            harmonics[x][y][2] = sqrt(3/(4*pi)) * (z/r) # Y10
            harmonics[x][y][3] = sqrt(3/(4*pi)) * (x/r) # Y1+1
            harmonics[x][y][4] = (1/2)*sqrt(15/pi) * (x*y/pow(r,2)) # Y2-2
            harmonics[x][y][5] = (1/2)*sqrt(15/pi) * (y*z/pow(r,2)) # Y2-1
            harmonics[x][y][6] = (1/4)*sqrt(5/pi) * (3*pow(z,2) - pow(r,2))/pow(r,2) # Y20
            harmonics[x][y][7] = (1/2)*sqrt(15/pi) * (z*x/pow(r,2)) # Y2+1
            harmonics[x][y][8] = (1/4)*sqrt(15/pi) * (pow(x,2) - pow(y,2))/pow(r,2) # Y2+2
        else:
            colors[x,y,:] = 0

# using least square to solve for coeffs
harmonics_flaten = harmonics.reshape(-1, harmonics.shape[-1]).T
h1, h2, h3, h4, h5, h6, h7, h8, h9 = harmonics_flaten
coeff_sh = np.zeros([9,3], dtype = float)
color_sim = np.zeros([width, height, 3], dtype = float) # img in simulation lighting, in contrast with real lighting
# define R(N)=sum(coeff(i,j)*Y(i,j)) and error function
def func(harmonics, coeff):
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeff
    h1, h2, h3, h4, h5, h6, h7, h8, h9 = harmonics
    return c1*h1 + c2*h2 + c3*h3 + c4*h4 + c5*h5 + c6*h6 + c7*h7 + c8*h8 + c9*h9
def error(coeff, harmonics, target):
    return target - func(harmonics, coeff)
for channel in range(0, 3):
    # set target to one of the channels and solve for its coeffs
    target = colors[:,:,channel].flatten()
    coeff_guess = [1,1,1,1,1,1,1,1,1]
    para = leastsq(error, coeff_guess, args=(harmonics_flaten, target))
    coeff_sh[:,channel] = para[0]
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = para[0]
    color_sim[:,:,channel] = (c1*h1 + c2*h2 + c3*h3 + c4*h4 + c5*h5 + c6*h6 + c7*h7 + c8*h8 + c9*h9).reshape(width, height)

# find points on cal img that has corresponding missing gradients in gradmap
new_gradmap = old_gradmap.copy()
# from gradient get coordinates
def get_xy(dy, dx):
    a = sqrt((pow(r,2)*pow(dx,2))/(1+pow(dx,2)+pow(dy,2)))
    b = sqrt((pow(r,2)*pow(dy,2))/(1+pow(dx,2)+pow(dy,2)))
    if (dx > 0 and dy > 0):
        return center_w-a, center_h-b
    elif (dx > 0 and dy <= 0):
        return center_w-a, center_h+b
    elif (dx <= 0 and dy > 0):
        return center_w+a, center_h-b
    elif (dx <= 0 and dy <= 0):
        return center_w+a, center_h+b
for channel in range(0,3):
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeff_sh[:,channel]
    for i in range(0, 601):
        for j in range(0, 601):
            if (old_gradmap[i][j][channel] == 0):
                dx = (i - 300)/100
                dy = (j - 300)/100
                x, y = get_xy(dx, dy)
                z = sqrt(pow(r, 2) - pow((x - center_w), 2) - pow((y - center_h), 2))
                # most xyz aren't int, so can't use previous harmonic matrix
                h1 = sqrt(1/(4*pi)) # Y0
                h2 = sqrt(3/(4*pi)) * (y/r) # Y1n1
                h3 = sqrt(3/(4*pi)) * (z/r) # Y10
                h4 = sqrt(3/(4*pi)) * (x/r) # Y1p1
                h5 = (1/2)*sqrt(15/pi) * (x*y/pow(r,2)) # Y2n2
                h6 = (1/2)*sqrt(15/pi) * (y*z/pow(r,2)) # Y2n1
                h7 = (1/4)*sqrt(5/pi) * (3*pow(z,2) - pow(r,2))/pow(r,2) # Y20
                h8 = (1/2)*sqrt(15/pi) * (z*x/pow(r,2)) # Y2p1
                h9 = (1/4)*sqrt(15/pi) * (pow(x,2) - pow(y,2))/pow(r,2) # Y2p2
                I = c1*h1 + c2*h2 + c3*h3 + c4*h4 + c5*h5 + c6*h6 + c7*h7 + c8*h8 + c9*h9
                new_gradmap[i][j][channel] = I

# saving the gradmap
np.save("gradmap.npy", new_gradmap)
print("gradmap saved")

'''
# visualize gradient-intensity table (3D)
fig = plt.figure(figsize = (8,8), dpi = 80)
ax = Axes3D(fig, auto_add_to_figure = False)
x = np.arange(-3, 3.01, 0.01)
y = np.arange(-3, 3.01, 0.01)
x, y = np.meshgrid(x, y)
ax = fig.add_subplot(1, 1, 1, projection='3d') # first plot
fig.add_axes(ax)
ax.plot_surface(x, y, old_red, rstride = 10, cstride = 10, cmap = plt.get_cmap('rainbow'))
# ax.contourf(x, y, red, zdir = 'z', offset = -1, cmap = 'rainbow') # draw isoheight
# ax.scatter(x, y, red, c='r')
ax.set_xlabel('p')
ax.set_ylabel('q')
ax.set_zlabel('R')
#ax.set_xlim3d(0, width + 1)
#ax.set_ylim3d(0, height + 1)
#ax.set_zlim3d(0, height + 1)
ax.set_title('gradient-intensity map')
plt.show()
'''
'''
display = new_gradmap[:,:,0]
# MIT style
x = np.arange(-3, 3.01, 0.5)
y = np.arange(-3, 3.01, 0.5)
x, y = np.meshgrid(x, y)
plt.xlabel('p')
plt.ylabel('q')
# plt.title("gradient-intensity map (Red)")
plt.imshow(display, origin='lower', extent=[-3, 3, -3, 3])
plt.colorbar()
plt.show()
'''
'''
# initialize rgbmap
countmap = np.zeros([255,255,255], dtype = int)
rgbmap = np.zeros([255, 255, 255, 2], dtype = float)
test = np.zeros([1054,2], dtype = float)
for x in range(0, 601):
    for y in range(0, 601):
        R = int(new_gradmap[x][y][0])
        G = int(new_gradmap[x][y][1])
        B = int(new_gradmap[x][y][2])
        # dz/dx
        rgbmap[R][G][B][0] = (rgbmap[R][G][B][0] * countmap[R][G][B] + ((x - 300) / 100)) / (countmap[R][G][B] + 1)
        # dz/dy
        rgbmap[R][G][B][1] = (rgbmap[R][G][B][1] * countmap[R][G][B] + ((y - 300) / 100)) / (countmap[R][G][B] + 1)
        # countmap
        if (R == 124 and G == 94 and B == 144):
            count = countmap[R][G][B]
            test[count][0] = ((x - 300) / 100)
            test[count][1] = ((y - 300) / 100)
        countmap[R][G][B] += 1

#print(rgbmap[124][94][144][0])
#print(rgbmap[124][94][144][1])
plt.scatter(test[:,0], test[:,1])
plt.show()
'''
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


# saving the rgbmap
np.save("rgbmap.npy", rgbmap)
print("rgmap saved")
'''
'''
# using the rgbmap
rgbmap = np.load("rgbmap.npy")
'''
'''
# using another image for testing
img = cv2.imread("./ok/p3.jpg")
img = cv2.GaussianBlur(img,(7,7),0)
height, width = img.shape[0:2]

# gradient map of the testing img
grad = np.zeros([width, height, 2], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        R = img[y][x][2]
        G = img[y][x][1]
        B = img[y][x][0]
        grad[x][y][0] = rgbmap[R][G][B][0]
        grad[x][y][1] = rgbmap[R][G][B][1]

# saving the rgbmap
np.save("grad.npy", grad)
print("grad saved")
'''
