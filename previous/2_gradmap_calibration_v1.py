# newest: relative color intensity
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, pi
from scipy.optimize import leastsq

# import training/calibration image and background
img = cv2.imread("./img/set2/p3.jpg")
img = cv2.GaussianBlur(img,(3,3),0) # same as MIT GelSlim
#bg_img = cv2.imread("./img/background_light.jpg")
#bg_img = cv2.GaussianBlur(bg_img,(3,3),0)
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

print("3")

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

print("4")

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

print("5")

# adding dI/dR for each channel
final_gradmap = np.zeros([601,601,3,3],dtype = float)
final_gradmap[:,:,:,0] = new_gradmap[:,:,:]
dR = np.gradient(new_gradmap[:,:,0]) # dR[0][:,:] gives p, dR[1][:,:] gives q
dG = np.gradient(new_gradmap[:,:,1])
dB = np.gradient(new_gradmap[:,:,2])
for i in range(0, 601):
    for j in range(0, 601):
        # dR/dp and dR/dq
        final_gradmap[i, j, 0, 1] = dR[0][i, j]
        final_gradmap[i, j, 0, 2] = dR[1][i, j]
        final_gradmap[i, j, 1, 1] = dG[0][i, j]
        final_gradmap[i, j, 1, 2] = dG[1][i, j]
        final_gradmap[i, j, 2, 1] = dB[0][i, j]
        final_gradmap[i, j, 2, 2] = dB[1][i, j]

# saving the gradmap
np.save("gradmap.npy", final_gradmap)
print("gradmap saved")
