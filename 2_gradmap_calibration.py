# newest: relative color intensity
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, pi
from scipy.optimize import leastsq
from scipy import interpolate

# import training/calibration image, background, and mask
raw_img = cv2.imread("./calibration/raw_img.jpg")
raw_img = cv2.GaussianBlur(raw_img,(3,3),0)
bg_img = cv2.imread("./calibration/bg_img.jpg")
bg_img = cv2.GaussianBlur(bg_img,(3,3),0)
contact_mask = np.load("./calibration/contact_mask.npy")

# get difference in seperate color channels
diff_img = np.zeros_like(contact_mask)
diff_img = (raw_img.astype(int) - bg_img.astype(int))
#print(diff_img.shape)
#print(contact_mask.shape)
for color_channel in range(0, 3):
    diff_img[:,:,color_channel] = diff_img[:,:,color_channel] * contact_mask
'''
# testing
display = diff_img[:,:,2]
plt.imshow(display)
plt.colorbar()
plt.show()
'''
# calibration image data
height, width = raw_img.shape[0:2]
cy = height / 2
cx = width / 2
p2mm = 58.16
radius = 4 * p2mm # the actual radius of the calibration sphere
r = cx # the contact circle radius

print("1")

# gradient table corresponding to calibration image
calgrads = np.zeros([width, height, 2], dtype = float) # 0:dz/dx; 1:dz/dy
gradmod = np.zeros([width, height], dtype = float)
hmap = np.zeros([width, height], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        if (contact_mask[x,y] != 0):
            calgrads[y][x][0] = - (x - cx) * pow((pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2)), -0.5)
            calgrads[y][x][1] = - (y - cy) * pow((pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2)), -0.5)
            gradmod[y][x] = sqrt(pow(calgrads[y][x][0],2) + pow(calgrads[y][x][1],2))
            # hmap[x][y] = sqrt(pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2))

# for later convinence
gx = calgrads[:,:,0]
gy = calgrads[:,:,1]
'''
# testing
plt.xlabel('x')
plt.ylabel('y')
display = gx
plt.imshow(display)
plt.show()
'''
print(2)

# record color channels
colors = np.zeros([width, height, 3], dtype = int)
colors[:,:,0] = diff_img[:,:,2] # red
colors[:,:,1] = diff_img[:,:,1] # green
colors[:,:,2] = diff_img[:,:,0] # blue

# Grad-RGB map
gradmap = np.zeros([401,401,3], dtype = float)
countmap = np.zeros([401,401], dtype = int)
for x in range(0, width):
    for y in range(0, height):
        if (contact_mask[x,y] != 0):
            dx = np.clip(floor(gx[x][y] * 100) + 200, 0, 400) # limit dx to within 0 and 200
            dy = np.clip(floor(gy[x][y] * 100) + 200, 0, 400)
            gradmap[dx][dy][0] = (gradmap[dx][dy][0] * countmap[dx][dy] + colors[y][x][0]) / (countmap[dx][dy] + 1)
            gradmap[dx][dy][1] = (gradmap[dx][dy][1] * countmap[dx][dy] + colors[y][x][1]) / (countmap[dx][dy] + 1)
            gradmap[dx][dy][2] = (gradmap[dx][dy][2] * countmap[dx][dy] + colors[y][x][2]) / (countmap[dx][dy] + 1)
            countmap[dx][dy] += 1

# gradmap = np.round(gradmap).astype(int)
'''
# testing
plt.xlabel('x')
plt.ylabel('y')
display = gradmap[:,:,0]
plt.imshow(display)
plt.show()
'''

print("3")

# using spherical harmonics model to populate gradmap
harmonics = np.zeros([width, height, 9], dtype = float)
for x in range(0, width):
    for y in range(0, height):
        if (contact_mask[x,y] != 0):
            z = sqrt(pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2))
            harmonics[x][y][0] = sqrt(1/(4*pi)) # Y00
            harmonics[x][y][1] = sqrt(3/(4*pi)) * (y/radius) # Y1-1
            harmonics[x][y][2] = sqrt(3/(4*pi)) * (z/radius) # Y10
            harmonics[x][y][3] = sqrt(3/(4*pi)) * (x/radius) # Y1+1
            harmonics[x][y][4] = (1/2)*sqrt(15/pi) * (x*y/pow(radius,2)) # Y2-2
            harmonics[x][y][5] = (1/2)*sqrt(15/pi) * (y*z/pow(radius,2)) # Y2-1
            harmonics[x][y][6] = (1/4)*sqrt(5/pi) * (3*pow(z,2) - pow(radius,2))/pow(radius,2) # Y20
            harmonics[x][y][7] = (1/2)*sqrt(15/pi) * (z*x/pow(radius,2)) # Y2+1
            harmonics[x][y][8] = (1/4)*sqrt(15/pi) * (pow(x,2) - pow(y,2))/pow(radius,2) # Y2+2
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
    coeff_guess = [0,0,0,0,0,0,0,0,0]
    para = leastsq(error, coeff_guess, args=(harmonics_flaten, target))
    coeff_sh[:,channel] = para[0]
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = para[0]
    color_sim[:,:,channel] = (c1*h1 + c2*h2 + c3*h3 + c4*h4 + c5*h5 + c6*h6 + c7*h7 + c8*h8 + c9*h9).reshape(width, height)

print("4")

# find points on cal img that has corresponding missing gradients in gradmap
# from gradient get coordinates
def get_xy(dy, dx):
    a = sqrt((pow(radius,2)*pow(dx,2))/(1+pow(dx,2)+pow(dy,2)))
    b = sqrt((pow(radius,2)*pow(dy,2))/(1+pow(dx,2)+pow(dy,2)))
    if (dx > 0 and dy > 0):
        return cx-a, cy-b
    elif (dx > 0 and dy <= 0):
        return cx-a, cy+b
    elif (dx <= 0 and dy > 0):
        return cx+a, cy-b
    elif (dx <= 0 and dy <= 0):
        return cx+a, cy+b

for channel in range(0,3):
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeff_sh[:,channel]
    for i in range(0, 401):
        for j in range(0, 401):
            #if (countmap[i][j] == 0): # only fill in the empty ones
            dx = (i - 200) / 100
            dy = (j - 200) / 100
            x, y = get_xy(dx, dy)
            z = sqrt(pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2))
            # most xyz aren't int, so can't use previous harmonic matrix
            h1 = sqrt(1/(4*pi)) # Y0
            h2 = sqrt(3/(4*pi)) * (y/radius) # Y1n1
            h3 = sqrt(3/(4*pi)) * (z/radius) # Y10
            h4 = sqrt(3/(4*pi)) * (x/radius) # Y1p1
            h5 = (1/2)*sqrt(15/pi) * (x*y/pow(radius,2)) # Y2n2
            h6 = (1/2)*sqrt(15/pi) * (y*z/pow(radius,2)) # Y2n1
            h7 = (1/4)*sqrt(5/pi) * (3*pow(z,2) - pow(radius,2))/pow(radius,2) # Y20
            h8 = (1/2)*sqrt(15/pi) * (z*x/pow(radius,2)) # Y2p1
            h9 = (1/4)*sqrt(15/pi) * (pow(x,2) - pow(y,2))/pow(radius,2) # Y2p2
            I = c1*h1 + c2*h2 + c3*h3 + c4*h4 + c5*h5 + c6*h6 + c7*h7 + c8*h8 + c9*h9
            gradmap[j][i][channel] = I

# testing
plt.xlabel('dx')
plt.ylabel('dy')
display = gradmap[:,:,0]
plt.imshow(display)
plt.show()

print("5")

# adding dI/dR for each channel
final_gradmap = np.zeros([401,401,3,3],dtype = float)
final_gradmap[:,:,:,0] = gradmap[:,:,:]
dR = np.gradient(gradmap[:,:,0]) # dR[0][:,:] gives p, dR[1][:,:] gives q
#print(len(dR))
#print(dR[0][0,0])
#print(dR[1][0,0])
#print(gradmap[0:2,0:2,0])
dG = np.gradient(gradmap[:,:,1])
dB = np.gradient(gradmap[:,:,2])
for i in range(0, 401):
    for j in range(0, 401):
        # dR/dp and dR/dq
        final_gradmap[i, j, 0, 1] = dR[0][j, i] # Transpose
        final_gradmap[i, j, 0, 2] = dR[1][j, i]
        final_gradmap[i, j, 1, 1] = dG[0][j, i]
        final_gradmap[i, j, 1, 2] = dG[1][j, i]
        final_gradmap[i, j, 2, 1] = dB[0][j, i]
        final_gradmap[i, j, 2, 2] = dB[1][j, i]

# saving the gradmap
np.save("gradmap.npy", final_gradmap)
print("gradmap saved")
