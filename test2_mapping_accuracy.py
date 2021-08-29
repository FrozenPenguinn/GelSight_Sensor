import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import atan2, pi, sqrt

# import background and contact image
bg_img = cv2.imread("./testing/bg_img.jpg")
raw_img = cv2.imread("./testing/raw_img.jpg")
bg_img = cv2.GaussianBlur(bg_img, (3, 3), 0)
raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)

diff_img = np.zeros_like(bg_img)
diff_img = (raw_img.astype(int) - bg_img.astype(int))

height, width = raw_img.shape[0:2]
center_y = int(height / 2)
center_x = int(width / 2)
p2mm = 58.16
radius = 2 * p2mm

# cut out a line segment of the difference image
color_change = diff_img[center_y,:,2]
# find gradients in form of degrees
degrees = np.zeros([width], dtype = float)
for x in range(0, width):
    degrees[x] = atan2(- (x - center_x) * pow((pow(radius, 2) - pow((x - center_x), 2)), -0.5), 1) * 180 / pi

# get gradients with current calibration image
grad = np.load( "grad.npy" )
gx = grad[:,:,0]
gy = grad[:,:,1]
ydim, xdim = height, width
cy, cx, r = ydim/2, xdim/2, xdim/2

# Yaw (direction of gradient), Pitch (magnitude of gradient)
'''
abs = np.sqrt(gx**2 + gy**2)
u = np.divide(gx, abs, out=np.zeros_like(gx), where=abs!=0)
v = np.divide(gy, abs, out=np.zeros_like(gy), where=abs!=0)
'''
pitch = np.zeros([ydim, xdim], dtype = float)
yaw = np.zeros([ydim, xdim], dtype = float)
pitch_ground = np.zeros([ydim, xdim], dtype = float)
yaw_ground = np.zeros([ydim, xdim], dtype = float)
gradx_ground = np.zeros([ydim, xdim], dtype = float)
grady_ground = np.zeros([ydim, xdim], dtype = float)
for x in range(0, xdim):
    for y in range(0, ydim):
        if (pow(x-cx,2)+pow(y-cy,2) <= pow(r,2)):
            # intermediate
            gradx_ground[y][x] = - (x - cx) * pow((pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2)), -0.5)
            grady_ground[y][x] = - (y - cy) * pow((pow(radius, 2) - pow((x - cx), 2) - pow((y - cy), 2)), -0.5)
            # ground truth values
            pitch_ground[y, x] = atan2(sqrt(pow(gradx_ground[y, x],2) + pow(grady_ground[y, x],2)),1)  * 180 / pi
            yaw_ground[y, x] = atan2(grady_ground[y, x], gradx_ground[y, x]) * 180 / pi
            # testing values
            pitch[y, x] = atan2(sqrt(pow(gx[y, x],2) + pow(gy[y, x],2)),1) * 180 / pi
            yaw[y, x] = atan2(gy[y, x], gx[y, x]) * 180 / pi

# flatten
pitch_flat = pitch.flatten()
pitch_ground_flat = pitch_ground.flatten()
yaw_flat = yaw.flatten()
yaw_ground_flat = yaw_ground.flatten()

# correction
for index in range(0, yaw_flat.shape[0]):
    if (yaw_ground_flat[index] > 0 and yaw_flat[index] < -50):
        yaw_flat[index] = - yaw_flat[index]

# ground truth line
line1_x = [0,60]
line1_y = [0,60]
line2_x = [-200,200]
line2_y = [-200,200]

# find r_squared
r_squared_pitch = "$r^{2}$ = " + str(round(pow(np.corrcoef(pitch_ground_flat, pitch_flat)[0,1],2),3))
r_squared_yaw = "$r^{2}$ = " + str(round(pow(np.corrcoef(yaw_ground_flat, yaw_flat)[0,1],2),3))

matplotlib.rc('figure', figsize=(10, 5))
fig, axes = plt.subplots(1, 2)
# set labels
plt.setp(axes[:], xlabel='Ground Truth')
plt.setp(axes[:], ylabel='Measured Value')
axes[0].set_title("Pitch of Surface Normal (deg)")
axes[1].set_title("Yaw of Surface Normal (deg)")
axes[0].plot(pitch_ground_flat, pitch_flat, 'o', color='blue', markersize=1)
axes[0].plot(line1_x, line1_y, color='red')
axes[0].text(28, -0.5, r_squared_pitch, fontsize=10,  color='black')
axes[1].plot(yaw_ground_flat, yaw_flat, 'o', color='blue', markersize=1)
axes[1].plot(line2_x, line2_y, color='red')
axes[1].text(100, -205, r_squared_yaw, fontsize=10,  color='black')
plt.show()
