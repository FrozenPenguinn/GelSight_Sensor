import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, pi

# import background and contact image
bg_img = cv2.imread("./calibration/bg_img.jpg")
raw_img = cv2.imread("./calibration/raw_img.jpg")
bg_img = cv2.GaussianBlur(bg_img, (3, 3), 0)
raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)

diff_img = np.zeros_like(bg_img)
diff_img = (raw_img.astype(int) - bg_img.astype(int))

height, width = raw_img.shape[0:2]
center_y = int(height / 2)
center_x = int(width / 2)
p2mm = 58.16
radius = 4 * p2mm

# cut out a line segment of the difference image
color_change = diff_img[center_y,:,2]
# find gradients in form of degrees
degrees = np.zeros([width], dtype = float)
for x in range(0, width):
    degrees[x] = atan2(- (x - center_x) * pow((pow(radius, 2) - pow((x - center_x), 2)), -0.5), 1) * 180 / pi

print(color_change[0])
y = np.linspace(-152, 152, 304)

plt.title("Our Sensor")
plt.xlabel('Color Change')
plt.ylabel('Surface Normal Pitch (deg)')
plt.plot(color_change, degrees, 'x', color='blue')
#display = diff_img[:,:,2]
#plt.imshow(display)
plt.show()
