import cv2
import numpy as np
import matplotlib.pyplot as plt

# import background and contact image
bg_img = cv2.imread("./calibration/bg_img.jpg")
raw_img = cv2.imread("./calibration/raw_img.jpg")
depth = np.load("./calibration/depth.npy")
grad = np.load("./calibration/grad.npy")
print(grad.shape)
#bg_img = cv2.GaussianBlur(bg_img, (3, 3), 0)
#raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)

# simulation
#bg_img = np.load("./sim_img/0_color.npy")[0,:,:,:]
#raw_img = np.load("./sim_img/1_color.npy")[0,:,:,:]

# Select ROI
roi = cv2.selectROI("Image", raw_img, showCrosshair=False, fromCenter=False)
# Crop image
raw_img_crop = raw_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
bg_img_crop = bg_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
depth = depth[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
grad = grad[:,int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# save calibration parameters
cv2.imwrite("./testing/bg_img.jpg", bg_img_crop)
cv2.imwrite("./testing/raw_img.jpg", raw_img_crop)
np.save("./testing/original_depth.npy", depth)
np.save("./testing/original_grad.npy", grad)
print("saving successful")
