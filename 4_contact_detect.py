import cv2
import numpy as np
import matplotlib.pyplot as plt

# import background and contact image
bg_img = cv2.imread("./img/set3/8.jpg")
raw_img = cv2.imread("./img/set3/10.jpg")
bg_img = cv2.GaussianBlur(bg_img, (3, 3), 0)
raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)

# Select ROI
roi = cv2.selectROI("Image", raw_img, showCrosshair=False, fromCenter=False)
# Crop image
raw_img_crop = raw_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
bg_img_crop = bg_img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# save calibration parameters
cv2.imwrite("./testing/bg_img.jpg", bg_img_crop)
cv2.imwrite("./testing/raw_img.jpg", raw_img_crop)
print("saving successful")
