import cv2
import numpy as np
import matplotlib.pyplot as plt

# import background and contact image
bg_img = cv2.imread("./img/set3/8.jpg")
raw_img = cv2.imread("./img/set3/10.jpg")
bg_img = cv2.GaussianBlur(bg_img, (3, 3), 0)
raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)

# rough estimation of contact zone using diff_img
diff_img = np.max(np.abs(raw_img.astype(np.float32) - bg_img),axis = 2)
contact_mask = (diff_img > 20).astype(np.uint8)
contours, _ = cv2.findContours(contact_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
areas = [cv2.contourArea(c) for c in contours]
sorted_areas = np.sort(areas)
cnt = contours[areas.index(sorted_areas[-1])] #the biggest contour
(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
print(center, radius)
height, width = diff_img.shape[0:2]

# precise location of contact zone by hand
key = -1
while key != 108: # l
    center = (int(x), int(y))
    radius = int(radius)
    im2show = cv2.circle(np.array(raw_img), center, radius, (0, 40, 0), 2)
    cv2.imshow('contact zone', im2show.astype(np.uint8))
    key = cv2.waitKey(0)
    if key == 119: # w
        y -= 1
    elif key == 115: # s
        y += 1
    elif key == 97: # a
        x -= 1
    elif key == 100: # d
        x += 1
    elif key == 106: # j
        radius += 1
    elif key == 107: # k
        radius -= 1

# create a numpy array for masking
contact_mask = np.zeros_like(contact_mask)
cv2.circle(contact_mask, center, radius, (1), -1)
'''
# show contact mask for confirmation
plt.title("masking area")
display = contact_mask
plt.imshow(display, origin='lower', extent=[0, width, 0, height])
plt.colorbar()
plt.show()
'''
# crop to square size
x, y, radius = int(x), int(y), int(radius)
raw_img = raw_img[y-radius : y+radius, x-radius : x+radius]
bg_img = bg_img[y-radius : y+radius, x-radius : x+radius]
contact_mask = contact_mask[y-radius : y+radius, x-radius : x+radius]

# save calibration parameters
cv2.imwrite("./calibration/bg_img.jpg", bg_img)
cv2.imwrite("./calibration/raw_img.jpg", raw_img)
np.save("./calibration/contact_mask.npy", contact_mask)
print("saving successful")
