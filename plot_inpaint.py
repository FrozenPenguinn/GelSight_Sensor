
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.restoration import inpaint

image_orig = cv2.imread("./testing/raw_img.jpg")
image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

# Create mask with three defect regions: left, middle, right respectively
mask = np.zeros(image_orig.shape[:-1])

for x in range(0,4):
    for y in range(0,4):
        mask[20*x+20:20*x+30, 20*y+15:20*y+25] = 1

# Defect image over the same region in each color channel
image_defect = image_orig.copy()
for layer in range(image_defect.shape[-1]):
    image_defect[np.where(mask)] = 0

image_result = inpaint.inpaint_biharmonic(image_defect, mask, multichannel=True)

fig, axes = plt.subplots(ncols=2, nrows=2)
ax0, ax1, ax2, ax3 = axes.ravel()

ax0.set_title('Original image')
ax0.imshow(image_orig)
ax0.axis('off')

ax1.set_title('Mask')
ax1.imshow(mask, cmap=plt.cm.gray)
ax1.axis('off')

ax2.set_title('Defected image')
ax2.imshow(image_defect)
ax2.axis('off')

ax3.set_title('Inpainted image')
ax3.imshow(image_result)
ax3.axis('off')

fig.tight_layout()
plt.show()
