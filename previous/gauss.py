
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./ok/p3.jpg')

blur = cv2.GaussianBlur(img,(7,7),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
cv2.imwrite("./img/gauss.jpg", blur)
plt.show()
