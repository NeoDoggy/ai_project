import cv2
import matplotlib.pyplot as plt
im = cv2.imread("./tmp.png",cv2.IMREAD_GRAYSCALE)
im=~im
plt.imshow(im)
plt.show()