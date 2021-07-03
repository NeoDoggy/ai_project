import cv2
import matplotlib.pyplot as plt
im=cv2.imread("./tmp.png",cv2.IMREAD_GRAYSCALE)
im=cv2.resize(im,(28,28),interpolation=cv2.INTER_NEAREST)
im=~im
print(im.flatten())
plt.imshow(im)
plt.show()