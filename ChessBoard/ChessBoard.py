import numpy as np
import matplotlib.pyplot as plt

def fill(i, j, img):
    c = 32 * i
    d = 32 * (i + 1)
    e = 32 * j
    f = 32 * (j + 1)
    for x in range(c, d):
        for y in range(e, f):
            img[x, y] = 255
            
img = np.zeros([256, 256])
for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 0:
            fill(i, j, img)

plt.imshow(img, cmap='gray')
plt.show()