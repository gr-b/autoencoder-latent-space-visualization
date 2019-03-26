import matplotlib.pyplot as plt
import numpy as np

img1 = np.random.rand(28, 28)
img2 = np.random.rand(28, 28)

fig, axes = plt.subplots()

ax1 = plt.subplot(1, 2, 1)
ax1.imshow(img1, cmap="gray")

ax2 = plt.subplot(1, 3, 1)
ax2.imshow(img2, cmap="gray")

plt.show()

