# plotting
import matplotlib.pyplot as plt
import numpy as np

def show_side_by_side(images, titles, cmap="gray"):
    n = len(images)
    plt.figure(figsize=(4*n, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

