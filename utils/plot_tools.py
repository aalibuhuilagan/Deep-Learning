# -*- coding: utf-8 -*-
# 绘图工具

import matplotlib.pyplot as plt

def plot_compare(images, titles):
    plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()