import matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import color
from os import listdir
import os
# https://web.mit.edu/sinhalab/Papers/yip_sinha_ColorFaces.pdf
# paper for accuracy of histograms with diffrent color filters

average_img1 = 0
# average_img2 = 0
# average_img3 = 0
# average_img4 = 0

img = cv2.imread("img/output-mens-hoogte(yolov8m-seg)/19-Opname mens hoogte-seg172.jpg")
for d in os.listdir("C:/Users/nickt/Desktop/Research/img"):
    print(d)
    for images in os.listdir(f"C:/Users/nickt/Desktop/Research/img/{d}"):
        
        img2 = cv2.imread(f"C:/Users/nickt/Desktop/Research/img/{d}/{images}")

        average_img1 = 0
        test = hog(
            img,
            orientations=8,
            channel_axis=-1,
        )
        test2 = hog(
        img2,
        orientations=8,
        channel_axis=-1,
        )

        colors = ("b", "g", "r")
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            hist[0] = 0

            hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
            hist2[0] = 0

            base_base = cv2.compareHist(hist, hist, 0)
            base_test1 = cv2.compareHist(hist, hist2, 0)
            average_img1 += base_test1

            # print("color:", color, "img 1:", base_test1)

        eind_result_1 = average_img1 / 3
        if(eind_result_1 > 0.75):
            print(f"{d}/{images}")
            print("Combined simularities img 1: ", eind_result_1)
print("finished")
# plt.plot(test)
# plt.title("Image Histogram GFG")
# plt.show()
