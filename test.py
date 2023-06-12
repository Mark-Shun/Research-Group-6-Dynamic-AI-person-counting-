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
colors = ("b", "g", "r")




for im in os.listdir("C:/Users/nickt/Desktop/Research/img/dataset/1"):
    persoon_id = im[0]
    img = cv2.imread(f"C:/Users/nickt/Desktop/Research/img/dataset/1/{im}")
    id_counter = 0
    id_sum_average = 0
    counter = 0
    sum_average = 0
    
    all_count = 0
    all_sum_average = 0
    all_id_sum_average = 0

    for d in os.listdir("C:/Users/nickt/Desktop/Research/img/organised-data"):        
        for images in os.listdir(f"C:/Users/nickt/Desktop/Research/img/organised-data/{d}"):
            
            img2 = cv2.imread(f"C:/Users/nickt/Desktop/Research/img/organised-data/{d}/{images}")
            average_img1 = 0
            
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                hist[0] = 0

                hist2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
                hist2[0] = 0

                base_base = cv2.compareHist(hist, hist, 0)
                base_test1 = cv2.compareHist(hist, hist2, 0)
                average_img1 = average_img1 + base_test1

            eind_result_1 = average_img1 / 3
            if(persoon_id == d):
                id_counter += 1
                id_sum_average += eind_result_1
            else:
                counter += 1
                sum_average += eind_result_1

                # print(f"{d}/{images}")
                # print("Combined simularities img 1: ", eind_result_1)
    result_sum_average = sum_average/counter
    result_id_sum_average = id_sum_average/id_counter
    print(f"Not person:{result_sum_average}")        
    print(f"Right person:{result_id_sum_average}\n")

    all_count += 1
    all_sum_average += result_sum_average
    all_id_sum_average += result_id_sum_average
    
print(f"ALL Not person:{all_sum_average/all_count}")        
print(f"ALL Right person:{all_id_sum_average/all_count}")
print("finished")
# plt.plot(test)
# plt.title("Image Histogram GFG")
# plt.show()
