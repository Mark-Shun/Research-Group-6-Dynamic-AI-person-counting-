import matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import color
from os import listdir
import os
import pickle


with open("results.pkl", "rb") as cp_file:
    results = pickle.load(cp_file)



min = min([result[1] for result in results])
max = max([result[1] for result in results])

predictions =[]
predictions2 =[]
thresholds = []


for threshold in np.arange(min, max, 0.1):
    threshold = round(threshold,1)
    thresholds += [threshold]
    val = 0
    for result in results:
        prediction = False
        if result[1] < threshold:
            prediction = True
        if prediction == result[0]:
            val += 1
    predictions += [val/len(results)]

plt.plot(thresholds, predictions)
plt.show()