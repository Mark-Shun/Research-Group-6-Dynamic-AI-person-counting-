import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

img = cv2.imread('nic2.jpg')
img2 = cv2.imread('nic1.jpg')

assert img is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(img2)

histograms = []
hist1rgb = []
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    histr[0] = 0
    plt.subplot(223), plt.plot(histr,color = col)
    plt.xlim([0,256])
    histograms.append(histr)
hist1rgb = np.array(list(zip(histograms[0], histograms[1], histograms[2])))

histograms2 = []
hist2rgb  = []
for i,col in enumerate(color):
    histr2 = cv2.calcHist([img2],[i],None,[256],[0,256])
    histr2[0] = 0
    plt.subplot(224), plt.plot(histr2,color = col)
    plt.xlim([0,256])
    histograms2.append(histr2)
hist2rgb = np.array(list(zip(histograms2[0], histograms2[1], histograms2[2])))

plt.show()

for compare_method in range(4):
    base_base = cv2.compareHist(hist1rgb, hist1rgb, compare_method)
    base_test1 = cv2.compareHist(hist1rgb, hist2rgb, compare_method)

    print(
        "Method:",
        compare_method,
        "Perfect, Base-Half, Base-Test(1), Base-Test(2) :",
        base_base,
        "/",
        base_test1,
    )