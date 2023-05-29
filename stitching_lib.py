import stitching
import cv2

settings = {"detector": "sift", "confidence_threshold": 0.2, "nfeatures" : 500}
stitcher = stitching.Stitcher(**settings)

panorama = stitcher.stitch(["./images/image1.jpg", "./images/image2.jpg", "./images/image3.jpg", "./images/image4.jpg", "./images/image5.jpg"])

cv2.imwrite("./images/stitched_image.jpg", panorama)
cv2.waitKey(0)