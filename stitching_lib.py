import os
import stitching
import cv2
import time

settings = {"detector": "sift", "confidence_threshold": 0.2, "nfeatures" : 1000, "try_use_gpu" : True}
stitcher = stitching.Stitcher(**settings)
input_images = []
directory = 'images/backyard2/'

for file in os.listdir(directory):
    if(file != "stitched_image.jpg"):
        f = os.path.join(directory, file)
        if os.path.isfile(f):
            input_images.append(f)

start_time = time.time()
panorama = stitcher.stitch(input_images)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Finished stitching directory: {directory}.\n Processed {len(input_images)} images, which took: {elapsed_time} seconds.")
cv2.imwrite(f"{directory}stitched_image.jpg", panorama)
cv2.waitKey(0)