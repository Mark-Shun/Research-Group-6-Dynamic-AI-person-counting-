#TODO Create an usable function for track.py with arguments for time print and writing/saving panorama image.
import argparse
from pathlib import Path
import os
import stitching
import cv2
import time

settings = {"detector": "sift", "confidence_threshold": 0.2, "nfeatures" : 1000, "try_use_gpu" : True}
stitcher = stitching.Stitcher(**settings)
input_images = []
def stitch_directory(directory_path, save_image=False, print_info = True):
    directory = directory_path

    for file in os.listdir(directory):
        if(file != "stitched_image.jpg"):
            f = os.path.join(directory, file)
            if os.path.isfile(f):
                input_images.append(f)

    print(f"Start stitching process in: {directory}")
    start_time = time.time()
    panorama = stitcher.stitch(input_images)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if(print_info == True):
        print(f"Finished stitching directory: {directory}.\n Processed {len(input_images)} images, which took: {elapsed_time} seconds.")
    if(save_image == True):
        cv2.imwrite(f"{directory}/stitched_image.jpg", panorama)
    return panorama

def main(directory_path_arg, save_image_arg, print_info_arg):
    directory_path = Path(directory_path_arg).resolve()
    stitch_directory(directory_path, save_image_arg, print_info_arg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to stitch images in a folder together and create a panorama picture.")
    parser.add_argument("directory_path", type=str, help="Provides the path to the directory with images inside to be stitched together.")
    parser.add_argument("--save_image", type=bool, default=True, help="Determines whether or not to save the resulting image in the directory.")
    parser.add_argument("--print_info", type=bool, default=True, help="Determines whether or not to print duration and amount information.")

    args = parser.parse_args()
    main(args.directory_path, args.save_image, args.print_info)