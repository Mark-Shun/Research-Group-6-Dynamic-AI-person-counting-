# This python script stitches seperate images together with the help or feature extraction, to create a new panoramic image.
import sys
import cv2
import numpy as np
from collections import namedtuple

# Set minimum match condition
MIN_MATCH_COUNT = 10

images = ["./images/image1.jpg", "./images/image2.jpg"]

def generate_panorama(input_images, features_amount):
    image_objects = []
    ImageData = namedtuple('ImageData', ['name','image', 'gray_image', 'keypoints', 'descriptors'])
    orb = cv2.ORB_create(nfeatures=features_amount)
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    for file in input_images:
        image = cv2.imread(file)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        image_data = ImageData(file,image,gray_image,keypoints,descriptors)
        image_objects.append(image_data)

    descriptors = [object.descriptors for object in image_objects]
    matches = bf.knnMatch(descriptors[0], descriptors[1], k=2)

    # for object in image_objects:
    #     cv2.namedWindow(object.name, cv2.WINDOW_NORMAL)
    #     cv2.imshow(object.name,cv2.drawKeypoints(object.image, object.keypoints, None, (255,0,255)))
    # cv2.waitKey(0)
    all_matches = []
    good_matches = []
    for m, n in matches:
       all_matches.append(m)
       if m.distance < 0.6 * n.distance:
          good_matches.append(m)
    good_amount = len(good_matches)
    if good_amount > MIN_MATCH_COUNT:
       # Convert keypoints to an argument for findHomography
       source_points = np.float32([ image_objects[0].keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
       destination_points = np.float32([ image_objects[1].keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

       # Establish an homography
       M, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)

       stitched_image = warpImages(image_objects[1].image, image_objects[0].image, M)
       cv2.namedWindow("Stitched image", cv2.WINDOW_NORMAL)
       cv2.imshow("Stitched image", stitched_image)
       cv2.waitKey(0)
    else:
        print(f"Not enough good matches found: [{good_amount}/{MIN_MATCH_COUNT}]")     
        matches_image = draw_matches(image_objects[0].gray_image, image_objects[0].keypoints, image_objects[1].gray_image, image_objects[1].keypoints, good_matches)
        cv2.namedWindow("Found matches", cv2.WINDOW_NORMAL)
        cv2.imshow("Found matches", matches_image)
        cv2.waitKey(0)

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
  r, c = image1.shape[:2]
  r1, c1 = image2.shape[:2]

  # Create a blank image with the size of the first image + second image
  output_image = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
  output_image[:r, :c, :] = np.dstack([image1, image1, image1])
  output_image[:r1, c:c+c1, :] = np.dstack([image2, image2, image2])

  # Go over all of the matching points and extract them
  for match in matches:
    image1_idx = match.queryIdx
    image2_idx = match.trainIdx
    (x1, y1) = keypoints1[image1_idx].pt
    (x2, y2) = keypoints2[image2_idx].pt

    # Draw circles on the keypoints
    cv2.circle(output_image, (int(x1),int(y1)), 4, (0, 255, 255), 1)
    cv2.circle(output_image, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)

    # Connect the same keypoints
    cv2.line(output_image, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
    
  return output_image

def warpImages(image1, image2, H):

  rows1, cols1 = image1.shape[:2]
  rows2, cols2 = image2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_image = cv2.warpPerspective(image2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_image[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = image1

  return output_image

generate_panorama(images,2000)