import cv2

# Load the two images
image1 = cv2.imread('coen.jpg')

image2 = cv2.imread('0-seg7.jpg', cv2.IMREAD_GRAYSCALE)

# Create FAST detector and BRISK descriptor
detector = cv2.FastFeatureDetector_create()
descriptor = cv2.BRISK_create()

# Detect keypoints and compute descriptors for both images
keypoints1 = detector.detect(image1, None)
keypoints1, descriptors1 = descriptor.compute(image1, keypoints1)

keypoints2 = detector.detect(image2, None)
keypoints2, descriptors2 = descriptor.compute(image2, keypoints2)

# Match keypoints using brute-force matcher
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

# Apply distance threshold to filter out matches
distance_threshold = 50
filtered_matches = [match for match in matches if match.distance < distance_threshold]

# Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

match_percentage = (len(filtered_matches) / len(keypoints1)) * 100
print(match_percentage)

# Display the matched image

cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    # Read frames from the camera
    ret, frame = cap.read()

    keypoints2 = detector.detect(frame, None)
    keypoints2, descriptors2 = descriptor.compute(frame, keypoints2)

    # Match keypoints using brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Apply distance threshold to filter out matches
    distance_threshold = 50
    filtered_matches = [match for match in matches if match.distance < distance_threshold]

    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, frame, keypoints2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_percentage = (len(filtered_matches) / len(keypoints1)) * 100
    print(match_percentage)


    # If frame reading is successful
    if ret:
        # Display the frame in a window called "Camera Feed"
        cv2.imshow("Camera Feed", matched_image)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()


cv2.imshow('Matches', matched_image)

