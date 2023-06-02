import cv2
import numpy as np
import matplotlib.pyplot as plt


percentageFAST = []
percentageSIFT = []
percentageORB = []
percentageBRIEF = []

def computeNumbers(arr, name=''):
    sum = np.array(arr).sum()
    average = sum / len(arr)
    length = len(arr)
    #print('\n',name, 'number of measurements:', length, '\nSum: ', sum, '\nAverage: ', average, '\n')

    string = f"{name} number of measurements: {length} Sum: {sum} Average: {average} "
    
    return string, [name, length, sum, average]

    

def FAST(image1, image2, show=True, numberOfMeasurements=None):
    text = ''
    data = None


    # Create FAST detector and BRISK descriptor
    detector = cv2.FastFeatureDetector_create()
    #detector = cv2.AgastFeatureDetector_create()

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

    # Calculate match percentage
    if not len(keypoints1):
        match_percentage = 0
    
    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, filtered_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image and match percentage
    if show:
        cv2.imshow('FAST', matched_image)
        
    print("Match Percentage:", match_percentage)
    percentageFAST.append(match_percentage)

    if numberOfMeasurements < len(percentageFAST):
        text, data = computeNumbers(percentageFAST, 'FAST')

    return matched_image, text, data


def SIFT(image1, image2, show=True, numberOfMeasurements=None):
    text = ''
    data = None


    # Create a SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for the images
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw the top 10 matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Apply distance threshold to filter out matches
    distance_threshold = 50
    filtered_matches = [match for match in matches if match.distance < distance_threshold]

    # Calculate match percentage
    if not len(keypoints1):
        match_percentage = 0
    
    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    if show:
        cv2.imshow('SIFT', matched_image)

    print("Match Percentage:", match_percentage)
    percentageSIFT.append(match_percentage)

    if numberOfMeasurements < len(percentageSIFT):
        text, data = computeNumbers(percentageSIFT, 'SIFT')

    return matched_image, text, data


def ORB(image1, image2, show=True, numberOfMeasurements=None):
    text = ''
    data = None

    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Create a Brute-Force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors of both images
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Apply distance threshold to filter out matches
    distance_threshold = 50
    filtered_matches = [match for match in matches if match.distance < distance_threshold]

    # Calculate match percentage
    if not len(keypoints1):
        match_percentage = 0
    
    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    if show:
        cv2.imshow('ORB', matched_image)

    print("Match Percentage:", match_percentage)
    percentageORB.append(match_percentage)

    if numberOfMeasurements < len(percentageORB):
        text, data = computeNumbers(percentageORB, 'ORB')

    return matched_image, text, data


def BRIEF(image1, image2, show=True, numberOfMeasurements=None):
    text = ''


    # Initialize BRIEF
    # requires opencv-contrib-python
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = brief.detectAndCompute(image1, None)
    keypoints2, descriptors2 = brief.detectAndCompute(image2, None)

    # Create a Brute-Force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors of both images
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    distance_threshold = 50
    filtered_matches = [match for match in matches if match.distance < distance_threshold]

    # Calculate match percentage
    if not len(keypoints1):
        match_percentage = 0
    
    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    if show:
        cv2.imshow('BRIEF', matched_image)

    print("Match Percentage:", match_percentage)
    percentageSIFT.append(match_percentage)

    if numberOfMeasurements < len(percentageBRIEF):
        text = computeNumbers(percentageBRIEF, 'BRIEF')

    return matched_image, text


def showHistogram(data1=None, data2=None, data3=None):
    fig, ax = plt.subplots()

    name = []
    height = []
    colors = ['red', 'blue', 'green']

    if data1:
        name.append(data1[0])
        height.append(data1[3])

    if data2:
        name.append(data2[0])
        height.append(data2[3])
    
    if data3:
        name.append(data3[0])
        height.append(data3[3])

    ax.bar(name, height, color=colors)

    ax.set_xlabel('Algoritme')
    ax.set_ylabel('Average %')
    ax.set_title('Gemiddelde accuracy')
    plt.show(block=False)
    plt.close()
    return

if __name__ == '__main__':
    directory = 'runs\\track\\exp124\\crops-seg\\videoplayback\\person'

    import os
    from PIL import Image

    for map in os.listdir(directory):
        firstPicture = None
        for picture in os.listdir(directory + '/' + map):
            path = os.path.join(directory, map, picture)
            if firstPicture is None:
                firstPicture = np.array(Image.open(path))
                continue

            print(path)
            image = Image.open(path)
            imageArr = np.array(image)

            imageFAST, infoFAST, dataFAST = FAST(firstPicture, imageArr, numberOfMeasurements=10, show=True)
