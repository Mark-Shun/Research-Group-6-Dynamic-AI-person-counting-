import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

percentageFAST = []
percentageSIFT = []
percentageORB = []
percentageBRIEF = []

person = []

distance_threshold = 40
lengthMatches = -1


def computeNumbers(arr, name=''):
    sum = np.array(arr).sum()
    average = sum / len(arr)
    length = len(arr)
    # print('\n',name, 'number of measurements:', length, '\nSum: ', sum, '\nAverage: ', average, '\n')

    string = f"{name} number of measurements: {length} Sum: {sum} Average: {average} "

    return string, [name, length, sum, average]


def FAST(image1, image2, show=True, numberOfMeasurements=0, printStats=False):
    text = ''
    data = None

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create FAST detector and BRISK descriptor
    detector = cv2.FastFeatureDetector_create()
    # detector = cv2.AgastFeatureDetector_create()

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
    filtered_matches = [
        match for match in matches[:lengthMatches] if match.distance < distance_threshold]

    if printStats:
        print('\nFAST')
        print('filtered_matches: ', len(filtered_matches))
        print('matches: ', len(matches))
        print('keypoints1: ', len(keypoints1))
        print('keypoints2: ', len(keypoints2))
        print('\n')

    # Calculate match percentage
    if not len(keypoints1):
        match_percentage = 0

    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    # Draw matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image and match percentage
    if show:
        cv2.imshow('FAST', matched_image)

    # print('FAST: ', len(keypoints1), len(keypoints2), len(filtered_matches))
    # print("Match Percentage:", match_percentage)
    percentageFAST.append(match_percentage)

    if numberOfMeasurements < len(percentageFAST):
        text, data = computeNumbers(percentageFAST, 'FAST')

    return matched_image, text, data


def SIFT(image1, image2, show=True, numberOfMeasurements=None, printStats=False):
    text = ''
    data = None

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create a SIFT object
    sift = cv2.SIFT_create(nfeatures=2000, nOctaveLayers=8, edgeThreshold=0)

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
    filtered_matches = [match for match in matches[:lengthMatches]
                        if match.distance < distance_threshold]

    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Apply distance threshold to filter out matches

    if printStats:
        print('\nSIFT')
        print('filtered_matches: ', len(filtered_matches))
        print('matches: ', len(matches))
        print('keypoints1: ', len(keypoints1))
        print('keypoints2: ', len(keypoints2))
        print('\n')

    # Calculate match percentage
    # print('SIFT: ', len(keypoints1), len(keypoints2), len(filtered_matches))

    if not len(keypoints1):
        match_percentage = 0

    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    if show:
        cv2.imshow('SIFT', matched_image)

    # print("Match Percentage:", match_percentage)
    percentageSIFT.append(match_percentage)

    if numberOfMeasurements < len(percentageSIFT):
        text, data = computeNumbers(percentageSIFT, 'SIFT')

    return matched_image, text, data


def ORB(image1, image2, show=True, numberOfMeasurements=None, printStats=False):
    text = ''
    data = None

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000, edgeThreshold=10)

    # Detect and compute keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top 10 matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Apply distance threshold to filter out matches
    filtered_matches = [
        match for match in matches[:lengthMatches] if match.distance < distance_threshold]

    if printStats:
        print('\nORB')
        print('filtered_matches: ', len(filtered_matches))
        print('matches: ', len(matches))
        print('keypoints1: ', len(keypoints1))
        print('keypoints2: ', len(keypoints2))
        print('\n')

    # Calculate match percentage

    if not len(keypoints1):
        match_percentage = 0

    else:
        match_percentage = (len(filtered_matches) / len(keypoints1)) * 100

    # print('ORB: ', len(keypoints1), len(keypoints2), len(filtered_matches))

    if show:
        cv2.imshow('ORB', matched_image)

    # print("Match Percentage:", match_percentage)
    percentageORB.append(match_percentage)

    if numberOfMeasurements < len(percentageORB):
        text, data = computeNumbers(percentageORB, 'ORB')

    return matched_image, text, data


def BRIEF(image1, image2, show=True, numberOfMeasurements=None, printStats=False):
    text = ''

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

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
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                                    matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    distance_threshold = 50
    filtered_matches = [
        match for match in matches if match.distance < distance_threshold]

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


def testAlgorithms(directory: str):
    global percentageFAST, percentageSIFT, percentageORB, person
    data = []
    for map in os.listdir(directory):
        # percentageFAST.clear()
        # percentageSIFT.clear()
        # percentageORB.clear()
        print('folder: ', map, '\n\n')
        for fileName1 in os.listdir(directory + '/' + map):
            for fileName2 in os.listdir(directory + '/' + map):
                if fileName1[:3] == fileName2[:3]:
                    continue
                print(fileName1, fileName2)
                if map == '1':
                    person.append(False)
                else:
                    person.append(True)
                image1 = np.array(Image.open(
                    directory + '/' + map + '/' + fileName1))
                image2 = np.array(Image.open(
                    directory + '/' + map + '/' + fileName2))

                imageFAST, infoFAST, dataFAST = FAST(
                    image1, image2, numberOfMeasurements=1, show=False)
                imageSIFT, infoSIFT, dataSIFT = SIFT(
                    image1, image2, numberOfMeasurements=1, show=False)
                imageORB, infoORB, dataORB = ORB(
                    image1, image2, numberOfMeasurements=1, show=False)

        print('\n\n', infoFAST)
        print(infoSIFT)
        print(infoORB, '\n\n')
        data.append("folder: ")
        data.append(map)
        data.append(infoFAST)
        data.append(infoSIFT)
        data.append(infoORB)
        data.append('\n\n')

    for str in data:
        print(str)


def save_image_with_colored_keypoints(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a FAST detector object
    fast = cv2.FastFeatureDetector_create()

    # Detect keypoints using FAST
    keypoints = fast.detect(gray, None)

    # Generate random colors for keypoints
    colors = np.random.randint(0, 255, (len(keypoints), 3), dtype=np.uint8)

    # Draw keypoints on the image with different colors
    image_with_keypoints = np.copy(image)
    for i, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        cv2.circle(image_with_keypoints, (int(x), int(y)),
                   6, tuple(colors[i].tolist()), 2)

    # Save the image with keypoints
    cv2.imwrite(output_path, image_with_keypoints)

    print(f"Image with colored keypoints saved at: {output_path}")


def test():
    image1 = cv2.imread('nick.png')
    image2 = cv2.imread('mark.png')

    imageFAST, infoFAST, dataFAST = FAST(
        image1, image2, numberOfMeasurements=1, show=False)
    imageSIFT, infoSIFT, dataSIFT = SIFT(
        image1, image2,  numberOfMeasurements=1, show=False)
    imageORB, infoORB, dataORB = ORB(
        image1, image2, numberOfMeasurements=1, show=False)

    imageFASTResized = cv2.resize(imageFAST, (640, 240))
    imageSIFTResized = cv2.resize(imageSIFT, (640, 240))
    imageORBResized = cv2.resize(imageORB, (640, 240))

    combined_image = np.vstack(
        (imageFASTResized, imageSIFTResized, imageORBResized))
    cv2.imshow('foto', combined_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def computeThreshold():
    with open("checkpoint.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)

    percentageArrTrue = []
    percentageArrFalse = []
    for algorithm in range(3):
        for threshold in range(0, 100):
            percentageTrue = 0
            percentageFalse = 0
            for index in range(len(cp[0])):
                if (cp[algorithm][index] > threshold) and cp[3][index]:
                    percentageTrue += 1
                if (cp[algorithm][index] < threshold) and cp[3][index]:
                    percentageTrue -= 1
                if (cp[algorithm][index] < threshold) and not cp[3][index]:
                    percentageTrue += 1
                if (cp[algorithm][index] > threshold) and not cp[3][index]:
                    percentageTrue -= 1

            percentageArrTrue.append(percentageTrue)
            percentageArrFalse.append(percentageFalse)

        print(algorithm, '\n\n', percentageArrTrue, '\n\n')
        percentageArrTrue.clear()


def naarNick():
    with open("checkpoint.pkl", "rb") as cp_file:
        cp = pickle.load(cp_file)

    arrFAST = []
    arrSIFT = []
    arrORB = []

    for i in range(len(cp[3])):
        arrFAST.append([cp[3][i], cp[0][i]])
    for i in range(len(cp[3])):
        arrSIFT.append([cp[3][i], cp[1][i]])
    for i in range(len(cp[3])):
        arrORB.append([cp[3][i], cp[2][i]])

    with open("FAST.pkl", "wb") as cp_file:
        pickle.dump(arrFAST, cp_file)
    with open("SIFT.pkl", "wb") as cp_file:
        pickle.dump(arrSIFT, cp_file)
    with open("ORB.pkl", "wb") as cp_file:
        pickle.dump(arrORB, cp_file)


def testAlgorithms2():
    global percentageFAST, percentageSIFT, percentageORB, person
    with open("imgs.pkl", "rb") as cp_file:
        imgs = pickle.load(cp_file)
    for compare in imgs:
        image1 = np.array(Image.open(compare[0]))
        image2 = np.array(Image.open(compare[1]))

        print(compare)
        if compare[0][5] == compare[1][5]:
            person.append(True)
        else:
            person.append(False)

        imageFAST, infoFAST, dataFAST = FAST( image1, image2, numberOfMeasurements=1, show=False)
        imageSIFT, infoSIFT, dataSIFT = SIFT( image1, image2, numberOfMeasurements=1, show=False)
        imageORB, infoORB, dataORB = ORB( image1, image2, numberOfMeasurements=1, show=False)

        


if __name__ == '__main__':
    # testAlgorithms('dataset')
    # with open("checkpoint.pkl", "wb") as cp_file:
    #        pickle.dump([percentageFAST, percentageSIFT, percentageORB, person], cp_file)
    # naarNick()
    # computeThreshold()
    testAlgorithms2()
    naarNick()
    
