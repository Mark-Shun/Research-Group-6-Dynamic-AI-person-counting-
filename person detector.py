import os
from PIL import Image
import numpy as np

def deleteBadPictures(directory):
    list = []
    for map in os.listdir(directory):
        for picture in os.listdir(directory + '/' + map):
            path = os.path.join(directory, map, picture)
            image = Image.open(path)
            imageArr = np.array(image)

            numPixels = imageArr.shape[0] * imageArr.shape[1]
            blackPixels = np.count_nonzero(imageArr == np.array([0,0,0]))
            percentage = int(blackPixels / numPixels / 3 * 100)

            if percentage > 50 or imageArr.shape[0] < 100 or imageArr.shape[1] < 100:
                os.remove(os.path.join(directory, map, picture))

            print(os.path.join(directory,map,picture), percentage)
    return list
    
def main():
    deleteBadPictures('person')


if __name__ == '__main__':
    main()
    