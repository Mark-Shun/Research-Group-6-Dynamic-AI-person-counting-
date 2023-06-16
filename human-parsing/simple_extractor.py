#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

"""
Run with
python3 simple_extractor.py --dataset 'lip' --model-restore 'checkpoints/final.pth' --input-dir 'inputs' --output-dir 'outputs'

add > out.txt to save output to file
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from tqdm import tqdm
import shutil
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
import sys
np.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True) # Make sure numpy prints the whole array and without scientific notation
debug = False
save_original = True

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}
minumim_parts_index = [2,4,6,11,14]
minumim_parts_label = [dataset_settings['atr']['label'][2], dataset_settings['atr']['label'][4], dataset_settings['atr']['label'][6], dataset_settings['atr']['label'][11], dataset_settings['atr']['label'][14]]
minimum_pixels = 500

import time

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print( 'func:%r took: %2.4f sec' % \
            (f.__name__, te-ts))
        return result

    return timed
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_average_rgb_per_class(original_img, mask_img, amount_classes, label, img_name):
    """ Returns the average rgb per class
    Args: 
        The original image, the mask image, the amount of classes and the label
    Returns: 
        The average rgb per class in a 2d array with the following format: [r,g,b,amount_of_pixels] for each class
    """
    average_rgb = np.zeros((amount_classes, 5))
    average_rgb[0][4] = img_name[0]
    for i in range(0, mask_img.shape[0]):
        for j in range(0, mask_img.shape[1]):
                if mask_img[i][j] >= 1: # 0 is background so we skip it
                    if original_img[i][j][0] == 0 and original_img[i][j][1] == 0 and original_img[i][j][2] == 0:
                        continue
                    average_rgb[mask_img[i][j]][0] += original_img[i][j][0]
                    average_rgb[mask_img[i][j]][1] += original_img[i][j][1]
                    average_rgb[mask_img[i][j]][2] += original_img[i][j][2]
                    average_rgb[mask_img[i][j]][3] +=1
    for i in range(0, amount_classes): # calculate average
        average_rgb[i][0] = average_rgb[i][0] / average_rgb[i][3]
        average_rgb[i][1] = average_rgb[i][1] / average_rgb[i][3]
        average_rgb[i][2] = average_rgb[i][2] / average_rgb[i][3]
    if debug == True:
        for row_index, row in enumerate(average_rgb):
            if row[3] == 0:
                continue
            print(row,label[row_index])
        print("\n")
    return average_rgb


""" Returns the difference between the average rgb of the image and the average rgb of the another image for each class
Args: 
    The average rgb of the image, the average rgb of the other image
Returns:
    The difference between the average rgb of the image and the average rgb of the other image for each class
"""
def compare_rgb(rgb_table, rgb_to_compare):
    result = np.zeros((rgb_table.shape[0],3 ))
    for i in range(0, rgb_table.shape[0]):
        for j in range(0, 3):
            result[i][j] = abs((rgb_table[i][j]-rgb_to_compare[i][j])/rgb_to_compare[i][j])
    return result

""" Calculates the total rgb of the image of the minimum parts
Args: 
    The table with the average rgb per class
Returns:
    The total rgb of the image of the minimum parts

"""
def calculate_total(table):
    total = 0
    for row_index, row in enumerate(table):
        if row_index in minumim_parts_index:
            total += row[0]
            total += row[1]
            total += row[2]
    return total

""" Prints the table with the average rgb per class
Args:
    The table with the average rgb per class, the labels of the dataset
"""

def  print_table(table, label):
    total = 0
    for row_index, row in enumerate(table):
        if row_index in minumim_parts_index:
            total += row[0]
            total += row[1]
            total += row[2]
        if np.isnan(row[0]) == True: # If the average rgb is nan we skip it since there are no pixels of that class
            continue
        print(row,label[row_index])
    print(total)
    print("\n")

""" Compares the average rgb of the image with the average rgb of the other images
Args:
    The table with the average rgb per class of the image to compare with
"""
def compare_persons(rgb_table):
    total_table = np.zeros((len(os.listdir("persons")), 50))
    for foldernames in os.listdir("persons"):
        for filenames in os.listdir("persons/"+foldernames):
            average_rgb = np.load(f"persons/{foldernames}/{filenames}")
            total_table[int(foldernames)-1][int(filenames[0])-1] = calculate_total(compare_rgb(rgb_table, average_rgb))
    return total_table

def add_person(rgb_table, minimum_accuracy, filename):
    total_table = compare_persons(rgb_table)
    table = (total_table>0) == (total_table<=minimum_accuracy)
    for i in range(0, len(table)):
        for j in range(0, len(table[i])):
            if table[i][j] == True:
                if os.path.exists(f"persons/{i+1}") == False:
                    os.mkdir(f"persons/{i+1}")
                np.save(f"persons/{i+1}/{filename[:-4]}-{(len(os.listdir(f'persons/{i+1}/'))+1)}.npy", rgb_table)
                if save_original == True:
                    if os.path.exists(f"persons-original/{i+1}") == False:
                        os.mkdir(f"persons-original/{i+1}")
                    os.rename(f"inputs/{filename}", f"persons-original/{i+1}/{len(os.listdir(f'persons/{i+1}/'))+1}-{filename}-{total_table[i][j]}.jpg")
                    return
                else:
                    os.remove(f"inputs/{filename}")
                    return
    
    os.mkdir(f"persons/{len(os.listdir('persons/'))+1}")
    np.save(f"persons/{len(os.listdir('persons/'))}/{filename[:-4]}-1.npy", rgb_table)
    if save_original == True:
        if os.path.exists(f"persons-original/{len(os.listdir('persons-original/'))+1}") == False:
                os.mkdir(f"persons-original/{len(os.listdir('persons-original/'))+1}")
        os.rename(f"inputs/{filename}", f"persons-original/{len(os.listdir('persons/'))}/1-{filename}")
    else:
        os.remove(f"inputs/{filename}")

def check_min(img_name, average_rgb, parsing_result):
    isin = np.isin(minumim_parts_index, parsing_result)
    if np.all(isin) == False:
            os.rename(f"inputs/{img_name}", f"wrong-data/{img_name[:-4]}-{isin}-error.jpg")
            return False # Skip images that don't have all the minimum parts
    return True


def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cpu()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    palette = get_palette(num_classes)
    i = 0
    numpy_array = []
    with torch.no_grad():
        #Make list of persons
        persons = []

        for idx, batch in enumerate(tqdm(dataloader)):
            
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            output = model(image.cpu())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)

            #Check for required body parts
            if not np.isin(minumim_parts_index,parsing_result).all():
                print("Image is not valid: " + img_name)
                quit()

            #Person dict name and body parts
            person = {"name": img_name, "id":img_name[0], "parts":[]}

            #Get average color of all body parts
            for body_part in minumim_parts_index:
                #Load image
                original_img = np.asarray(Image.open(f"inputs/{img_name}"))
                #Find body part indexes and get original image values of body part
                body_part_pixels = original_img[np.where(parsing_result == body_part)]
                #Get Color average value of body part
                mean_body_part_color = body_part_pixels.mean(axis=0)
                #Save average color
                person["parts"] += [mean_body_part_color]
            #Save parts to person
            person["parts"] = np.array(person["parts"])
            persons += [person]
  
    results = []
    # Calc matching percent
    for person1 in persons:
        # Match all imgs with all imgs
        for person2 in persons:
            # Check for same img
            if person1["name"] == person2["name"]:
                continue
            # Is the same person
            same_person = person1["id"] == person2["id"]
            #Calculate perc with Euclidean Distance
            perc = np.linalg.norm(person1["parts"].flatten() - person2["parts"].flatten())/255
            #Save result
            results += [[same_person, perc]]
    #Save Pickle        
    with open("results.pkl", "wb") as result_file:
        pickle.dump(results, result_file)

    

            # average_rgb = np.expand_dims(get_average_rgb_per_class(np.asarray(Image.open(f"inputs/{img_name}")), np.asarray(parsing_result, dtype=np.uint8), 18, label, img_name),axis=0)
            # if idx == 0:
            #     numpy_array = average_rgb
            # if idx == 6 or idx == 64:
            #     print(img_name)

            # else:
            #     numpy_array = np.append(numpy_array, average_rgb, axis=0)
            # if idx % 100 == 0:
            #     np.save(f"average-rgb/{idx}-checkpoint.npy", average_rgb)
            # # continue
    
            # if check_min(img_name, average_rgb, parsing_result) == False:
            #     print(img_name)
            
            # parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            # output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            # output_img.putpalette(palette)
            # output_img.save(parsing_result_path)
           
            
            # i+=1
            # if args.logits:
            #     logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
            #     np.save(logits_result_path, logits_result)
            # # add_person(average_rgb, 7.0, img_name)

    # np.save(f"average-rgb/final.npy", numpy_array)
    

if __name__ == '__main__':
    main()


