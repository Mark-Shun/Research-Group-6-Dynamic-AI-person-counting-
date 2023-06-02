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
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
import sys
np.set_printoptions(threshold=sys.maxsize)

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


def get_average_rgb_per_class(original_img, mask_img, amount_classes):
    print(mask_img.shape, original_img.shape)
    average_rgb = np.zeros((amount_classes, 4))
    for i in range(0, mask_img.shape[0]):
        for j in range(0, mask_img.shape[1]):
                if mask_img[i][j] >= 1:
                    if original_img[i][j][0] == 0 and original_img[i][j][1] == 0 and original_img[i][j][2] == 0:
                        continue
                    
                    average_rgb[mask_img[i][j]][0] += original_img[i][j][0]
                    average_rgb[mask_img[i][j]][1] += original_img[i][j][1]
                    average_rgb[mask_img[i][j]][2] += original_img[i][j][2]
                    average_rgb[mask_img[i][j]][3] +=1

    for i in range(0, amount_classes):
        average_rgb[i][0] = average_rgb[i][0] / average_rgb[i][3]
        average_rgb[i][1] = average_rgb[i][1] / average_rgb[i][3]
        average_rgb[i][2] = average_rgb[i][2] / average_rgb[i][3]
    return average_rgb

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
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            img_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            print(img_name)
            output = model(image.cpu())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            average_rgb = get_average_rgb_per_class(np.asarray(Image.open(f"/home/daan/Huiswerk/Jaar 3/r2d2/research-groep-6/human-parsing/inputs/{img_name}")), np.asarray(parsing_result, dtype=np.uint8), 20)
            # print(np.asarray(average_rgb))
            output_img.save(parsing_result_path)
            if args.logits:
                logits_result_path = os.path.join(args.output_dir, img_name[:-4] + '.npy')
                np.save(logits_result_path, logits_result)
    return


if __name__ == '__main__':
    main()
