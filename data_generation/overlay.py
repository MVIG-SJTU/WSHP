'''
Overlay origin image and painted parsing label.

>>> python overlay.py --origin_img_root /path/to/origin_img --parsing_img_root /path/to/parsing_img --outputDir /path/to/output
>>>
'''

import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--origin_img_root", help="path to origin img")
parser.add_argument("--parsing_img_root", help="path to parsing img")
parser.add_argument("--outputDir", help="where to put output files")
parser.add_argument("--factor", type=int, default=1, help='multiply factor')
parser.add_argument("--aug", type=float, default=0, help='augmentation factor for crop')
a = parser.parse_args()

origin_img_root = a.origin_img_root
parsing_img_root= a.parsing_img_root
output_path = a.outputDir

body_part_color = np.array([
    [0, 0, 0], 
    [128, 0, 0], 
    [0, 128, 0], 
    [128, 128, 0], 
    [0, 0, 128], 
    [128, 0, 128], 
    [0, 128, 128], 
    [128, 128, 128], 
    [64, 0, 0], 
    [192, 0, 0], 
    [64, 128, 0]], 
    dtype=np.uint8)

def paint(mask_img):
    assert(len(mask_img.shape) == 2)
    return body_part_color[mask_img]

def overlay(origin_img, parsing_img):
    overlay_img = origin_img*0.7 + parsing_img[:,:,[2,1,0]]*0.9
    overlay_img = (overlay_img > 255) * 255 + overlay_img * (overlay_img <= 255)
    return overlay_img

if not os.path.exists(output_path):
    os.makedirs(output_path)

for root, dirs, files in os.walk(parsing_img_root):
    for file in files:
        origin_img = cv2.imread(os.path.join(origin_img_root, file[0:len(file)-4]+'.jpg'))
        parsing_img = cv2.imread(os.path.join(root, file), 0)
        overlay_img = overlay(origin_img, paint(parsing_img))

        cv2.imwrite(os.path.join(output_path, file), overlay_img)
        print(file)
