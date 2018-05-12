'''
Merge parsing result of cropped poses together to be the label of the whole origin image.

>>> python merge_parsing_result.py --outputDir /path/to/output --parsing_root /root_of_refinement_network/results/${experiment_name}/test_latest/images --origin_img_root /path/to/origin_img --json_file_root /path/to/pose_json_file --aug 0.25
>>>
'''

import numpy as np
import os
import cv2
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--json_file_root", help="path to json file")
parser.add_argument("--origin_img_root", help="path to origin img")
parser.add_argument("--parsing_root", help="path to parsing results")
parser.add_argument("--outputDir", help="where to put output files")
parser.add_argument("--factor", type=int, default=1, help='multiply factor')
parser.add_argument("--aug", type=float, default=0.25, help='augmentation factor for crop')
opt = parser.parse_args()

origin_img_root = opt.origin_img_root
json_file_root = opt.json_file_root
parsing_root = opt.parsing_root

json_file = open(json_file_root, "r")
json_string = json_file.readline()
json_dict = json.loads(json_string)

if not os.path.exists(opt.outputDir):
    os.makedirs(opt.outputDir)

# coco to pascal keypoints order
coco2pascal = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 7]
# the 6th keypoint is missing in coco

num_images = 0
for k, v in json_dict.items():
    num_images += 1
    image_id = k
    origin_img = cv2.imread(os.path.join(origin_img_root, image_id))
    all_prior = np.zeros(origin_img.shape, dtype=np.uint8)
    bodies = v["bodies"]
    for i in range(len(bodies)):
        '''
        The following process of raw_pose and bbox should be the same as in crop_pose_and_generate_testing_prior.py
        '''
        body = bodies[i]
        keypoints = body["joints"]
        raw_pose = np.zeros((1, 32), dtype=float)
        min_x = keypoints[0]
        max_x = min_x
        min_y = keypoints[1]
        max_y = min_y
        for j in range(15):
            x = keypoints[3*j]
            y = keypoints[3*j+1]
            raw_pose[0][2*coco2pascal[j]] = x
            raw_pose[0][2*coco2pascal[j]+1] = y
            if x < min_x:
                min_x = x
            elif x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            elif y > max_y:
                max_y = y
        raw_pose[0][2*6] = (raw_pose[0][2*2] + raw_pose[0][2*3]) / 2
        raw_pose[0][2*6+1] = (raw_pose[0][2*2+1] + raw_pose[0][2*3+1]) / 2
        if max_x > origin_img.shape[1] or max_y > origin_img.shape[0]-1:
            print(max_x, max_y)
            print(image_id + " pose outside img")

        # deal with bbox
        bbox = [min_x, min_y, max_x, max_y]
        xaug = int((max_x - min_x + 1) * opt.aug)
        yaug = int((max_y - min_y + 1) * opt.aug)
        bbox[0] = max(bbox[0] - xaug, 0)
        bbox[1] = max(bbox[1] - yaug, 0)
        bbox[2] = min(bbox[2] + xaug, origin_img.shape[1]-1)
        bbox[3] = min(bbox[3] + yaug, origin_img.shape[0]-1)
        print('bbox', bbox)

        prior = cv2.imread(os.path.join(parsing_root, image_id.split('.')[0] + '_' + str(i) + '_fake_B_postprocessed.png'))
        prior = cv2.resize(prior, (bbox[2]+1-bbox[0], bbox[3]+1-bbox[1]), interpolation=cv2.INTER_NEAREST)
        all_prior[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = np.maximum(all_prior[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1], prior)

        print(image_id, i, num_images)
    # all_prior = all_prior + (all_prior == 0) * 255
    cv2.imwrite(os.path.join(opt.outputDir, image_id[:len(image_id)-3]+'png'), all_prior[:, :, 0])

print('finished')
