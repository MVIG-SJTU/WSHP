'''
Given a json file containing poses for images(one image may have more than one pose corresponding to different people),
crop out each people and generate corresponding prior.
Then we can use the test mode of the pre-trained refinement model to generate parsing result for each cropped pose.

>>> python crop_pose_and_generate_testing_prior.py --PASCALPoseFileRoot /path/to/pascal_pose_file.csv --PASCALMaskImgDir /path/to/pascal_mask_img --n 3 --k 3 --aug 0.25 --origin_img_root /path/to/origin_img --json_file_root /path/to/pose_json_file --outputDir /path/to/output
>>>
'''

import argparse
import json
from generate_prior_util import *

parser = argparse.ArgumentParser()
parser.add_argument("--PASCALPoseFileRoot", help="path to PASCAL pose file")
parser.add_argument("--PASCALMaskImgDir", help="path to PASCAL mask images")
parser.add_argument("--origin_img_root", help="path to origin img")
parser.add_argument("--json_file_root", help="path to json file")
parser.add_argument("--outputDir", help="where to put output files")
parser.add_argument("--draw_skeleton", action="store_true", help="draw skeleton to check the format of keypoints")
parser.add_argument("--n", type=int, default=5, help="number of close images picked first time")
parser.add_argument("--k", type=int, default=3, help="number of close images picked for prior generation in n picked images")
parser.add_argument("--aug", type=float, default=0.25, help='augmentation factor for crop')
opt = parser.parse_args()

json_file_root = opt.json_file_root
origin_img_root = opt.origin_img_root
json_file = open(json_file_root, "r")
json_string = json_file.readline()
json_dict = json.loads(json_string)
print('length of json_dict', len(json_dict))

pascal_poses, pascal_img_names, pascal_pose_dict = load_pascal_pose(opt.PASCALPoseFileRoot)
print('length of pascal_img', len(pascal_img_names))

if not os.path.exists(opt.outputDir):
    os.makedirs(opt.outputDir)
img_dir = os.path.join(opt.outputDir, 'img')
prior_dir = os.path.join(opt.outputDir, 'prior')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(prior_dir):
    os.makedirs(prior_dir)

if opt.draw_skeleton:
    skeleton_dir = os.path.join(opt.outputDir, 'skeleton')
    if not os.path.exists(skeleton_dir):
        os.makedirs(skeleton_dir)

# alphapose to pascal keypoints order
alphapose2pascal = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 7]
# the 6th keypoint is missing

num_images = 0
for k, v in json_dict.items():
    num_images += 1
    image_id = k
    origin_img = cv2.imread(os.path.join(origin_img_root, image_id))
    bodies = v["bodies"]
    for i in range(len(bodies)):
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
            raw_pose[0][2*alphapose2pascal[j]] = x
            raw_pose[0][2*alphapose2pascal[j]+1] = y
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

        prior = generate_prior_single_person(bbox, raw_pose, opt.PASCALMaskImgDir, pascal_poses, pascal_img_names, pascal_pose_dict, opt.n, opt.k)
        prior = prior[:, :, [2, 1, 0]]
        img = origin_img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
        if opt.draw_skeleton:
            skeleton_img = drawSkeleton(origin_img, raw_pose)
            cv2.imwrite(os.path.join(skeleton_dir, image_id.split('.')[0]+'_'+str(i)+'.jpg'), skeleton_img)
        cv2.imwrite(os.path.join(img_dir, image_id.split('.')[0]+'_'+str(i)+'.jpg'), img)
        cv2.imwrite(os.path.join(prior_dir, image_id.split('.')[0]+'_'+str(i)+'.jpg'), prior)

        print(image_id, i, num_images)

print('finished')
