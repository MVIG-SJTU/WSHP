'''
We use Pascal dataset, which has both keypoints and segmentation annotations, to generate prior for other dataset which has only keypoints information.
In order to train our refinement network, we need to generate prior for each pascal image, this is what this code for.

>>> python generate_pascal_training_prior.py --PASCALPoseFileRoot /path/to/pascal_pose_file.csv --PASCALMaskImgDir /path/to/pascal_mask_img --outputDir /path/to/output --n 5 --k 3
>>>
'''

import argparse
from generate_prior_util import *


parser = argparse.ArgumentParser()
parser.add_argument("--PASCALPoseFileRoot", help="path to PASCAL pose file")
parser.add_argument("--PASCALMaskImgDir", help="path to PASCAL mask images")
parser.add_argument("--outputDir", help="where to put output files")
parser.add_argument("--n", type=int, default=5, help="number of close images picked first time")
parser.add_argument("--k", type=int, default=3, help="number of close images picked for prior generation in n picked images")
opt = parser.parse_args()

# load PASCAL pose
pascal_poses, pascal_img_names, pascal_pose_dict = load_pascal_pose(opt.PASCALPoseFileRoot)

if not os.path.exists(opt.outputDir):
    os.makedirs(opt.outputDir)

for i in range(len(pascal_img_names)):
    pascal_name = pascal_img_names[i]
    print('processing', pascal_name)
    pascal_mask_img = cv2.imread(os.path.join(opt.PASCALMaskImgDir, pascal_name + ".png"), 0)
    if not os.path.exists(os.path.join(opt.outputDir, pascal_name)):
        os.makedirs(os.path.join(opt.outputDir, pascal_name))
    pascal_average_parsing = generate_prior_single_person([0, 0, pascal_mask_img.shape[1] - 1, pascal_mask_img.shape[0] - 1],
        pascal_pose_dict[pascal_name], opt.PASCALMaskImgDir, pascal_poses,
        pascal_img_names, pascal_pose_dict, opt.n, opt.k, exclude_self=True, save_dir=os.path.join(opt.outputDir, pascal_name))
    pascal_average_parsing = pascal_average_parsing[:, :, [2, 1, 0]]
    cv2.imwrite(os.path.join(opt.outputDir, pascal_name + ".png"), pascal_average_parsing)
    print(pascal_name, i)
