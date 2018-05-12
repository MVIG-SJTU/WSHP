'''
Variables and functions for prior generation.
The order of keypoints is:
    0-'right ankle' 1-'right knee' 2-'right hip' 3-'left hip' 4-'left knee' 5-'left ankle' 6-'pelvis' 7-'thorax' 8-'neck'
    9-'head' 10-'right wrist' 11-'right elbow' 12-'right shoulder' 13-'left shoulder' 14-'left elbow' 15-'left wrist'.
When pelvis is missing, we use the midpoint of two hips instead.
Thorax is unused and set to be the same(0, 0).
'''

import numpy as np
import os
import copy
import cv2
import random


'''
Used for aligning torso.
'''
RIGHT_LEG = [6, 2, 1, 0]
LEFT_LEG = [6, 3, 4, 5]
RIGHT_ARM = [6, 8, 12, 11, 10]
LEFT_ARM = [6, 8, 13, 14, 15]
HEAD = [6, 8, 9]
SPINE = [6, 8]
RIGHT_ARM_SPINE = [8, 12, 11, 10]
LEFT_ARM_SPINE = [8, 13, 14, 15]
HEAD_SPINE = [8, 9]

'''
Colors for each part of body.
'''
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

'''
Merge 1 - 10 labels to 1 - 6 labels,
e.g. left && right upper arm ==> upper arm.
'''
merge_mask = np.array([0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.uint8)

'''
Morphing is done for each body part associated with following skeleton lines and then merged together.
'''
main_skeleton_lines = [
    [], 
    [8, 9], 
    [6, 8], 
    [13, 14], 
    [12, 11], 
    [14, 15], 
    [11, 10], 
    [3, 4], 
    [2, 1], 
    [4, 5], 
    [1, 0]]

'''
Lines of skeleton.
'''
skeletonLines = [
    [0, 1],
    [1, 2],
    [2, 6],
    [6, 3],
    [3, 4],
    [4, 5],
    [6, 8],
    [8, 9],
    [8, 12],
    [12, 11],
    [11, 10],
    [8, 13],
    [13, 14],
    [14, 15]
]

'''
Dictionary of (color : RGB) pair.
'''
colorDict = {
    "purple": [255, 0, 128],
    "orange": [1, 96, 254],
    "light_blue": [255, 141, 28],
    "dark_blue": [232, 0, 0],
    "red": [0, 0, 255]
}

'''
Colors for each skeleton line.
'''
skeletonColor = ["orange", "orange", "orange", "light_blue", "light_blue", "light_blue",
    "purple", "purple", "red", "red", "red", "dark_blue", "dark_blue", "dark_blue"
]


def drawSkeleton(img, pose):
    '''
    Given an image and a pose, draw the skeleton on that image.

    :param img:
        Image to draw skeleton on.
    :param pose:
        Pose of shape (1 x 32).
    :return:
        Image with skeleton lines.
    '''
    retImg = copy.deepcopy(img)
    pose = pose[0]
    for i in range(len(skeletonLines)):
        a = skeletonLines[i][0]
        b = skeletonLines[i][1]
        cv2.line(retImg, (int(pose[a*2]), int(pose[a*2+1])), (int(pose[b*2]), int(pose[b*2+1])), list(map(lambda i: i*0.6, colorDict[skeletonColor[i]])), 3)
    return retImg


def align_torso(poses):
    '''
    Align torso length to the same(50).

    :param poses:
        2-dimension array. The shape should be N x 32.
    :return:
        Aligned pose array.
    '''
    poses_new = copy.deepcopy(poses)
    for i in range(1, 4):
        poses_new[:, (2 * RIGHT_LEG[i]):(2 * RIGHT_LEG[i] + 2)] = poses[:, (2 * RIGHT_LEG[i]):(2 * RIGHT_LEG[i] + 2)] - poses[:, (2 * RIGHT_LEG[i - 1]):(2 * RIGHT_LEG[i - 1] + 2)]
        poses_new[:, 2 * LEFT_LEG[i]:2 * LEFT_LEG[i] + 2] = poses[:, 2 * LEFT_LEG[i]:2 * LEFT_LEG[i] + 2] - poses[:, 2 * LEFT_LEG[i - 1]:2 * LEFT_LEG[i - 1] + 2]
    for i in range(1, 5):
        poses_new[:, 2 * RIGHT_ARM[i]:2 * RIGHT_ARM[i] + 2] = poses[:, 2 * RIGHT_ARM[i]:2 * RIGHT_ARM[i] + 2] - poses[:, 2 * RIGHT_ARM[i - 1]:2 * RIGHT_ARM[i - 1] + 2]
        poses_new[:, 2 * LEFT_ARM[i]:2 * LEFT_ARM[i] + 2] = poses[:, 2 * LEFT_ARM[i]:2 * LEFT_ARM[i] + 2] - poses[:, 2 * LEFT_ARM[i - 1]:2 * LEFT_ARM[i - 1] + 2]
    for i in range(1, 3):
        poses_new[:, 2 * HEAD[i]:2 * HEAD[i] + 2] = poses[:, 2 * HEAD[i]:2 * HEAD[i] + 2] - poses[:, 2 * HEAD[i - 1]:2 * HEAD[i - 1] + 2]

    ratio = 50 / np.sqrt(np.square(poses_new[:, 16:17]) + np.square(poses_new[:, 17:18]))
    poses_ret = poses_new * np.tile(ratio[:, 0:1], [1, 32])

    for i in range(1, 4):
        poses_ret[:, (2 * RIGHT_LEG[i]):(2 * RIGHT_LEG[i] + 2)] = poses_ret[:, (2 * RIGHT_LEG[i]):(2 * RIGHT_LEG[i] + 2)] + poses_ret[:, (2 * RIGHT_LEG[i - 1]):(2 * RIGHT_LEG[i - 1] + 2)]
        poses_ret[:, 2 * LEFT_LEG[i]:2 * LEFT_LEG[i] + 2] = poses_ret[:, 2 * LEFT_LEG[i]:2 * LEFT_LEG[i] + 2] + poses_ret[:, 2 * LEFT_LEG[i - 1]:2 * LEFT_LEG[i - 1] + 2]
    for i in range(1, 2):
        poses_ret[:, (2 * SPINE[i]):(2 * SPINE[i] + 2)] = poses_ret[:, (2 * SPINE[i]):(2 * SPINE[i] + 2)] + poses_ret[:, (2 * SPINE[i - 1]):(2 * SPINE[i - 1] + 2)]
    for i in range(1, 4):
        poses_ret[:, 2 * RIGHT_ARM_SPINE[i]:2 * RIGHT_ARM_SPINE[i] + 2] = poses_ret[:, 2 * RIGHT_ARM_SPINE[i]:2 * RIGHT_ARM_SPINE[i] + 2] + poses_ret[:, 2 * RIGHT_ARM_SPINE[i - 1]:2 * RIGHT_ARM_SPINE[i - 1] + 2]
        poses_ret[:, 2 * LEFT_ARM_SPINE[i]:2 * LEFT_ARM_SPINE[i] + 2] = poses_ret[:, 2 * LEFT_ARM_SPINE[i]:2 * LEFT_ARM_SPINE[i] + 2] + poses_ret[:, 2 * LEFT_ARM_SPINE[i - 1]:2 * LEFT_ARM_SPINE[i - 1] + 2]
    for i in range(1, 2):
        poses_ret[:, 2 * HEAD_SPINE[i]:2 * HEAD_SPINE[i] + 2] = poses_ret[:, 2 * HEAD_SPINE[i]:2 * HEAD_SPINE[i] + 2] + poses_ret[:, 2 * HEAD_SPINE[i - 1]:2 * HEAD_SPINE[i - 1] + 2]

    center = np.array((250, 250), dtype=float)
    centers = np.tile(center, (poses.shape[0], 1))
    centers = centers - poses_ret[:, 12:14]
    centers = np.tile(centers, (1, 16))
    poses_ret = poses_ret + centers

    return poses_ret


def load_pascal_pose(pascal_pose_file_root):
    '''
    Load preprocessed Pascal pose file.

    :param pascal_pose_file_root:
        Root of pascal pose file.
    :return:
        pose_arr: 2-dimension array of aligned pose of shape N x 32, thorax set to be (0, 0).
        img_names: list of image names.
        pose_dict: dictionary of (image name : unaligned pose[1 x 32]) pairs.
    '''
    pascal_pose_file = open(pascal_pose_file_root, "r")

    pose_list = []
    img_names = []
    pose_dict = {}

    line_count = 0
    while True:
        line = pascal_pose_file.readline()
        if not line:
            break
        words = line.split(",")
        img_name = words[0]
        pose_tmp = np.zeros((1, 32), dtype=float)
        for i in range(16):
            x = words[1 + 3 * i]
            y = words[1 + 3 * i + 1]
            # words[1 + 3 * i + 2] is_visible (not used)
            
            pose_tmp[0][2 * i] = float(x)
            pose_tmp[0][2 * i + 1] = float(y)

        has_negative = False
        for i in range(16):
            if pose_tmp[0][2 * i] < 0 or pose_tmp[0][2 * i + 1] < 0:
                has_negative = True
                break
        if has_negative:
            continue
        pose_tmp[0][12] = (pose_tmp[0][4] + pose_tmp[0][6]) / 2
        pose_tmp[0][13] = (pose_tmp[0][5] + pose_tmp[0][7]) / 2
        pose_list.append(pose_tmp)
        img_names.append(img_name[0:len(img_name) - 4])
        pose_dict[img_name[0:len(img_name) - 4]] = pose_tmp

        line_count += 1

    pascal_pose_file.close()

    pose_arr = np.zeros((len(pose_list), 32), dtype=float)
    for i in range(len(pose_list)):
        pose_arr[i] = pose_list[i]

    pose_arr = align_torso(pose_arr)
    pose_arr[:, 14:16] = np.tile(np.zeros((1, 2), dtype=float), (len(pose_list), 1))
    return pose_arr, img_names, pose_dict


def paint(mask_img, merge):
    '''
    Paint parsing result(mask) to color image.

    :param mask_img:
        1-channel parsing result.
    :param merge:
        0 or 1. If 1, merge 10 parts into 6 parts.
    :return:
        Corresponding color image.
    '''
    assert (len(mask_img.shape) == 2)
    if merge:
        return body_part_color[merge_mask[mask_img]]
    else:
        return body_part_color[mask_img]


def morphing(origin_mask_img, origin_pose, target_pose, target_size):  # target_size [width, height]
    '''
    According to origin pose and target pose, morph the origin mask image so as to get the same pose as the target pose.

    :param origin_mask_img:
        Origin mask image, 1-channel, of labels 0-10 (0 for backgraound).
    :param origin_pose:
        1-dimension pose array, of shape (32, ).
    :param target_pose:
        1-dimension pose array, of shape (32, ).
    :param target_size:
        Target image size: [width, height].
    :return:
        Color image of morphed mask image, of size target_size.
    '''
    assert (len(origin_mask_img.shape) == 2)
    assert (len(origin_pose.shape) == 1)
    assert (len(target_pose.shape) == 1)

    target_mask_img = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    # morphing for each part
    for label in range(1, 11):
        origin_size = np.array([origin_mask_img.shape[1], origin_mask_img.shape[0]], dtype=int)
        origin_body_part = origin_mask_img * (origin_mask_img == label)
        a = main_skeleton_lines[label][0]
        b = main_skeleton_lines[label][1]
        origin_pose_part_a = np.array([origin_pose[a * 2], origin_pose[a * 2 + 1]], dtype=float)
        origin_pose_part_b = np.array([origin_pose[b * 2], origin_pose[b * 2 + 1]], dtype=float)
        origin_pose_part_tensor = origin_pose_part_b - origin_pose_part_a
        target_pose_part_a = np.array([target_pose[a * 2], target_pose[a * 2 + 1]], dtype=float)
        target_pose_part_b = np.array([target_pose[b * 2], target_pose[b * 2 + 1]], dtype=float)
        target_pose_part_tensor = target_pose_part_b - target_pose_part_a
        origin_pose_part_length = np.sqrt(np.sum(np.square(origin_pose_part_tensor)))
        target_pose_part_length = np.sqrt(np.sum(np.square(target_pose_part_tensor)))
        # scaling ratio
        scale_factor = target_pose_part_length / origin_pose_part_length
        if scale_factor == 0:
            continue
        # rotating angle
        theta = - (np.arctan2(target_pose_part_tensor[1], target_pose_part_tensor[0]) - np.arctan2(
            origin_pose_part_tensor[1], origin_pose_part_tensor[0])) * 180 / np.pi

        ''' scale '''
        origin_size[0] *= scale_factor
        origin_size[1] *= scale_factor
        origin_pose_part_a *= scale_factor
        origin_pose_part_b *= scale_factor
        origin_body_part = cv2.resize(origin_body_part, (origin_size[0], origin_size[1]),
                                      interpolation=cv2.INTER_NEAREST)
        # print("finish scale", label)

        ''' translate to the center in case rotation out of the image '''
        origin_pose_part_center = (origin_pose_part_a + origin_pose_part_b) / 2
        origin_center = origin_size / 2
        tx = origin_center[0] - int(origin_pose_part_center[0])
        ty = origin_center[1] - int(origin_pose_part_center[1])
        tm = np.float32([[1, 0, tx], [0, 1, ty]])
        origin_body_part = cv2.warpAffine(origin_body_part, tm, (origin_size[0], origin_size[1]))
        # print("finish translate", label)

        ''' rotate '''
        rm = cv2.getRotationMatrix2D((origin_center[0], origin_center[1]), theta, 1)
        origin_body_part = cv2.warpAffine(origin_body_part, rm, (origin_size[0], origin_size[1]))
        origin_body_part = (origin_body_part != 0) * label
        # print("finish rotate", label)

        ''' crop and paste '''
        target_pose_part_center = (target_pose_part_a + target_pose_part_b) / 2
        target_pose_part_center[0] = int(target_pose_part_center[0])
        target_pose_part_center[1] = int(target_pose_part_center[1])
        if target_pose_part_center[1] >= origin_center[1]:
            origin_row_low = 0
            target_row_low = target_pose_part_center[1] - origin_center[1]
        else:
            origin_row_low = origin_center[1] - target_pose_part_center[1]
            target_row_low = 0
        if (target_size[1] - target_pose_part_center[1]) >= (origin_size[1] - origin_center[1]):
            origin_row_high = origin_size[1]
            target_row_high = target_pose_part_center[1] + origin_size[1] - origin_center[1]
        else:
            origin_row_high = origin_center[1] + target_size[1] - target_pose_part_center[1]
            target_row_high = target_size[1]
        if target_pose_part_center[0] >= origin_center[0]:
            origin_col_low = 0
            target_col_low = target_pose_part_center[0] - origin_center[0]
        else:
            origin_col_low = origin_center[0] - target_pose_part_center[0]
            target_col_low = 0
        if (target_size[0] - target_pose_part_center[0]) >= (origin_size[0] - origin_center[0]):
            origin_col_high = origin_size[0]
            target_col_high = target_pose_part_center[0] + origin_size[0] - origin_center[0]
        else:
            origin_col_high = origin_center[0] + target_size[0] - target_pose_part_center[0]
            target_col_high = target_size[0]
        origin_row_low = int(origin_row_low)
        target_row_low = int(target_row_low)
        origin_row_high = int(origin_row_high)
        target_row_high = int(target_row_high)
        origin_col_low = int(origin_col_low)
        target_col_low = int(target_col_low)
        origin_col_high = int(origin_col_high)
        target_col_high = int(target_col_high)
        target_mask_img[target_row_low:target_row_high, target_col_low:target_col_high] = np.maximum(
            target_mask_img[target_row_low:target_row_high, target_col_low:target_col_high],
            origin_body_part[origin_row_low:origin_row_high, origin_col_low:origin_col_high])
        # print("finish crop and paste", label)

    return paint(target_mask_img, merge=True)


def random_k_within_n(n, k):
    '''
    Get random k number in [0, n).

    :param n:
    :param k:
    :return:
    '''
    arr = np.arange(n)
    random.shuffle(arr)
    return arr[0:k]


def find_nearest_pose(aligned_pose, pascal_poses):
    '''
    Given a aligned pose, order the aligned pascal pose array and return the index array.

    :param aligned_pose:
        Aligned pose(throax set to (0, 0)), 1-d(32, ) or 2-d(1, 32).
    :param pascal_poses:
        Aligned pascal poses, of shape(N x 32).
    :return:
        Ordered index array.
    '''
    pose_tmp = np.tile(aligned_pose, (pascal_poses.shape[0], 1))
    pose_pascal_distance = np.sum(np.square(pose_tmp - pascal_poses), axis=1)
    distance_index = np.argsort(pose_pascal_distance)
    return distance_index


def generate_prior_single_person(bbox, raw_pose, PASCALMaskImgDir, pascal_poses, pascal_img_names, pascal_pose_dict, n, k, exclude_self=False, save_dir=None):
    '''
    Generate prior for a single person.

    :param bbox:
        Bounding box of the person(can be image size).
        The pose will be aligned according to the upper left coordinate of bbox.
        The generated prior image size is the same as bbox.
        The lower right point is inclusive.
    :param raw_pose:
        Unaligned pose of shape (1, 32).
    :param save_dir:
        Path to save n nearest priors. If is None, do not save.
    :param PASCALMaskImgDir:
        Path to pascal mask images.
    :param pascal_poses:
        2-d array of aligned pascal poses.
    :param pascal_img_names:
        List of pascal image names.
    :param pascal_pose_dict:
        Dictionary of (image name : unaligned pose) pairs.
    :param n:
        Pick n nearest poses.
    :param k:
        Average nearest k priors.
    :param exclude_self:
        If 1, when we are finding nearest pose for pascal image in pascal images, the nearset one must be itself, so that the first one should be excluded.
        If 0, the first one not excluded.
    :return:
        The averaged prior. At the same time, n priors are saved in save_dir.
    '''
    # raw_pose: [[x1,y1,x2,y2,...,x16,y16]]
    aligned_pose = align_torso(raw_pose)
    aligned_pose[:, 14:16] = np.tile(np.zeros((1, 2), dtype=float), (1, 1))
    distance_index = find_nearest_pose(aligned_pose, pascal_poses)
    if exclude_self:
        distance_index = distance_index[1:]

    # bbox: [x1, y1, x2, y2]
    width = bbox[2] - bbox[0] + 1
    heigth = bbox[3] - bbox[1] + 1
    pose = copy.deepcopy(raw_pose)
    pose -= np.tile(np.array([bbox[0], bbox[1]]), (1, 16))
    pose = pose[0]
    origin_size = np.array([int(width), int(heigth)], dtype=int)
    average_parsing = np.zeros((heigth, width, 3), dtype=float)

    close_pascal_index = distance_index[random_k_within_n(n, k)]
    for j in range(n):
        if distance_index[j] not in close_pascal_index and save_dir is None:
            continue
        pascal_name = pascal_img_names[distance_index[j]]
        # get PASCAL mask img and morph
        print(j, 'picked pascal', pascal_name)
        pascal_mask_img = cv2.imread(os.path.join(PASCALMaskImgDir, pascal_name + ".png"), 0)
        # cv2.imwrite(os.path.join(save_dir, 'origin_' + str(j) + '_' + pascal_name + '.png'), paint(pascal_mask_img))
        pascal_pose = pascal_pose_dict[pascal_name][0]
        morphingImg = morphing(pascal_mask_img, pascal_pose, pose, origin_size)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, str(j) + '.png'), morphingImg[:, :, [2, 1, 0]])
        if distance_index[j] in close_pascal_index:
            average_parsing += morphingImg

    average_parsing /= k
    return average_parsing
