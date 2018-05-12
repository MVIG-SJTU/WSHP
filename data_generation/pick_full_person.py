# coding: utf-8

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
import shutil
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outputpath',dest='outputpath', help='path of output', default="")
    parser.add_argument('--inputpath',dest='inputpath', help='path of inputpath', default="")
    args = parser.parse_args()
    return args

def filter_pose(intputpath, outputpath, imgname):
    save = True
    for pid in range(len(rmpe_results[imgname])):
        pose = np.array(rmpe_results[imgname][pid]['keypoints']).reshape(-1,3)[:,:3]
        for idx_c in range(16):
            if (pose[idx_c,2]) < 0.15:
                save = False
                break
        if save == False:
            break
    if save == False:
        return False
    return True

        
if __name__ == '__main__':
    args = parse_args()
    outputpath = args.outputpath
    inputpath = args.inputpath
    jsonpath = os.path.join(args.outputpath,"POSE/alpha-pose-results-forvis.json")
    
    result3={}
    with open(jsonpath) as f:
        rmpe_results = json.load(f)
    for imgname in tqdm(rmpe_results.keys()):
        if filter_pose(inputpath, outputpath, imgname):
            for pid in range(len(rmpe_results[imgname])):
                if imgname not in result3.keys():
                    result3[imgname]={}
                    result3[imgname]['version']=0.1
                    result3[imgname]['bodies']=[]
                tmp={'joints':[]}
                indexarr=[27,24,36,33,30,39,42,45,6,3,0,9,12,15,21]
                for i in indexarr:
                    tmp['joints'].append(rmpe_results[imgname][pid]['keypoints'][i])
                    tmp['joints'].append(rmpe_results[imgname][pid]['keypoints'][i+1])
                    tmp['joints'].append(rmpe_results[imgname][pid]['keypoints'][i+2])
                result3[imgname]['bodies'].append(tmp)
    with open("full-person.json",'w') as json_file:
        json_file.write(json.dumps(result3))