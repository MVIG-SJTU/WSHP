from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] > 3:
        image_numpy_1 = image_numpy[0:3, :, :]
        image_numpy_2 = image_numpy[3:, :, :]
        image_numpy_1 = (np.transpose(image_numpy_1, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy_2 = (np.transpose(image_numpy_2, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy_1.astype(imtype), image_numpy_2.astype(imtype)

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


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


def postprocess_parsing(image_numpy, isTrain):
    imtype = image_numpy.dtype
    image_numpy = image_numpy[:, :, 0:1]
    image_numpy = np.tile(image_numpy, (1, 1, 7)).astype(int)
    standard = np.arange(7)
    standard *= 35
    standard = standard.reshape((1, 1, 7))
    standard = np.tile(standard, (image_numpy.shape[0], image_numpy.shape[1], 1))
    diff = np.abs(image_numpy - standard)
    min_index = np.argmin(diff, axis=2).reshape(((image_numpy.shape[0], image_numpy.shape[1], 1)))
    image_numpy = np.tile(min_index, (1, 1, 3))
    if isTrain:
        image_numpy = image_numpy * 35
    return image_numpy.astype(imtype)


def paint_color(image_numpy):
    image_np = image_numpy[:, :, 0]
    if image_np.max() > 10:
        image_np = image_np / 35
    image_np = image_np.astype(np.uint8)
    return body_part_color[image_np]


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
