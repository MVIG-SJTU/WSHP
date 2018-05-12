import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import ntpath
import numpy as np


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot)
        self.dir_img = os.path.join(self.dir_AB, 'img')
        self.dir_priors = os.path.join(self.dir_AB, 'prior')
        self.dir_parsing = os.path.join(self.dir_AB, 'parsing')

        self.img_paths = sorted(make_dataset(self.dir_img))
        self.parsing_paths = sorted(make_dataset(self.dir_parsing))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        # img
        img_path = self.img_paths[index % self.__len__()]
        short_path = ntpath.basename(img_path)
        img_name = os.path.splitext(short_path)[0]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        img = transforms.ToTensor()(img)

        # prior
        prior_indexes = np.arange(self.opt.n)
        if self.opt.shuffle:
            random.shuffle(prior_indexes)
        prior = torch.zeros(3, self.opt.loadSize, self.opt.loadSize)
        for i in range(self.opt.k):
            morphed_prior_path = os.path.join(self.dir_priors, img_name, str(prior_indexes[i]) + '.png')
            morphed_prior = Image.open(morphed_prior_path).convert('RGB')
            morphed_prior = morphed_prior.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            morphed_prior = transforms.ToTensor()(morphed_prior)
            prior += morphed_prior
        prior /= self.opt.k

        # parsing
        parsing_path = self.parsing_paths[index % self.__len__()]
        parsing = Image.open(parsing_path).convert('RGB')
        parsing = parsing.resize((self.opt.loadSize, self.opt.loadSize), Image.NEAREST)
        parsing = transforms.ToTensor()(parsing)

        w = img.size(2)
        h = img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        img = img[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        prior = prior[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        parsing = parsing[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        prior = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(prior)
        parsing = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(parsing)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img = img.index_select(2, idx)
            prior = prior.index_select(2, idx)
            parsing = parsing.index_select(2, idx)

        A = torch.cat([img, prior], dim=0)
        B = parsing
        if output_nc == 1: # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': img_path, 'B_paths': parsing_path}

    def __len__(self):
        return min(len(self.img_paths), len(self.parsing_paths))

    def name(self):
        return 'AlignedDataset'
