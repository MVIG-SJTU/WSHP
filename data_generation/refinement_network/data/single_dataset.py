import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)
        self.dir_img = os.path.join(self.dir_A, 'img')
        self.dir_prior = os.path.join(self.dir_A, 'prior')

        self.img_paths = sorted(make_dataset(self.dir_img))
        self.prior_paths = sorted(make_dataset(self.dir_prior))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        img = transforms.ToTensor()(img)

        prior_path = self.prior_paths[index]
        prior = Image.open(prior_path).convert('RGB')
        prior = prior.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
        prior = transforms.ToTensor()(prior)

        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        prior = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(prior)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        A = torch.cat([img, prior], dim=0)

        return {'A': A, 'A_paths': img_path}

    def __len__(self):
        return len(self.img_paths)

    def name(self):
        return 'SingleDataset'
