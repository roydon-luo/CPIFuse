import os
import cv2
import random
import torch.utils.data as data
from utils import augment_img, uint2tensor3

class DatasetReloss(data.Dataset):
    def __init__(self, phase):
        super(DatasetReloss, self).__init__()
        self.n_channels = 3
        self.patch_size = 64
        self.sigma = 25
        self.phase = phase
        if self.phase == 'train':
            self.S0_path = 'trainset/S0/'
        elif self.phase == 'val':
            self.S0_path = 'valset/S0/'

    def __getitem__(self, index):
        phase = self.phase
        S0_train = os.listdir('trainset/S0/')
        Fuse_train = os.listdir('trainset/Fuse/')
        S0_val = os.listdir('valset/S0/')
        Fuse_val = os.listdir('valset/Fuse/')
        if phase == 'train':
            S0_img = S0_train[index]
            Fuse_img = Fuse_train[index]
            S0_path = 'trainset/S0/' + S0_img
            Fuse_path = 'trainset/Fuse/' + Fuse_img
            BGR_S0    = cv2.imread(S0_path)
            BGR_Fuse    = cv2.imread(Fuse_path)
            RGB_S0  = cv2.cvtColor(BGR_S0, cv2.COLOR_BGR2RGB)
            RGB_Fuse = cv2.cvtColor(BGR_Fuse, cv2.COLOR_BGR2RGB)

            """
            # --------------------------------
            # get S0/Fuse patch pairs
            # --------------------------------
            """
            H, W, _ = RGB_S0.shape
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_S0 = RGB_S0[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_Fuse = RGB_Fuse[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_S0, patch_Fuse = augment_img(patch_S0, mode=mode), augment_img(patch_Fuse, mode=mode)
            img_S0 = uint2tensor3(patch_S0)
            img_Fuse = uint2tensor3(patch_Fuse)
            return {'S0': img_S0,  'Fuse': img_Fuse,
                    'S0_path': S0_path, 'Fuse_path': Fuse_path}
        elif phase == 'val':
            S0_img = S0_val[index]
            Fuse_img = Fuse_val[index]
            S0_path = 'valset/S0/' + S0_img
            Fuse_path = 'valset/Fuse/' + Fuse_img
            BGR_S0 = cv2.imread(S0_path)
            BGR_Fuse = cv2.imread(Fuse_path)
            RGB_S0 = cv2.cvtColor(BGR_S0, cv2.COLOR_BGR2RGB)
            RGB_Fuse = cv2.cvtColor(BGR_Fuse, cv2.COLOR_BGR2RGB)
            img_S0 = uint2tensor3(RGB_S0)
            img_Fuse = uint2tensor3(RGB_Fuse)
            return {'S0': img_S0, 'Fuse': img_Fuse,
                    'S0_path': S0_path, 'Fuse_path': Fuse_path}

    def __len__(self):
        return len(os.listdir(self.S0_path))




