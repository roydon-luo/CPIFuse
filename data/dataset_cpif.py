import cv2
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetCPIF(data.Dataset):
    def __init__(self, opt):
        super(DatasetCPIF, self).__init__()
        print('Dataset: CPIF for Color Polarization Image Fusion.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(opt['dataroot_A'])
        self.paths_B = util.get_image_paths(opt['dataroot_B'])

    def __getitem__(self, index):

        # ------------------------------------
        # get S0 image and DoLP image
        # ------------------------------------
        # print('input channels:', self.n_channels)
        S0_path = self.paths_A[index]
        DoLP_path = self.paths_B[index]
        RGB_S0 = util.imread_uint(S0_path, self.n_channels)
        RGB_DoLP = util.imread_uint(DoLP_path, self.n_channels)
        max_DoLP = np.max(RGB_DoLP, axis=2, keepdims=True)
        YCbCr_S0 = cv2.cvtColor(RGB_S0, cv2.COLOR_RGB2YCrCb)
        # YCbCr_DoLP = cv2.cvtColor(RGB_DoLP, cv2.COLOR_RGB2YCrCb)
        # YCbCr_DoLP = YCbCr_DoLP[:, :, 0:1]

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get S0/DoLP patch pairs
            # --------------------------------
            """
            H, W, _ = RGB_S0.shape
            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_S0 = YCbCr_S0[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # patch_DoLP = YCbCr_DoLP_Y[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_DoLP = max_DoLP[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_S0, patch_DoLP = util.augment_img(patch_S0, mode=mode), util.augment_img(patch_DoLP, mode=mode)
            img_S0 = util.uint2tensor3(patch_S0)
            img_DoLP = util.uint2tensor3(patch_DoLP)
            return {'A': img_S0, 'B': img_DoLP, 'A_path': S0_path, 'B_path': DoLP_path}

        else:
            """
            # --------------------------------
            # get S0/DoLP image pairs
            # --------------------------------
            """
            YCbCr_S0 = util.uint2single(YCbCr_S0)
            max_DoLP = util.uint2single(max_DoLP)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_S0 = util.single2tensor3(YCbCr_S0)
            img_DoLP = util.single2tensor3(max_DoLP)

            return {'A': img_S0, 'B': img_DoLP, 'A_path': S0_path, 'B_path': DoLP_path}

    def __len__(self):
        return len(self.paths_A)
