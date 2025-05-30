import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
import cv2

class Dataset(data.Dataset):
    def __init__(self, root_A, root_B, in_channels):
        super(Dataset, self).__init__()
        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_A = util.get_image_paths(root_A)
        self.paths_B = util.get_image_paths(root_B)
        self.inchannels = in_channels

    def __getitem__(self, index):

        # ------------------------------------
        # get DoLP image and S0 image
        # ------------------------------------
        A_path = self.paths_A[index]
        B_path = self.paths_B[index]
        img_A = util.imread_uint(A_path, self.inchannels)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_RGB2YCrCb)
        img_B = util.imread_uint(B_path, self.inchannels)
        img_B = np.max(img_B, axis=2, keepdims=True)
       
        """
        # --------------------------------
        # get testing image pairs
        # --------------------------------
        """
        img_A = util.uint2single(img_A)
        img_B = util.uint2single(img_B)
        # --------------------------------
        # HWC to CHW, numpy to tensor
        # --------------------------------
        img_A = util.single2tensor3(img_A)
        img_B = util.single2tensor3(img_B)

        return {'A': img_A, 'B': img_B, 'A_path': A_path, 'B_path': B_path}

    def __len__(self):
        return len(self.paths_A)
