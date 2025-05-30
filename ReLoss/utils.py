import torch
import numpy as np
import cv2
from pyciede2000 import ciede2000

def cal_deltaE(S0, image_fused):
    S0 = tensor2uint(S0)
    image_fused = tensor2uint(image_fused)
    lab1 = cv2.cvtColor(S0, cv2.COLOR_BGR2Lab)
    lab2 = cv2.cvtColor(image_fused, cv2.COLOR_BGR2Lab)
    L1, a1, b1 = cv2.split(lab1)
    L2, a2, b2 = cv2.split(lab2)
    tmp_list = []
    for i in range(L1.shape[0]):
        for j in range(L1.shape[1]):
            tmp = ciede2000((L1[i, j], a1[i, j], b1[i, j]), (L2[i, j], a2[i, j], b2[i, j]))
            tmp_list.append(tmp['delta_E_00'])
    return sum(tmp_list)/(L1.shape[0]*L1.shape[1])

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.item() / batch_size)
    return res

def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_val = 0
        self.local_sum = 0
        self.local_count = 0

    def update(self, val, n=1):
        self.local_val = val
        self.local_sum += val * n
        self.local_count += n
        self.val = self.local_val
        self.sum = self.local_sum
        self.count = self.local_count
        self.avg = self.sum / self.count
