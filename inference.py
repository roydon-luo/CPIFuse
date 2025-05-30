import argparse
import cv2
import numpy as np
import os
import torch
import sys
from models.network_CPIFuse import CPIFuse as net
from utils import utils_image as util
from data.testset import Dataset as D
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='fusion', help='sr')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='./Model/test_model')
    parser.add_argument('--iter_number', type=str,
                        default='40000')
    parser.add_argument('--root_path', type=str, default='./Dataset/testsets/',
                        help='input test image root folder')
    parser.add_argument('--dataset', type=str, default='Monno/',
                        help='input test image name (Monno, Qiu, CPIF)')
    parser.add_argument('--A_dir', type=str, default='S0',
                        help='input test image name')
    parser.add_argument('--B_dir', type=str, default='DOLP',
                        help='input test image name')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=3, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
    if os.path.exists(model_path):
        print(f'loading model from {args.model_path}')
    else:
        print('Traget model path: {} not existing!!!'.format(model_path))
        sys.exit()
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    a_dir = os.path.join(args.root_path, args.dataset, args.A_dir)
    b_dir = os.path.join(args.root_path, args.dataset, args.B_dir)
    print(a_dir)
    os.makedirs(save_dir, exist_ok=True)
    test_set = D(a_dir, b_dir, args.in_channel)
    test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
    cnt = 0
    for i, test_data in enumerate(test_loader):

        imgname = test_data['A_path'][0]
        img_cr = test_data['A'][:, 1:2, :, :]
        img_cb = test_data['A'][:, 2:3, :, :]
        img_aY = test_data['A'][:, 0:1, :, :]
        img_b = test_data['B']
        img_a = img_aY.to(device)
        img_b = img_b.to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_a.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
            img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
            img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
            img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fused_img = infer(img_a, img_b, model, args, window_size)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
            print(f"Processing time is: {inference_time:.3f} ms")
            fused_img = fused_img[..., :h_old * args.scale, :w_old * args.scale]
            output = util.tensor2uint(fused_img)
            img_cr = util.tensor2uint(img_cr)
            img_cb = util.tensor2uint(img_cb)
            output = np.concatenate([np.expand_dims(output, axis=-1), np.expand_dims(img_cr, axis=-1), np.expand_dims(img_cb, axis=-1)], axis=-1)
            output = cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB)
            cnt = cnt + inference_time
        save_name = os.path.join(save_dir, os.path.basename(imgname))
        util.imsave(output, save_name)
    print(f"Average processing time is: {cnt/len(test_loader):.3f} ms")

def define_model(args):
    model = net(upscale=args.scale, in_chans=1, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=32, num_heads=[8, 8, 8, 8],
                mlp_ratio=2, upsampler=None, resi_connection='1conv')
    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model


def setup(args):   
    save_dir = f'results/CPIFuse_{args.dataset}'
    folder = os.path.join(args.root_path, args.dataset, 'A_Y')
    print('folder:', folder)
    border = 0
    window_size = 8

    return folder, save_dir, border, window_size

def infer(img_a, img_b, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_a, img_b)
    else:
        # test the image tile by tile
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
