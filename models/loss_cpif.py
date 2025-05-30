import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
from models.loss_dnn import ReLoss

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


class L_MSE(nn.Module):
    def __init__(self):
        super(L_MSE, self).__init__()

    def forward(self, S0, image_fused):
        Loss_MSE = F.mse_loss(image_fused, S0, reduction='mean')
        return Loss_MSE


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.gradconv = MultiGrad()

    def forward(self, S0, DoLP, image_fused):
        gradient_S0 = self.gradconv(S0)
        gradient_DoLP = self.gradconv(DoLP)
        gradient = torch.max(gradient_S0.abs(), gradient_DoLP.abs())*torch.sign(gradient_S0 + gradient_DoLP)
        gradient_fused = self.gradconv(image_fused)
        Loss_gradient = F.huber_loss(gradient_fused, gradient, reduction='mean', delta=0.1)
        return Loss_gradient


class MultiGrad(nn.Module):
    def __init__(self):
        super(MultiGrad, self).__init__()
        dx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        dy = torch.tensor([[-1, 2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        dxx = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=torch.float32)
        dyy = torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=torch.float32)
        dxy = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        dx = dx.cuda()
        dy = dy.cuda()
        dxx = dxx.cuda()
        dyy = dyy.cuda()
        dxy = dxy.cuda()
        self.dx = dx.unsqueeze(0).unsqueeze(0)
        self.dy = dy.unsqueeze(0).unsqueeze(0)
        self.dxx = dxx.unsqueeze(0).unsqueeze(0)
        self.dxy = dxy.unsqueeze(0).unsqueeze(0)
        self.dyy = dyy.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        Fx = F.conv2d(x, self.dx, padding=1)
        Fy = F.conv2d(x, self.dy, padding=1)
        Fxx = F.conv2d(x, self.dxx, padding=1)
        Fyy = F.conv2d(x, self.dyy, padding=1)
        Fxy = F.conv2d(x, self.dxy, padding=1)
        Grad = torch.cat((Fx, Fy, Fxx, Fxy, Fyy), dim=1)
        return Grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Freq(nn.Module):
    def __init__(self):
        super(L_Freq, self).__init__()
        self.FocalFreq = FocalFrequencyLoss()

    def forward(self, S0, DoLP, image_fused):
        Loss_low_Freq, Loss_high_Freq = self.FocalFreq(S0, DoLP, image_fused)
        return Loss_low_Freq, Loss_high_Freq


class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = 1.0
        self.alpha = 1.0
        self.patch_factor = 1
        self.ave_spectrum = False
        self.log_matrix = False
        self.batch_matrix = False

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fft2(y, norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 2, onesided=False, normalized=True)
        return freq

    def gaussian_filter(self, h, w, sigma, mu):
        x, y = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
        d = np.sqrt(x * x + y * y)
        sigma_, mu_ = sigma, mu  # 0.5, 0
        g = np.exp(-((d - mu_) ** 2 / (1.0 * sigma_ ** 2)))

        return g

    def loss_formulation(self, real_freq1, real_freq2, recon_freq, matrix=None, gaussian_=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            gaussian_ = gaussian_.view(1, recon_freq.size(1), recon_freq.size(2), recon_freq.size(3),
                                         recon_freq.size(4))
            recon_high_freq = recon_freq
            recon_high_freq[:, :, :, :, :, 0] = (1 - gaussian_) * recon_high_freq[:, :, :, :, :, 0]
            recon_low_freq = recon_freq
            recon_low_freq[:, :, :, :, :, 0] = gaussian_ * recon_low_freq[:, :, :, :, :, 0]
            real1_high_freq = real_freq1
            real1_high_freq[:, :, :, :, :, 0] = (1 - gaussian_) * real1_high_freq[:, :, :, :, :, 0]
            real1_low_freq = real_freq1
            real1_low_freq[:, :, :, :, :, 0] = gaussian_ * real1_low_freq[:, :, :, :, :, 0]
            real2_high_freq = real_freq2
            real2_high_freq[:, :, :, :, :, 0] = (1 - gaussian_) * real2_high_freq[:, :, :, :, :, 0]
            real2_low_freq = real_freq2
            real2_low_freq[:, :, :, :, :, 0] = gaussian_ * real2_low_freq[:, :, :, :, :, 0]
            matrix_tmp1 = (recon_high_freq - real1_high_freq) ** 2
            # matrix_tmp1 = (recon_low_freq - real1_low_freq) ** 2
            matrix_tmp1 = torch.sqrt(matrix_tmp1[..., 0] + matrix_tmp1[..., 1]) ** self.alpha
            matrix_tmp2 = (recon_high_freq - real2_high_freq) ** 2
            # matrix_tmp2 = (recon_low_freq - real2_low_freq) ** 2
            matrix_tmp2 = torch.sqrt(matrix_tmp2[..., 0] + matrix_tmp2[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp1 = torch.log(matrix_tmp1 + 1.0)
                matrix_tmp2 = torch.log(matrix_tmp2 + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp1 = matrix_tmp1 / matrix_tmp1.max()
                matrix_tmp2 = matrix_tmp2 / matrix_tmp2.max()
            else:
                matrix_tmp1 = matrix_tmp1 / matrix_tmp1.max(-1).values.max(-1).values[:, :, :, None, None]
                matrix_tmp2 = matrix_tmp2 / matrix_tmp2.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp1[torch.isnan(matrix_tmp1)] = 0.0
            matrix_tmp1 = torch.clamp(matrix_tmp1, min=0.0, max=1.0)
            weight_matrix1 = matrix_tmp1.clone().detach()

            matrix_tmp2[torch.isnan(matrix_tmp2)] = 0.0
            matrix_tmp2 = torch.clamp(matrix_tmp2, min=0.0, max=1.0)
            weight_matrix2 = matrix_tmp2.clone().detach()

            new_weight_matrix1 = weight_matrix1 / (weight_matrix1 + weight_matrix2)
            new_weight_matrix2 = weight_matrix2 / (weight_matrix1 + weight_matrix2)

        # frequency distance using (squared) Euclidean distance
        tmp_low = (recon_low_freq - real1_low_freq) ** 2
        freq_distance0 = tmp_low[..., 0] + tmp_low[..., 1]

        tmp1_high = (recon_high_freq - real1_high_freq) ** 2
        freq_distance1 = tmp1_high[..., 0] + tmp1_high[..., 1]

        tmp2_high = (recon_high_freq - real2_high_freq) ** 2
        freq_distance2 = tmp2_high[..., 0] + tmp2_high[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss_low = freq_distance0
        loss_high = torch.max(freq_distance1, freq_distance2)
        # loss_high = new_weight_matrix1 * freq_distance1 + new_weight_matrix2 * freq_distance2
        return torch.mean(loss_low), torch.mean(loss_high)

    def forward(self, target1, target2, pred, matrix=None, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq1 = self.tensor2freq(target1)
        target_freq2 = self.tensor2freq(target2)

        gaussian = self.gaussian_filter(pred_freq.size(3), pred_freq.size(4), 10.0, 0)
        gaussian_ = np.fft.ifftshift(gaussian)
        gaussian_ = torch.Tensor(gaussian_).cuda()

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq1 = torch.mean(target_freq1, 0, keepdim=True)
            target_freq2 = torch.mean(target_freq2, 0, keepdim=True)

        # calculate focal frequency loss
        loss_low, loss_high = self.loss_formulation(target_freq1, target_freq2, pred_freq, matrix, gaussian_)
        return loss_low, loss_high


class fusion_loss_cpif(nn.Module):
    def __init__(self):
        super(fusion_loss_cpif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_MSE = L_MSE()
        self.L_Freq = L_Freq()
        state_dict = torch.load('./models/loss_module_deltae.ckpt', map_location='cpu')
        self.L_DNN = ReLoss()
        self.L_DNN.load_state_dict(state_dict)

    def gaussian_filter(self, h, w, sigma, mu):
        x, y = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
        d = np.sqrt(x * x + y * y)
        sigma_, mu_ = sigma, mu  # 0.5, 0
        g = np.exp(-((d - mu_) ** 2 / (1.0 * sigma_ ** 2)))

        return g

    def forward(self, S0, DoLP, image_fused):
        kernel_size = 3
        mean_kernel = torch.ones(1, 1, kernel_size, kernel_size)/(kernel_size * kernel_size)
        mean_kernel = mean_kernel.cuda()
        S0_Y = S0[:, 0:1, :, :]
        S0_lf = torch.cat((F.conv2d(S0_Y, mean_kernel, padding=kernel_size // 2),
                          F.conv2d(S0[:, 1:2, :, :], mean_kernel, padding=kernel_size // 2),
                          F.conv2d(S0[:, 2:3, :, :], mean_kernel, padding=kernel_size // 2)), dim=1)
        Fuse_lf = torch.cat((F.conv2d(image_fused, mean_kernel, padding=kernel_size // 2),
                          F.conv2d(S0[:, 1:2, :, :], mean_kernel, padding=kernel_size // 2),
                          F.conv2d(S0[:, 2:3, :, :], mean_kernel, padding=kernel_size // 2)), dim=1)

        loss_DNN = 1 * self.L_DNN(S0_lf, Fuse_lf) # The output of ReLoss during training has been multiplied by 100
        loss_Gradient = 100 * self.L_Grad(S0_Y, DoLP, image_fused)
        loss_low_Frequency, loss_high_Frequency = self.L_Freq(S0_Y, DoLP, image_fused)
        loss_low_Frequency = loss_low_Frequency * 100
        loss_high_Frequency = loss_high_Frequency * 100

        fusion_loss = 2 * loss_DNN + 1 * loss_Gradient + 1 * loss_low_Frequency + 1 * loss_high_Frequency
        return fusion_loss, loss_DNN, loss_Gradient, loss_low_Frequency+loss_high_Frequency