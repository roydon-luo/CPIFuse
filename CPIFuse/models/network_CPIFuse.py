import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_A = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_A = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv_B = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_B = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out_A = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_B = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.concat_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        qkv_A = self.qkv_dwconv_A(self.qkv_A(x))
        qkv_B = self.qkv_dwconv_B(self.qkv_B(y))
        q_a, k_a, v_a = qkv_A.chunk(3, dim=1)
        q_b, k_b, v_b = qkv_B.chunk(3, dim=1)
        q_a = rearrange(q_a, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        k_a = rearrange(k_a, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        v_a = rearrange(v_a, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        q_b = rearrange(q_b, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        k_b = rearrange(k_b, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        v_b = rearrange(v_b, 'b (head c) h w -> b head c (h w)',
                        head=self.num_heads)
        q_a = torch.nn.functional.normalize(q_a, dim=-1)
        k_a = torch.nn.functional.normalize(k_a, dim=-1)
        q_b = torch.nn.functional.normalize(q_b, dim=-1)
        k_b = torch.nn.functional.normalize(k_b, dim=-1)
        attn_a = (q_b @ k_a.transpose(-2, -1)) * self.temperature
        attn_a = attn_a.softmax(dim=-1)

        out_a = (attn_a @ v_a)

        out_a = rearrange(out_a, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)

        out_a = self.project_out_A(out_a)
        out_a = out_a + x
        attn_b = -(q_a @ k_b.transpose(-2, -1)) * self.temperature
        attn_b = attn_b.softmax(dim=-1)

        out_b = (attn_b @ v_b)

        out_b = rearrange(out_b, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)

        out_b = self.project_out_B(out_b)
        out_b = out_b + y
        out = torch.concat([out_a, out_b], dim=1)
        out = self.concat_conv(out)
        return out


class Cross_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Cross_TransformerBlock, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.cattn = Cross_Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.cattn(self.norm1(x), self.norm1(y))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerBlock1(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock1, self).__init__()

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.downsampled = nn.Conv2d(dim, dim*2, kernel_size=3,
                  stride=2, padding=1, bias=bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x_ = x
        x = self.lrelu(self.downsampled(x))
        return x, x_


class TransformerBlock2(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock2, self).__init__()
        self.upsampled = Up_scale_pixel(dim)
        self.concat_conv = nn.Conv2d(dim*3//2, dim//2, kernel_size=1)
        self.norm1 = LayerNorm(dim//2, LayerNorm_type)
        self.attn = Attention(dim//2, num_heads, bias)
        self.norm2 = LayerNorm(dim//2, LayerNorm_type)
        self.ffn = FeedForward(dim//2, ffn_expansion_factor, bias)

    def forward(self, x, x_inter, y_inter):
        x = self.upsampled(x)
        x = torch.concat([x_inter, y_inter, x], dim=1)
        x = self.concat_conv(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Up_scale_pixel(nn.Module):
    def __init__(self, n_feat):
        super(Up_scale_pixel, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True), nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)


class CPIFuse(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, Ex_depths=2, Fusion_depths=1, Re_depths=2,
                 Ex_num_heads=8, Fusion_num_heads=8, Re_num_heads=8, img_range=1.,
                 **kwargs):
        super(CPIFuse, self).__init__()
        num_out_ch = in_chans
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        self.mean = torch.zeros(1, 1, 1, 1)
        self.conv_first1 = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2 = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.Ex_num_layers = Ex_depths
        self.Fusion_num_layers = Fusion_depths
        self.Re_num_layers = Re_depths
        self.embed_dim = embed_dim
        self.fusion_concat_conv = nn.Conv2d(2 * embed_dim * 3, 2 * embed_dim, kernel_size=1)
        self.re_concat_conv = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1)

        # build Restormer U-Net structure
        cnt_Ex = 0
        self.layers_Ex = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            cnt_Ex += 1
            layer = TransformerBlock1(dim=embed_dim*cnt_Ex,
                                     num_heads=Ex_num_heads,
                                     ffn_expansion_factor=2,
                                     bias=False,
                                     LayerNorm_type='WithBias')
            self.layers_Ex.append(layer)

        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = Cross_TransformerBlock(dim=embed_dim*cnt_Ex*2,
                                           num_heads=Fusion_num_heads,
                                           ffn_expansion_factor=2,
                                           bias=False,
                                           LayerNorm_type='WithBias')
            self.layers_Fusion.append(layer)

        self.layers_Reconstruction = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = TransformerBlock2(dim=embed_dim * cnt_Ex * 2,
                                           num_heads=Re_num_heads,

                                           ffn_expansion_factor=2,
                                           bias=False,
                                           LayerNorm_type='WithBias')
            self.layers_Reconstruction.append(layer)
            cnt_Ex += -1

        # high quality image reconstruction
        self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(embed_dim_temp, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward(self, A, B):
        # print("Initializing the model")
        x0 = A[:, 0:1, :, :]
        y0 = B

        self.mean_A = self.mean.type_as(x0)
        self.mean_B = self.mean.type_as(y0)
        self.mean = (self.mean_A + self.mean_B) / 2

        x0 = (x0 - self.mean_A) * self.img_range
        y0 = (y0 - self.mean_B) * self.img_range

        # Feature extraction
        x0 = self.lrelu(self.conv_first1(x0))
        x = self.lrelu(self.conv_first2(x0))
        y0 = self.lrelu(self.conv_first1(y0))
        y = self.lrelu(self.conv_first2(y0))

        inter_x = []
        inter_y = []
        for layer_Ex in self.layers_Ex:
            x, x_ = layer_Ex(x)
            y, y_ = layer_Ex(y)
            inter_x.append(x_)
            inter_y.append(y_)

        # Fusion
        for layer_Fusion in self.layers_Fusion:
            x = layer_Fusion(x, y)

        # Reconstruction
        cnt_Rec = 0
        for layer_Rec in self.layers_Reconstruction:
            x = layer_Rec(x, inter_x[1-cnt_Rec], inter_y[1-cnt_Rec])
            cnt_Rec += 1
        x = self.lrelu(self.conv_last1(x))
        x = self.conv_last2(x)

        x = x / self.img_range + self.mean
        return x