import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


class Up_scale_van(nn.Module):
    def __init__(self, in_channel,kernel_size):
        super(Up_scale_van, self).__init__()
        # self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)
        self.main=BasicConv(in_channel, in_channel//2,kernel_size,1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x=self.main(x)
        return x
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        return self.main(x)
class Corss_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Corss_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_A = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv_A = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.qkv_B = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_B = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out_A = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out_B = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.concat_conv=nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
    def forward(self, x,y):
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
        out_a=out_a+x
        attn_b = -(q_a @ k_b.transpose(-2, -1)) * self.temperature
        attn_b = attn_b.softmax(dim=-1)

        out_b = (attn_b @ v_b)

        out_b = rearrange(out_b, 'b head c (h w) -> b (head c) h w',
                          head=self.num_heads, h=h, w=w)

        out_b = self.project_out_B(out_b)
        out_b = out_b + y
        out=torch.concat([out_a,out_b],dim=1)
        out=self.concat_conv(out)
        return out

class Cross_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Cross_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.cattn = Corss_Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.cattn(self.norm1(x), self.norm1(y))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)
class Up_scale_van(nn.Module):
    def __init__(self, in_channel,kernel_size):
        super(Up_scale_van, self).__init__()
        # self.main = BasicConv(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)
        self.main=BasicConv(in_channel, in_channel//2,kernel_size,1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x=self.main(x)
        return x
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

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight
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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
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