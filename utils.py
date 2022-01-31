import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from basic import *


def window_partition(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class InputProj(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, norm_layer=None,
                 act_layer=nn.GELU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            act_layer()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)

        return x


class OutputProj(nn.Module):
    def __init__(self, in_channels=64, out_channels=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            act_layer()
        )

    def forward(self, x):
        x = self.proj(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, input_size):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        H, W = input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchShuffle(nn.Module):
    def __init__(self, dim, out_dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, out_dim * (dim_scale ** 2), bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x, input_size, out_size):
        H, W = input_size
        Hout, Wout = out_size
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).contiguous()
        if H % self.dim_scale != 0 or W % self.dim_scale != 0:
            H_pad = self.dim_scale - H % self.dim_scale
            W_pad = self.dim_scale - W % self.dim_scale
            x = F.pad(x, (0, 0, 0, W_pad, 0, H_pad))

        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x[:, :Hout, :Wout, :]
        x = x.reshape(B, -1, C // (self.dim_scale ** 2)).contiguous()
        x = self.norm(x)

        return x


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True, guide=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.proj_in = nn.Identity()
        self.guide = guide
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)

    def forward(self, x, x_guide=None):
        B_, N, C = x.shape
        if self.guide:
            kv = self.to_kv(x_guide).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.to_kv(x).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, input_size=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWCFF(nn.Module):
    def __init__(self, dim=32, out_dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim,
                                              kernel_size=3, stride=1, padding=1),
                                    act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, input_size):
        # bs x hw x c
        B, L, C = x.size()
        H, W = input_size
        assert H * W == L, "output H x W is not the same with L!"

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)  # bs, hidden_dim, 32x32
        x = self.dwconv(x)

        # flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 guide=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.guide = guide

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = LinearProjection(dim, num_heads, self.dim // num_heads, bias=qkv_bias, guide=guide)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_guide=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, x_guide)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'



class TransformerBlock(nn.Module):
    def __init__(self, dim, out_dim,
                 num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 token_mlp='dwc', guide=False):
        super().__init__()
        self.dim = out_dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.guide = guide
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(out_dim)
        self.attn = WindowAttention(
            out_dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            guide=guide)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        if token_mlp == 'dwc':
            self.mlp = DWCFF(out_dim, out_dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
            
        else:
            self.mlp = FFN(in_features=out_dim, out_features=out_dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)

        self.proj_in = nn.Identity()
        if dim != out_dim:
            self.proj_in = nn.Linear(dim, out_dim)

        self.H = None
        self.W = None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, x_guide=None):
        x = self.proj_in(x)
        B, L, C = x.shape
        H, W = self.H, self.W
        assert H * W == L, "input H x W is not the same with L!"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.guide:
            C_guide = x_guide.size(-1)
            x_guide = self.norm1(x_guide).view(B, H, W, -1)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.win_size - W % self.win_size) % self.win_size
        pad_b = (self.win_size - H % self.win_size) % self.win_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        if self.guide:
            x_guide = F.pad(x_guide, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, Hp, Wp, 1)).type_as(x).detach()  # 1 H W 1
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.win_size)  # nW, win_size, win_size, 1
            mask_windows = mask_windows.view(-1, self.win_size * self.win_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.type_as(x)
        else:
            attn_mask = None

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.guide:
                x_guide = torch.roll(x_guide, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        if self.guide:
            x_guide = window_partition(x_guide, self.win_size)  # nW*B, win_size, win_size, C
            x_guide = x_guide.view(-1, self.win_size * self.win_size, C_guide)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, x_guide=x_guide, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), (H, W)))

        del attn_mask

        return x


class GuideFormerLayer(nn.Module):
    def __init__(self, dim, out_dim,
                 depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 token_mlp='dwc', use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.blocks.append(TransformerBlock(dim=dim, out_dim=out_dim,
                                                         num_heads=num_heads, win_size=win_size,
                                                         shift_size=0 if (i % 2 == 0) else win_size // 2,
                                                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[i] if isinstance(drop_path,
                                                                                              list) else drop_path,
                                                         norm_layer=norm_layer, token_mlp=token_mlp))
            else:
                self.blocks.append(TransformerBlock(dim=out_dim, out_dim=out_dim,
                                                         num_heads=num_heads, win_size=win_size,
                                                         shift_size=0 if (i % 2 == 0) else win_size // 2,
                                                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[i] if isinstance(drop_path,
                                                                                              list) else drop_path,
                                                         norm_layer=norm_layer, token_mlp=token_mlp))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, input_size):
        H, W = input_size
        B, L, C = x.shape
        assert H * W == L, "input H x W is not the same with L!"

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x



class FusionLayer(nn.Module):
    def __init__(self, dim, out_dim,
                 depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0.1, norm_layer=nn.LayerNorm, 
                 token_mlp='dwc', use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:
                self.blocks.append(TransformerBlock(dim=dim, out_dim=out_dim,
                                                         num_heads=num_heads, win_size=win_size,
                                                         shift_size=0 if (i % 2 == 0) else win_size // 2,
                                                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[i] if isinstance(drop_path,
                                                                                              list) else drop_path,
                                                         norm_layer=norm_layer, token_mlp=token_mlp,
                                                         guide=True))
            else:
                self.blocks.append(TransformerBlock(dim=out_dim, out_dim=out_dim,
                                                         num_heads=num_heads, win_size=win_size,
                                                         shift_size=0 if (i % 2 == 0) else win_size // 2,
                                                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[i] if isinstance(drop_path,
                                                                                              list) else drop_path,
                                                         norm_layer=norm_layer, token_mlp=token_mlp,
                                                         guide=True))

    def forward(self, depth_feat, rgb_feat, input_size):
        H, W = input_size
        B, L, C = rgb_feat.shape
        assert H * W == L, "input H x W is not the same with L!"

        x = depth_feat
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rgb_feat)
            else:
                x = blk(x, x_guide=rgb_feat)  # B L 2C

        return x
