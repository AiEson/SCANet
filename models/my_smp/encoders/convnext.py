# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

import torch
import torch.nn as nn
from efficientnet_pytorch.model import MemoryEfficientSwish


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        group_split=[4, 4],
        kernel_sizes=[5],
        window_size=4,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
    ):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(
                nn.Conv2d(
                    3 * self.dim_head * group_head,
                    3 * self.dim_head * group_head,
                    kernel_size,
                    1,
                    kernel_size // 2,
                    groups=3 * self.dim_head * group_head,
                )
            )
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(
                nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias)
            )
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(
                dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias
            )
            self.global_kv = nn.Conv2d(
                dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias
            )
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = (
                nn.AvgPool2d(window_size, window_size)
                if window_size != 1
                else nn.Identity()
            )

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ln = LayerNorm(dim, data_format="channels_first")

    def high_fre_attntion(
        self,
        x: torch.Tensor,
        to_qkv: nn.Module,
        mixer: nn.Module,
        attn_block: nn.Module,
    ):
        """
        x: (b c h w)
        """
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = (
            mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        )  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(
        self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module
    ):
        """
        x: (b c h w)
        """
        b, c, h, w = x.size()

        q = (
            to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
        )  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = (
            to_kv(kv)
            .view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2))
            .permute(1, 0, 2, 4, 3)
            .contiguous()
        )  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):

        """
        x: (b c h w)
        """
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(
                self.high_fre_attntion(
                    x, self.qkvs[i], self.convs[i], self.act_blocks[i]
                )
            )
        if self.group_split[-1] != 0:
            res.append(
                self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool)
            )
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


import torchvision

from ._base import EncoderMixin

from timm.models.layers import make_divisible


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitCoordAtt(nn.Module):
    """Split-Attention (aka Splat)"""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        radix=4,
        rd_ratio=0.25,
        rd_channels=None,
        rd_divisor=8,
        act_layer=nn.GELU,
        norm_layer=None,
        drop_layer=None,
        **kwargs,
    ):
        super(SplitCoordAtt, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor
            )
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels,
            mid_chs,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            # bias=bias,
            **kwargs,
        )
        # self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=out_channels)
        self.conv2 = EfficientAttention(out_channels, kernel_sizes=[3], window_size=8)
        self.bn0 = (
            norm_layer(mid_chs)
            if norm_layer
            else LayerNorm(mid_chs, data_format="channels_first")
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2_0 = nn.Conv2d(attn_chs, out_channels, 1, groups=groups)
        self.fc2_1 = nn.Conv2d(attn_chs, out_channels, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)
        # self.bn_final = LayerNorm(out_channels)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):

        x = self.conv(x)  # 3x3 Conv C = mid_chs
        x = self.bn0(x)  # BN
        x = self.drop(x)
        x = self.act0(x)  # ReLU

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x

        # Replace with x Pool y Pool

        x_h = self.pool_h(x_gap)
        x_w = self.pool_w(x_gap).permute(0, 1, 3, 2)
        x_gap = self.conv2(x_gap)

        y = torch.cat([x_h, x_w], dim=2)
        # print(f'y.shape = {y.shape}')
        y = self.fc1(y)
        y = self.bn1(y)
        y = self.act1(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        # print(f'x_h.shape = {x_h.shape}')
        x_w = x_w.permute(0, 1, 3, 2)
        # print(f'x_w.shape = {x_w.shape}')

        a_h = self.fc2_0(x_h).sigmoid()
        # a_h = self.rsoftmax(a_h).reshape((B, -1, H, 1))
        # print(f'a_h.shape = {a_h.shape}')

        a_w = self.fc2_1(x_w).sigmoid()
        # a_c = x_c.sigmoid()
        # a_w = self.rsoftmax(a_w).reshape((B, -1, 1, W))
        # print(f'a_w.shape = {a_w.shape}')

        x_attn = a_h * a_w * x_gap
        # print(f'x_attn.shape = {x_attn.shape}')
        if self.radix > 1:
            x = x.sum(dim=1)
        else:
            x = x.reshape((B, RC, H, W))

        # print(f'x.shape = {x.shape}, x_attn.shape = {x_attn.shape}')
        # print(x.max(), x.min())
        x = x * x_attn
        # x = self.bn_final(x)
        return x


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, radix=4):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(
            dim, 4 * dim, kernel_size=1
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim, groups=dim)
        # print(dim * 4)
        self.sca = SplitCoordAtt(4 * dim, dim, groups=dim, radix=radix, kernel_size=3)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.pwconv2(x)

        # print(x.shape)

        x = self.sca(x)
        # print(x.shape)
        if self.gamma is not None:
            x = self.gamma * x

        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[3, 96, 192, 384, 768],
        drop_path_rate=0.4,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3],
        output_stride=32,
        radix=4,
    ):
        super().__init__()
        self.depths = depths
        # self.output_stride = output_stride
        # self.out_channels = out_channels
        self._out_channels = dims
        # print(dims)
        dims = dims[1:]

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        radix=radix,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(depths)):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        _init_weights()

    def forward_features(self, x):
        outs = []
        outs.append(x)
        for i in range(len(self.depths)):
            # print(i)
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            norm_layer = getattr(self, f"norm{i}")
            x_out = norm_layer(x)
            outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # self.weight = nn.Parameter(torch.ones(normalized_shape))
        # self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # self.eps = eps
        # self.data_format = data_format
        # if self.data_format not in ["channels_last", "channels_first"]:
        #     raise NotImplementedError
        # self.normalized_shape = (normalized_shape,)
        self.bn = nn.BatchNorm2d(normalized_shape)
        # nn.InstanceNorm2d

    def forward(self, x):
        # if self.data_format == "channels_last":
        #     return F.layer_norm(
        #         x, self.normalized_shape, self.weight, self.bias, self.eps
        #     )
        # elif self.data_format == "channels_first":
        #     u = x.mean(1, keepdim=True)
        #     s = (x - u).pow(2).mean(1, keepdim=True)
        #     x = (x - u) / torch.sqrt(s + self.eps)
        #     x = self.weight[:, None, None] * x + self.bias[:, None, None]
        #     return x
        return self.bn(x)


class ConvNeXtEncoder(ConvNeXt, EncoderMixin):
    def __init__(self, depth=4, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._in_channels = 3
        self._output_stride = 32

    def get_stages(self):
        return self.forward_features

    def forward(self, x):
        return self.get_stages()(x)

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)


convnext_encoders = {
    "convnext_b": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": dict(
            in_chans=3,
            depths=[3, 3, 27, 3],
            dims=[3, 128, 256, 512, 1024, 1024 * 2],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            out_indices=[0, 1, 2, 3],
        ),
    },
    "convnext_l": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": dict(
            in_chans=3,
            depths=[3, 3, 27, 3, 3],
            dims=[3, 192, 384, 768, 1536, 1536 * 2],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            out_indices=[0, 1, 2, 3],
        ),
    },
    "convnext_s": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": dict(
            in_chans=3,
            depths=[3, 3, 27, 3, 3],
            dims=[3, 96, 192, 384, 768, 768 * 2],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            out_indices=[0, 1, 2, 3],
        ),
    },
    "convnext_t": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": dict(
            in_chans=3,
            depths=[3, 3, 9, 3, 3],
            dims=[3, 96, 192, 384, 768, 768 * 2],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            out_indices=[0, 1, 2, 3, 4],
        ),
    },
    "convnext_xl": {
        "encoder": ConvNeXtEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": dict(
            in_chans=3,
            depths=[3, 3, 27, 3],
            dims=[3, 256, 512, 1024, 2048],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            out_indices=[0, 1, 2, 3],
        ),
    },
}
