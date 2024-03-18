import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import EncoderMixin
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import make_divisible
from timm.models.registry import register_model
from timm.models.resnet import ResNet


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
        radix=2,
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
            bias=bias,
            **kwargs,
        )
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2_0 = nn.Conv2d(attn_chs, out_channels, 1, groups=groups)
        self.fc2_1 = nn.Conv2d(attn_chs, out_channels, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_h = self.pool_h(x_gap)
        x_w = self.pool_w(x_gap).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.fc1(y) # Group Conv
        y = self.bn1(y)
        y = self.act1(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.fc2_0(x_h) # Group Conv
        a_h = self.rsoftmax(a_h).reshape((B, -1, H, 1))
        a_w = self.fc2_1(x_w) # Group Conv
        a_w = self.rsoftmax(a_w).reshape((B, -1, 1, W))

        x_attn = a_h * a_w # To Attn. Map
        if self.radix > 1:
            x = x.sum(dim=1)
        else:
            x = x.reshape((B, RC, H, W))
        return x * x_attn


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (7, 7),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "conv1.0",
        "classifier": "fc",
        **kwargs,
    }


default_cfgs = {
    "resnest14d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth"  # noqa
    ),
    "resnest26d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth"  # noqa
    ),
    "resnest50d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth"
    ),
    "resnest101e": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth",
        input_size=(3, 256, 256),
        pool_size=(8, 8),
    ),
    "resnest200e": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth",
        input_size=(3, 320, 320),
        pool_size=(10, 10),
        crop_pct=0.909,
        interpolation="bicubic",
    ),
    "resnest269e": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth",
        input_size=(3, 416, 416),
        pool_size=(13, 13),
        crop_pct=0.928,
        interpolation="bicubic",
    ),
    "resnest50d_4s2x40d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth",  # noqa
        interpolation="bicubic",
    ),
    "resnest50d_1s4x24d": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth",  # noqa
        interpolation="bicubic",
    ),
}


def cov_feature(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h * w
    x = x.reshape(batchsize, dim, M)
    I_hat = (-1.0 / M / M) * torch.ones(dim, dim, device=x.device) + (
        1.0 / M
    ) * torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    return y


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ResNeSCBottleneck(nn.Module):
    """ResNet Bottleneck"""

    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        radix=4,
        cardinality=1,
        base_width=64,
        avd=False,
        avd_first=False,
        is_first=False,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        att_dim=128,
        reduction=16,
    ):
        super(ResNeSCBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        norm_layer = LayerNorm
        act_layer = nn.GELU

        group_width = int(planes * (base_width / 64.0)) * cardinality
        # group_width = group_width // 4
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(group_width, group_width)
        self.act1 = act_layer()
        self.avd_first = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and avd_first
            else None
        )

        if self.radix >= 1:
            self.conv2 = SplitCoordAtt(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                radix=radix,
                norm_layer=norm_layer,
                drop_layer=drop_block,
            )
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width,
                group_width,
                kernel_size=3,
                stride=stride,
                padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality,
                # bias=False,
            )
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer()
        self.avd_last = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and not avd_first
            else None
        )

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.act3 = act_layer()

        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        # out = self.triplet_attention(out)
        out_o = out + shortcut
        # out_o = self.act3(out_o)
        return out_o


# def _create_resnesc(variant, pretrained=False, **kwargs):
#     return build_model_with_cfg(ResNet, f"{variant}", pretrained, **kwargs)


# @register_model
# def resnest14d(pretrained=False, **kwargs):
#     """ResNeSt-14d model. Weights ported from GluonCV."""
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[1, 1, 1, 1],
#         stem_type="deep",
#         stem_width=32,
#         avg_down=True,
#         base_width=64,
#         cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False),
#         **kwargs,
#     )
#     return _create_resnesc("resnest14d", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest26d(pretrained=False, **kwargs):
#     """ResNeSt-26d model. Weights ported from GluonCV."""
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[2, 2, 2, 2],
#         stem_type="deep",
#         stem_width=32,
#         avg_down=True,
#         base_width=64,
#         cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False),
#         **kwargs,
#     )
#     return _create_resnesc("resnest26d", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest50d(pretrained=False, **kwargs):
#     """ResNeSt-50d model. Matches paper ResNeSt-50 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'd' for deep stem, stem_width 32, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[3, 4, 6, 3],
#         stem_type="deep",
#         stem_width=32,
#         avg_down=True,
#         base_width=64,
#         cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False),
#         **kwargs,
#     )
#     return _create_resnesc("resnest50d", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest101e(pretrained=False, **kwargs):
#     """ResNeSt-101e model. Matches paper ResNeSt-101 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[3, 4, 23, 3],
#         stem_type="deep",
#         stem_width=64,
#         avg_down=True,
#         base_width=64,
#         cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False),
#         **kwargs,
#     )
#     return _create_resnesc("resnest101e", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest200e(pretrained=False, **kwargs):
#     """ResNeSt-200e model. Matches paper ResNeSt-200 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[3, 24, 36, 3],
#         stem_type="deep",
#         stem_width=64,
#         avg_down=True,
#         base_width=64,
#         cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False),
#         **kwargs,
#     )
#     return _create_resnesc("resnest200e", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest269e(pretrained=False, **kwargs):
#     """ResNeSt-269e model. Matches paper ResNeSt-269 model, https://arxiv.org/abs/2004.08955
#     Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
#     """
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[3, 30, 48, 8],
#         stem_type="deep",
#         stem_width=64,
#         avg_down=True,
#         base_width=64,
#         cardinality=1,
#         block_args=dict(radix=2, avd=True, avd_first=False),
#         **kwargs,
#     )
#     return _create_resnesc("resnest269e", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest50d_4s2x40d(pretrained=False, **kwargs):
#     """ResNeSt-50 4s2x40d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md"""
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[3, 4, 6, 3],
#         stem_type="deep",
#         stem_width=32,
#         avg_down=True,
#         base_width=40,
#         cardinality=2,
#         block_args=dict(radix=4, avd=True, avd_first=True),
#         **kwargs,
#     )
#     return _create_resnesc("resnest50d_4s2x40d", pretrained=pretrained, **model_kwargs)


# @register_model
# def resnest50d_1s4x24d(pretrained=False, **kwargs):
#     """ResNeSt-50 1s4x24d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md"""
#     model_kwargs = dict(
#         block=ResNeSCBottleneck,
#         layers=[3, 4, 6, 3],
#         stem_type="deep",
#         stem_width=32,
#         avg_down=True,
#         base_width=24,
#         cardinality=4,
#         block_args=dict(radix=1, avd=True, avd_first=True),
#         **kwargs,
#     )
#     return _create_resnesc("resnest50d_1s4x24d", pretrained=pretrained, **model_kwargs)


# ________________ Register Module To smp ______________________


class ResNeSCEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.global_pool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.act1),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def make_dilated(self, *args, **kwargs):
        raise ValueError("SegRSNet encoders do not support dilated mode")

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


resnest_weights = {
    "scanet-14d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth",  # noqa
    },
    "scanet-26d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth",  # noqa
    },
    "scanet-50d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth",  # noqa
    },
    "scanet-101e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth",  # noqa
    },
    "scanet-200e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth",  # noqa
    },
    "scanet-269e": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth",  # noqa
    },
    "scanet-50d_4s2x40d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth",  # noqa
    },
    "scanet-50d_1s4x24d": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in resnest_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }

scanet_encoders = {
    "scanet-14d": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-14d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [1, 1, 1, 1],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "scanet-26d": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-26d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [2, 2, 2, 2],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "scanet-50d": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-50d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "scanet-101e": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-101e"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 23, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "scanet-200e": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-200e"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 24, 36, 3],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 1,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "scanet-269e": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-269e"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 30, 48, 8],
            "stem_type": "deep",
            "stem_width": 64,
            "avg_down": True,
            "base_width": 64,
            "cardinality": 4,
            "block_args": {"radix": 2, "avd": True, "avd_first": False},
        },
    },
    "scanet-50d_4s2x40d": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-50d_4s2x40d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 40,
            "cardinality": 2,
            "block_args": {"radix": 4, "avd": True, "avd_first": True},
        },
    },
    "scanet-50d_1s4x24d": {
        "encoder": ResNeSCEncoder,
        "pretrained_settings": pretrained_settings["scanet-50d_1s4x24d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": ResNeSCBottleneck,
            "layers": [3, 4, 6, 3],
            "stem_type": "deep",
            "stem_width": 32,
            "avg_down": True,
            "base_width": 24,
            "cardinality": 4,
            "block_args": {"radix": 1, "avd": True, "avd_first": True},
        },
    },
}


# for k, v in timm_resnesc_encoders.items():
#     name = f"{k}"
#     smp.encoders.encoders[name] = v
#     # print(f"Added Model:\t{name}")
# print(f"Added SCA Models!!!")

# if __name__ == "__main__":
#     model = smp.Unet("scanet-14d", encoder_weights=None, in_channels=1, classes=1)
#     a = torch.randn(1, 1, 256, 256)
#     b = model(a)
#     print(b.shape)
