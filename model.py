import torch
import torch.nn as nn
import timm


class SEBlock(nn.Module):
    """Squeeze-and-Excitation блок."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialPyramidPooling(nn.Module):
    """Пирамидальный пулинг для учёта разномасштабных признаков."""
    def __init__(self, channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.conv = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        p1 = self.pool1(x)
        p1 = nn.functional.interpolate(p1, size=(h, w), mode="bilinear", align_corners=False)
        p2 = self.pool2(x)
        p2 = nn.functional.interpolate(p2, size=(h, w), mode="bilinear", align_corners=False)
        p4 = self.pool4(x)
        p4 = nn.functional.interpolate(p4, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([p1, p2, p4], dim=1)
        return self.conv(out)


def adapt_conv_stem(backbone, in_channels):
    """
    Адаптация свёрточного стека EfficientNet под произвольное число каналов.
    Сохраняет предобученные веса для первых трёх каналов, остальные инициализирует средним.
    """
    old = backbone.conv_stem
    k = old.kernel_size if hasattr(old, "kernel_size") else (3, 3)
    out_ch = old.out_channels
    stride = old.stride if hasattr(old, "stride") else (2, 2)
    padding = old.padding if hasattr(old, "padding") else (1, 1)
    new_conv = nn.Conv2d(in_channels, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False)
    with torch.no_grad():
        w_old = old.weight
        if w_old.shape[1] >= 3:
            new_conv.weight[:, :3, :, :] = w_old[:, :3, :, :]
            if in_channels > 3:
                mean_weight = w_old.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c+1, :, :] = mean_weight
        else:
            mean_weight = w_old.mean(dim=1, keepdim=True)
            for c in range(in_channels):
                new_conv.weight[:, c:c+1, :, :] = mean_weight
    backbone.conv_stem = new_conv
    return backbone


class MarineNet(nn.Module):
    """Основная модель классификации на базе EfficientNet-B0."""
    def __init__(self, in_channels):
        super().__init__()
        # Используем features_only=True для доступа к картам признаков
        backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, features_only=True)
        backbone = adapt_conv_stem(backbone, in_channels)
        self.backbone = backbone
        feature_channels = self.backbone.feature_info[-1]["num_chs"]
        self.se = SEBlock(feature_channels)
        self.spp = SpatialPyramidPooling(feature_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(feature_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        feats = self.se(feats)
        feats = self.spp(feats)
        feats = self.pool(feats)
        feats = feats.flatten(1)
        out = self.head(feats)
        return out.squeeze(1)  # логиты