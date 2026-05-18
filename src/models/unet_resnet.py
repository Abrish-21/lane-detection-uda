import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint


class UNetWithResNetEncoder(nn.Module):
    def __init__(self, n_classes=2, gradient_checkpointing=False):
        super().__init__()
        self.n_classes = n_classes
        self.gradient_checkpointing = gradient_checkpointing

        try:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        except AttributeError:
            resnet = models.resnet34(pretrained=True)

        # Encoder
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64,  H/2
        self.pool  = resnet.maxpool                                         # 64,  H/4
        self.enc2  = resnet.layer1                                          # 64,  H/4
        self.enc3  = resnet.layer2                                          # 128, H/8
        self.enc4  = resnet.layer3                                          # 256, H/16
        self.enc5  = resnet.layer4                                          # 512, H/32

        # Decoder
        self.up5  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec5 = self._make_decoder_block(256 + 256, 256)

        self.up4  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = self._make_decoder_block(128 + 128, 128)

        self.up3  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(64 + 64, 64)

        self.up2   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)

        # Weight init for decoder layers only
        self._init_decoder_weights()

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _init_decoder_weights(self):
        for m in [self.up5, self.dec5, self.up4, self.dec4,
                  self.up3, self.dec3, self.up2, self.final]:
            for layer in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def _crop_to_match(tensor, target):
        return tensor[:, :, :target.shape[2], :target.shape[3]]

    def _enc_forward(self, x):
        """Encoder forward, optionally with gradient checkpointing."""
        e1 = self.enc1(x)
        p  = self.pool(e1)
        if self.gradient_checkpointing and self.training:
            e2 = checkpoint(self.enc2, p,    use_reentrant=False)
            e3 = checkpoint(self.enc3, e2,   use_reentrant=False)
            e4 = checkpoint(self.enc4, e3,   use_reentrant=False)
            e5 = checkpoint(self.enc5, e4,   use_reentrant=False)
        else:
            e2 = self.enc2(p)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)
            e5 = self.enc5(e4)
        return e1, e2, e3, e4, e5

    def forward(self, x):
        e1, e2, e3, e4, e5 = self._enc_forward(x)

        d5 = self.up5(e5)
        d5 = self._crop_to_match(d5, e4)
        d5 = self.dec5(torch.cat([d5, e4], dim=1))

        d4 = self.up4(d5)
        d4 = self._crop_to_match(d4, e3)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        d3 = self._crop_to_match(d3, e2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        out = self.up2(d3)
        # Upsample back to input resolution if needed (handles the enc1 stride-2 step)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.final(out)
        return out