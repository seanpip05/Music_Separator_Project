import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c, dropout=0.1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False), # bias=False משפר יציבות עם BatchNorm
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout), # מוסיף חוסן למודל
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2)

        # Bottleneck - כאן קורה הקסם
        self.bottleneck = conv_block(256, 512, dropout=0.3)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final Layer - חייב Sigmoid בשביל Masking
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder עם Skip Connections
        d3 = self.up3(b)
        d3 = self.match_and_concat(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.match_and_concat(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.match_and_concat(e1, d1)
        d1 = self.dec1(d1)

        return self.final(d1)

    def match_and_concat(self, bridge, upsampled):
        # פונקציית עזר לחיבור בטוח של השכבות
        diffY = bridge.size()[2] - upsampled.size()[2]
        diffX = bridge.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        return torch.cat([bridge, upsampled], dim=1)