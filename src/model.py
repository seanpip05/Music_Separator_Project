import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = self.conv_block(128, 256)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), # מוסיף יציבות לאימון
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Bridge
        b = self.bridge(self.pool(e2))
        
        # Decoder
        d2 = self.up2(b)
        # תיקון למקרה של מידות לא זוגיות
        if d2.shape != e2.shape:
            d2 = F.pad(d2, [0, e2.shape[3] - d2.shape[3], 0, e2.shape[2] - d2.shape[2]])
            
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.pad(d1, [0, e1.shape[3] - d1.shape[3], 0, e1.shape[2] - d1.shape[2]])
            
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))