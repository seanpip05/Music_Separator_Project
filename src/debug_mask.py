import torch
from src.model import UNet
import numpy as np

model = UNet(out_channels=1)
model.load_state_dict(torch.load('models/unet_highres_vocals.pth', map_location='cpu'))
model.eval()

# ניצור מידע דמי (כמו שיר)
test_input = torch.rand(1, 1, 1025, 216) 
with torch.no_grad():
    output = model(test_input)

print(f"Mask Average Value: {output.mean().item():.4f}")
print(f"Mask Max Value: {output.max().item():.4f}")
print(f"Mask Min Value: {output.min().item():.4f}")