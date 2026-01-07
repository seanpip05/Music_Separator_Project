import torch
from src.model import UNet
import numpy as np

model = UNet(out_channels=1)
model.load_state_dict(torch.load('models/unet_highres_vocals.pth', map_location='cpu'))
model.eval()

# ניצור קלט דמי
test_input = torch.rand(1, 1, 1025, 216) 
with torch.no_grad():
    output = model(test_input)

print(f"Mean value: {output.mean().item():.4f}")
print(f"Max value: {output.max().item():.4f}")
print(f"Min value: {output.min().item():.4f}")