import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import UNet
from src.data_loader import MusicDataset
import os

# בחירת חומרה: DirectML ל-AMD בבית, אחרת CPU
try:
    import torch_directml
    device = torch_directml.device()
    print("Running on AMD GPU via DirectML")
except ImportError:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

def train():
    # הגדרות היפר-פרמטרים
    epochs = 20
    batch_size = 16
    learning_rate = 0.001
    
    # טעינת הדאטה (תוודא שהנתיבים האלו קיימים בתיקיית data/processed)
    # בשלב האימון הראשוני נתמקד ב-Vocals כדוגמה
    train_dataset = MusicDataset(
        mix_dir='data/processed/mix', 
        target_dir='data/processed/vocals'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # אתחול המודל, פונקציית הפסד ואופטימייזר
    model = UNet(out_channels=1).to(device)
    criterion = nn.MSELoss() # מודד את ההפרש בין המסיכה שהמודל יצר למקור
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass (עדכון המשקולות)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
        
        # שמירת המודל בסוף כל סבב (Epoch)
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), f'models/generalist_vocals.pth')

if __name__ == "__main__":
    train()