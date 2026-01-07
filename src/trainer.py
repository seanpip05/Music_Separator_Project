import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import UNet
from src.data_loader import MusicDataset
import os
import time

# הגדרת חומרה (AMD / NVIDIA / CPU)
def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        print("🚀 High-Resolution Training on AMD GPU (DirectML)")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Training on {device}")
    return device

device = get_device()

def train():
    # --- הגדרות לאימון לילה ---
    epochs = 20            # כ-7 שעות אימון (מתאים לשינת לילה)
    batch_size = 4 
    learning_rate = 0.0005 
    model_path = 'models/unet_highres_vocals.pth'
    
    # טעינת הדאטה
    train_dataset = MusicDataset(
        mix_dir='data/processed/mix', 
        target_dir='data/processed/vocals'
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # אתחול המודל
    model = UNet(out_channels=1).to(device)
    
    # טעינה אוטומטית של מה שהתאמנת עליו (כדי להמשיך ולא להתחיל מאפס)
    if os.path.exists(model_path):
        print(f"📦 Found existing model weights. Resuming progress...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # רשימה לשמירת היסטוריית ה-Loss לסיכום הסופי
    loss_history = []
    
    print(f"\n🌙 Overnight session started. Target: {epochs} epochs.")
    print(f"📊 Dataset Size: {len(train_dataset)} segments")
    print("-" * 45)

    overall_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print("⚠️ NaN detected in loss! Skipping batch...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # הדפסה כל 50 באצ'ים (כדי שהלוג יהיה נקי בבוקר)
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.6f}")
 
        
        # סיכום סוף אפוק
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        epoch_duration = time.time() - epoch_start_time
        
        print(f"\n✅ [Epoch {epoch+1} Done]")
        print(f"   - Average Loss: {avg_loss:.6f}")
        print(f"   - Duration: {epoch_duration/60:.2f} minutes")
        print("-" * 35)
        
        # שמירה בסוף כל אפוק (ביטחון למקרה של הפסקת חשמל)
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_path)

    # --- סיכום בוקר סופי ---
    total_duration = time.time() - overall_start_time
    print("\n" + "="*55)
    print("🏆 GOOD MORNING! TRAINING SESSION COMPLETE")
    print(f"Total Session Time: {total_duration/3600:.2f} hours")
    print(f"Starting Loss: {loss_history[0]:.6f}")
    print(f"Final Loss: {loss_history[-1]:.6f}")
    print("\n📈 Loss History (Per Epoch):")
    for i, loss in enumerate(loss_history):
        print(f"Epoch {i+1}: {loss:.6f}")
    print("="*55)

if __name__ == "__main__":
    train()