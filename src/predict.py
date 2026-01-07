import torch
import librosa
import numpy as np
import soundfile as sf
import os
from src.model import UNet

# הגדרות חומרה
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    try:
        import torch_directml
        device = torch_directml.device()
    except:
        pass

MODEL_PATH = 'models/generalist_vocals.pth'
INPUT_SONG = 'AUDIO/Manchild.mp3' 
OUTPUT_PATH = 'AUDIO/separated_vocals.wav'

def predict():
    # 1. טעינת המודל
    model = UNet(out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"🚀 Model loaded on {device}")

    # 2. טעינת 5 שניות בדיוק
    sr = 22050
    duration = 5
    y, _ = librosa.load(INPUT_SONG, sr=sr, duration=duration) 
    
    # הגדרת פרמטרים קבועים לשחזור איכותי
    n_fft = 2048
    hop_length = 512 # הפרמטר שקובע את ה"אורך" בזמן
    
    # הפיכה לספקטרוגרמה
    S_full = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S_full, ref=np.max)
    
    # נירמול (0-1)
    S_norm = (S_db + 80) / 80
    
    # הכנה למודל
    input_tensor = torch.tensor(S_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    # 3. הרצה דרך המודל
    print("🎵 Separating vocals...")
    with torch.no_grad():
        mask = model(input_tensor)
    
    # 4. שחזור האודיו
    mask = mask.squeeze().cpu().numpy()
    if mask.shape != S_full.shape:
        import scipy.ndimage
        mask = scipy.ndimage.zoom(mask, (S_full.shape[0]/mask.shape[0], S_full.shape[1]/mask.shape[1]))

    # במקום לשחזר מה-dB, אנחנו מכפילים את הספקטרוגרמה המקורית במסיכה
    # זה שומר על הדינמיקה המקורית של השיר
    S_vocals = S_full * mask 
    
    print("🔊 Reconstructing audio with original phase...")
    y_out = librosa.feature.inverse.mel_to_audio(S_vocals, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 5. שמירה
    sf.write(OUTPUT_PATH, y_out, sr)
    print(f"✅ Success! Vocals saved (5 seconds) to: {OUTPUT_PATH}")

if __name__ == "__main__":
    if not os.path.exists('AUDIO'):
        os.makedirs('AUDIO')
    predict()