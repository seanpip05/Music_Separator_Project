import torch
import librosa
import numpy as np
import soundfile as sf
from src.model import UNet
import os

# הגדרות מכשיר
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    try:
        import torch_directml
        device = torch_directml.device()
    except:
        pass

MODEL_PATH = 'models/unet_highres_vocals.pth'
INPUT_SONG = 'AUDIO/Manchild.mp3'
OUTPUT_PATH = 'AUDIO/vocals_highres.wav'

def predict():
    sr = 22050
    n_fft = 2048
    hop_length = 512

    # 1. טעינת מודל
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    model = UNet(out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. טעינת אודיו (10 שניות לבדיקה)
    print(f"📂 Loading: {INPUT_SONG}")
    y, _ = librosa.load(INPUT_SONG, sr=sr, duration=10)
    
    # 3. יצירת ספקטרוגרמה (STFT)
    stft_mix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mix_mag = np.abs(stft_mix)
    S_mix_phase = np.angle(stft_mix)

    # 4. נירמול (ליניארי 0-1)
    mix_max = np.max(S_mix_mag)
    S_norm = S_mix_mag / (mix_max + 1e-8)
    
    input_tensor = torch.tensor(S_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    # 5. ניבוי המסיכה
    print("🧠 Predicting and enhancing mask...")
    with torch.no_grad():
        mask = model(input_tensor).squeeze().cpu().numpy()

    # --- תיקון והקשחת המסיכה (השינוי הקריטי) ---
    # העלאה בחזקה (3) גורמת לערכים חלשים להיעלם ולחזקים להישאר
    mask = np.power(mask, 3)
    
    # סף חיתוך: כל מה שמתחת ל-0.15 הופך לשקט מוחלט
    mask[mask < 0.15] = 0
    
    # החלקת המסיכה למניעת רעשים מתכתיים (אופציונלי)
    import scipy.ndimage
    mask = scipy.ndimage.gaussian_filter(mask, sigma=0.5)
    # ------------------------------------------

    # תיקון ממדים אם צריך
    if mask.shape != S_norm.shape:
        from scipy.ndimage import zoom
        mask = zoom(mask, (S_norm.shape[0]/mask.shape[0], S_norm.shape[1]/mask.shape[1]))

    # 6. שחזור הסאונד
    S_vocals_mag = (S_norm * mask) * mix_max
    S_vocals_stft = S_vocals_mag * np.exp(1j * S_mix_phase)
    y_out = librosa.istft(S_vocals_stft, hop_length=hop_length)

    # 7. נירמול עוצמה סופי
    if np.max(np.abs(y_out)) > 0:
        y_out = y_out / np.max(np.abs(y_out))

    # 8. שמירה
    sf.write(OUTPUT_PATH, y_out, sr)
    print(f"✅ Enhanced Vocals saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    predict()