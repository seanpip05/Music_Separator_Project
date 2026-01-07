import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
from src.model import UNet
import os
from scipy.ndimage import zoom

st.set_page_config(page_title="UnMixer AI", layout="centered")
st.title("🎵 UnMixer AI: Pro Vocal Separator")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        try: import torch_directml; device = torch_directml.device()
        except: pass
    model = UNet(out_channels=1).to(device)
    model.load_state_dict(torch.load('models/generalist_vocals.pth', map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# סליידר לשליטה על רמת ניקוי הרעשים
noise_threshold = st.sidebar.slider("Noise Cleaning Strength", 0.0, 0.3, 0.1, 0.05)

uploaded_file = st.file_uploader("Upload Song", type=["mp3", "wav"])

if uploaded_file:
    with open("temp_in.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp_in.wav")
    
    if st.button("🚀 Separate Full Song"):
        with st.spinner("Processing full song..."):
            y, sr = librosa.load("temp_in.wav", sr=22050)
            sec_5 = 5 * sr
            output_audio = np.zeros_like(y)
            
            # עיבוד השיר בחלקים של 5 שניות
            for start in range(0, len(y), sec_5):
                end = start + sec_5
                if end > len(y): break
                
                chunk = y[start:end]
                S_full = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
                S_db = librosa.power_to_db(S_full, ref=np.max)
                S_norm = (S_db + 80) / 80
                
                inp = torch.tensor(S_norm).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    mask = model(inp).squeeze().cpu().numpy()
                
                # החלת ה-Threshold לניקוי המנגינה מהשקט
                mask[mask < noise_threshold] = 0 
                
                if mask.shape != S_full.shape:
                    mask = zoom(mask, (S_full.shape[0]/mask.shape[0], S_full.shape[1]/mask.shape[1]))
                
                S_vocals = S_full * mask
                chunk_out = librosa.feature.inverse.mel_to_audio(S_vocals, sr=sr)
                output_audio[start:start+len(chunk_out)] = chunk_out

            sf.write("out_vocals.wav", output_audio, sr)
            st.success("Separation Complete!")
            st.audio("out_vocals.wav")
            st.download_button("Download Vocals", open("out_vocals.wav", "rb"), "vocals.wav")