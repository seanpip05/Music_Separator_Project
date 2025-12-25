import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

# הגדרות נתיבים
audio_folder = 'audio'
output_folder = 'audio_chunks'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def process_audio(file_name):
    path = os.path.join(audio_folder, file_name)
    
    # 1. טעינת השיר (נריץ ב-Sample Rate של 22050 לצורך אימון מהיר)
    print(f"Loading {file_name}...")
    y, sr = librosa.load(path, sr=22050)
    
    # 2. חיתוך 5 השניות הראשונות
    duration_sec = 5
    samples_to_cut = duration_sec * sr
    y_cut = y[:samples_to_cut]
    
    # 3. שמירת החלק הקצר (מניפולציה ראשונה)
    output_path = os.path.join(output_folder, 'chunk_1.wav')
    sf.write(output_path, y_cut, sr)
    print(f"Saved 5-second chunk to: {output_path}")
    
    # 4. יצירת ספקטרוגרמה ויזואלית
    S = librosa.feature.melspectrogram(y=y_cut, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.title(f'Mel-Spectrogram of 5s chunk - {file_name}')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# הרצה על השיר שיש לך (תחליף את 'test_song.mp3' בשם הקובץ שלך)
process_audio('./Manchild.mp3')