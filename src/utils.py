import librosa
import numpy as np
import soundfile as sf

def spectrogram_to_audio(spec_db, sr=22050):
    # החזרת הדציבלים לסקאלה רגילה
    S = librosa.db_to_amplitude(spec_db)
    # שימוש באלגוריתם Griffin-Lim לשחזור הפאזה של גל הקול
    y_inv = librosa.griffinlim(S)
    return y_inv

def save_audio(y, path, sr=22050):
    sf.write(path, y, sr)