import os
import librosa
import soundfile as sf
import numpy as np

def slice_audio_files(raw_data_path, output_base_path, segment_len=5, sr=22050):
    """
    סורק את MUSDB18, חותך קטעים ושומר בתיקיות processed.
    """
    # יצירת תיקיות פלט
    for folder in ['mix', 'vocals', 'drums', 'bass', 'other']:
        os.makedirs(os.path.join(output_base_path, folder), exist_ok=True)

    # מעבר על כל תיקיות השירים ב-Dataset
    # MUSDB18 בנוי כך: כל שיר הוא תיקייה ובתוכה הקבצים
    for root, dirs, files in os.walk(raw_data_path):
        if 'mixture.wav' in files:
            song_name = os.path.basename(root)
            print(f"Processing song: {song_name}")
            
            # רשימת הכלים שאנחנו רוצים לחלץ
            targets = ['mixture', 'vocals', 'drums', 'bass', 'other']
            
            # טעינת כל הקבצים של השיר הספציפי
            audio_data = {}
            for t in targets:
                path = os.path.join(root, f"{t}.wav")
                y, _ = librosa.load(path, sr=sr)
                audio_data[t] = y
            
            # חיתוך לקטעים
            samples_per_segment = segment_len * sr
            num_segments = len(audio_data['mixture']) // samples_per_segment
            
            for i in range(num_segments):
                start = i * samples_per_segment
                end = start + samples_per_segment
                
                # שמירת כל קטע בתיקייה המתאימה
                chunk_id = f"{song_name}_part{i}.wav"
                
                for t in targets:
                    target_folder = 'mix' if t == 'mixture' else t
                    out_path = os.path.join(output_base_path, target_folder, chunk_id)
                    sf.write(out_path, audio_data[t][start:end], sr)

if __name__ == "__main__":
    # נתיב המקור שבו נמצאות תיקיות train ו-test
    BASE_RAW_PATH = 'data/raw' 
    
    # נעבד רק את תיקיית ה-train לצורך אימון המודל
    # (את ה-test נשמור כקבצים שלמים להצגה ב-Streamlit אחר כך)
    TRAIN_RAW_PATH = os.path.join(BASE_RAW_PATH, 'train')
    PROCESSED_PATH = 'data/processed'
    
    if os.path.exists(TRAIN_RAW_PATH):
        slice_audio_files(TRAIN_RAW_PATH, PROCESSED_PATH)
        print("✅ Done Preprocessing Train Set!")
    else:
        print(f"❌ Error: Could not find train folder at {TRAIN_RAW_PATH}")