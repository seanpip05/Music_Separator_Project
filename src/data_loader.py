import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import os

class MusicDataset(Dataset):
    def __init__(self, mix_dir, target_dir, sr=22050):
        self.mix_dir = mix_dir
        self.target_dir = target_dir
        self.sr = sr
        self.files = [f for f in os.listdir(mix_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.files)

    def _get_spec(self, path):
        y, _ = librosa.load(path, sr=self.sr)
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        # נירמול פשוט לטווח 0-1 (עוזר למודל)
        S_norm = (S_db + 80) / 80
        return torch.tensor(S_norm).unsqueeze(0).float()

    def __getitem__(self, idx):
        mix_path = os.path.join(self.mix_dir, self.files[idx])
        target_path = os.path.join(self.target_dir, self.files[idx])
        
        x = self._get_spec(mix_path)
        y = self._get_spec(target_path)
        
        return x, y