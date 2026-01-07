import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, mix_dir, target_dir, sr=22050, n_fft=2048, hop_length=512):
        self.mix_dir = mix_dir
        self.target_dir = target_dir
        self.files = [f for f in os.listdir(mix_dir) if f.endswith('.wav')]
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mix_path = os.path.join(self.mix_dir, self.files[idx])
        target_path = os.path.join(self.target_dir, self.files[idx])

        y_mix, _ = librosa.load(mix_path, sr=self.sr)
        y_target, _ = librosa.load(target_path, sr=self.sr)

        # STFT
        S_mix = np.abs(librosa.stft(y_mix, n_fft=self.n_fft, hop_length=self.hop_length))
        S_target = np.abs(librosa.stft(y_target, n_fft=self.n_fft, hop_length=self.hop_length))

        # נירמול בטוח (טווח 0 עד 1)
        def safe_norm(S):
            S_max = np.max(S)
            if S_max <= 0: return S
            return S / (S_max + 1e-8)

        mix_norm = safe_norm(S_mix)
        target_norm = safe_norm(S_target)

        # המרה ל-Tensor
        return torch.FloatTensor(mix_norm).unsqueeze(0), torch.FloatTensor(target_norm).unsqueeze(0)