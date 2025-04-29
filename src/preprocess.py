import os
import librosa # type: ignore
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

def preprocess_audio(audio_path, sample_rate=16000, n_mels=80):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel_spect = librosa.power_to_db(mel_spect)
    return log_mel_spect

def prepare_dataset(data_dir, output_dir, split='train'):
    df = pd.read_csv(os.path.join(data_dir, f'{split}.tsv'), sep='\t')
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(data_dir, 'clips', row['path'])
        transcription = row['sentence'].lower()
        features = preprocess_audio(audio_path)
        np.save(os.path.join(output_dir, f'{idx}_features.npy'), features)
        with open(os.path.join(output_dir, f'{idx}_transcription.txt'), 'w', encoding='utf-8') as f:
            f.write(transcription)

    

if __name__ == "__main__":
    data_dir = 'data/raw'
    
    for split in ['train', 'dev', 'test']:
        output_dir = f'data/processed/{split}'
        # Eski dosyaları temizle
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
        prepare_dataset(data_dir, output_dir, split)