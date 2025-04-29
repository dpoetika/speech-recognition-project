import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import SpeechModel
from train import CHARS, CHAR_TO_IDX, SpeechDataset, collate_fn  # Eğitim kodundan import

import os
import librosa # type: ignore
import numpy as np
import pandas as pd
def decode_predictions(log_probs, chars=CHARS, blank_idx=CHAR_TO_IDX["<blank>"]):
    
    _, max_indices = torch.max(log_probs, dim=2)  # (seq_len, batch) -> (batch, seq_len)
    max_indices = max_indices.permute(1, 0)
    
    decoded_texts = []
    for batch in max_indices:
        # Tekrarlanan karakterleri ve <blank>'leri kaldır
        previous_char = None
        current_text = []
        for idx in batch:
            idx = idx.item()
            if idx != blank_idx and idx != previous_char:
                current_text.append(chars[idx])
            previous_char = idx
        decoded_texts.append("".join(current_text))
    return decoded_texts

def test_model(model_path="./models/model.pth", test_data_dir="example/processed/train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modeli yükle
    model = SpeechModel(num_features=80, num_classes=len(CHARS))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Test veri seti
    test_dataset = SpeechDataset(test_data_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Metrikler ve kayıtlar
    total_cer = 0.0
    total_wer = 0.0
    sample_count = 0
    
    with torch.no_grad():
        for features, labels, feature_lengths, label_lengths in test_loader:
            # Girişi hazırla
            features = features.unsqueeze(2).to(device)  # (batch, max_time, 1, 80)
            features = features.permute(1, 0, 2, 3)      # (max_time, batch, 1, 80)
            
            # Forward pass
            outputs = model(features)
            outputs = outputs.permute(1, 0, 2)  # (seq_len, batch, num_classes)
            log_probs = outputs.log_softmax(2)
            
            # Tahminleri decode et
            pred_texts = decode_predictions(log_probs)
            
            # Gerçek transkripsiyonları al
            label_texts = []
            for label in labels:
                chars = [CHARS[idx] for idx in label if idx != CHAR_TO_IDX["<blank>"]]
                label_texts.append("".join(chars))
            
            # CER ve WER hesapla (basit bir implementasyon)
            for pred, true in zip(pred_texts, label_texts):
                # CER (Karakter Hata Oranı)
                cer = sum(c1 != c2 for c1, c2 in zip(pred, true)) / max(len(true), 1)
                total_cer += cer
                
                # WER (Kelime Hata Oranı)
                pred_words = pred.split()
                true_words = true.split()
                wer = sum(w1 != w2 for w1, w2 in zip(pred_words, true_words)) / max(len(true_words), 1)
                total_wer += wer
                
                sample_count += 1
            
            # Örnek çıktıları göster
            print("\nÖrnek Tahminler:")
            for i in range(min(3, len(pred_texts))):

                print(f"Gerçek: {label_texts[i]}")
                print(f"Tahmin: {pred_texts[i]}")
                print("------")
    
    # Ortalama metrikleri yazdır
    avg_cer = total_cer / sample_count
    avg_wer = total_wer / sample_count
    print(f"\nTest Tamamlandı!")
    print(f"Ortalama CER: {avg_cer:.4f}")
    print(f"Ortalama WER: {avg_wer:.4f}")


def preprocess_audio(audio_path, sample_rate=16000, n_mels=80):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel_spect = librosa.power_to_db(mel_spect)
    return log_mel_spect

def prepare_dataset(data_dir, output_dir, split='train'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    audio_path = os.path.join(data_dir, "Kayıt.mp3")
    
    features = preprocess_audio(audio_path)
    np.save(os.path.join(output_dir, f'0_features.npy'), features)
    with open(os.path.join(output_dir, f'0_transcription.txt'), 'w', encoding='utf-8') as f:
        f.write("bat kb görlir")


if __name__ == "__main__":

    data_dir = 'example/raw'
    
    for split in ['train']:
        output_dir = f'example/processed/train'
        # Eski dosyaları temizle
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))
        prepare_dataset(data_dir, output_dir, split)
    
    test_model()





