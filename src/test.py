import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import SpeechModel
from train import CHARS, CHAR_TO_IDX, SpeechDataset, collate_fn  # Eğitim kodundan import
import detector
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

def test_model(obj,model_path="./models/model.pth", test_data_dir="data/processed/test"):
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
                lwords = obj.list_words(pred_texts[i])
                dizi = obj.auto_correct(lwords)

                cumle = ""

                for kelime in dizi:
                    cumle += kelime + " "

                
                print(f"Gerçek: {label_texts[i]}")
                print(f"Tahmin: {pred_texts[i]}")
                print(f"Düzeltilmiş Metin: {cumle}")
                print("------")
    
    # Ortalama metrikleri yazdır
    avg_cer = total_cer / sample_count
    avg_wer = total_wer / sample_count
    print(f"\nTest Tamamlandı!")
    print(f"Ortalama CER: {avg_cer:.4f}")
    print(f"Ortalama WER: {avg_wer:.4f}")

if __name__ == "__main__":
    obj = detector.TurkishNLP()
    
    obj.create_word_set()
    test_model(obj)





