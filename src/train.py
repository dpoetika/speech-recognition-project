import os
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from model import SpeechModel

CHARS = list("abcçdefgğhıijklmnoöprsştuüvyz ") + ["<blank>"]
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}

class SpeechDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith("_features.npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.data_dir, f"{idx}_features.npy")
        transcription_path = os.path.join(self.data_dir, f"{idx}_transcription.txt")

        features = np.load(feature_path)
        features = torch.FloatTensor(features).transpose(0, 1)  # (zaman, 80)

        with open(transcription_path, "r", encoding="utf-8") as f:
            transcription = f.read().strip()
        labels = [CHAR_TO_IDX[char] for char in transcription if char in CHAR_TO_IDX]

        return features, torch.LongTensor(labels)

def collate_fn(batch):
    features, labels = zip(*batch)
    
    # Özellikleri (zaman, 80) formatında sabitle
    features = [f for f in features]
    labels = [l for l in labels]

    feature_lengths = torch.LongTensor([len(f) for f in features])
    label_lengths = torch.LongTensor([len(l) for l in labels])

    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=CHAR_TO_IDX["<blank>"])

    return features_padded, labels_padded, feature_lengths, label_lengths

def train_model():
    num_epochs = 100
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = []
    y = []
    Y = 0
    train_dataset = SpeechDataset("data/processed/train")
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)


    model = SpeechModel(num_features=80, num_classes=len(CHARS)).to(device)
    criterion = nn.CTCLoss(blank=CHAR_TO_IDX["<blank>"], zero_infinity=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        for batch_idx, (features, labels, feature_lengths, label_lengths) in enumerate(train_loader):
            # Girişi modele uygun hale getir
            features = features.unsqueeze(2).to(device)  # (batch, max_time, 1, 80)
            features = features.permute(1, 0, 2, 3)      # (max_time, batch, 1, 80)
            labels = labels.to(device)

            feature_lengths = feature_lengths.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()



            outputs = model(features)  # Çıkış: [batch, seq_len, num_classes]
            outputs = outputs.permute(1, 0, 2)  # [seq_len, batch, num_classes] ✅ CTC için doğru format
            log_probs = outputs.log_softmax(2)
            
            # CNN zaman boyutunu 4 kat azalttığı için uzunlukları güncelle
            adjusted_feature_lengths = (feature_lengths // 4).clamp(min=1)
            
            loss = criterion(log_probs, labels, adjusted_feature_lengths, label_lengths)
            loss.backward()
            optimizer.step()


            total_loss += loss.item()
            if (batch_idx + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                x.append(loss.item())
                Y += 1
                y.append(Y)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] tamamlandı, Ortalama Kayıp: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/modeliki.pth")
    print("Model kaydedildi: models/modeliki.pth")
    plt.figure(figsize = (10,5))
    plt.plot(range(1,Y+1), x,label = "Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")  
    plt.legend()
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    train_model()
