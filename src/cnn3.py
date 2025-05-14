import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio  # type: ignore
import time
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Learning Rate Scheduler
import random  # Data Augmentation için

# ========== Dataset ==========

class CommonVoiceDataset(Dataset):
    def __init__(self, tsv_path, audio_dir, augment=False):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.audio_dir = audio_dir
        self.char2idx = self._build_vocab()
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.augment = augment  # Data Augmentation için parametre

    def _build_vocab(self):
        chars = set()
        for s in self.data['sentence']:
            chars.update(s.lower())
        char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0: blank for CTC
        return char2idx

    def encode_transcript(self, transcript):
        return torch.tensor([self.char2idx[c] for c in transcript.lower() if c in self.char2idx], dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mp3_path = row['path'] if "C:/" in row['path'] else os.path.join(self.audio_dir, row['path'])
        transcript = row['sentence']

        waveform, sample_rate = torchaudio.load(mp3_path)
        
        # Data Augmentation: Random noise ekle
        if self.augment:
            noise = torch.randn_like(waveform) * 0.05  # Gürültü ekleme
            waveform += noise

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        transcript_encoded = self.encode_transcript(transcript)
        return waveform.squeeze(0), transcript_encoded
    


def collate_fn(batch):
    waveforms, transcripts = zip(*batch)

    max_len = max([w.shape[0] for w in waveforms])
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w

    max_tlen = max([t.shape[0] for t in transcripts])
    padded_targets = torch.full((len(transcripts), max_tlen), fill_value=0, dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    for i, t in enumerate(transcripts):
        padded_targets[i, :t.shape[0]] = t

    input_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    return padded_waveforms, padded_targets, input_lengths, target_lengths


# ========== Model ==========
class SimpleASRModel(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleASRModel, self).__init__()
        n_mels = 64  # 80'den 64'e düşürdük
        
        self.spec = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels, 
            n_fft=256,  # 400'den 256'ya düşürdük
            hop_length=128
        )

        # CNN Katmanları
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

        # GRU Katmanı - input_size düzeltildi
        self.rnn = nn.GRU(
            input_size=32 * (n_mels // 4),  # 64*(80//4)=1280 yerine 32*(64//4)=512
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=True
        )

        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512*2 yerine 512 (hidden_size=256 ve bidirectional)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, vocab_size + 1)
        )

    def forward(self, x):
        # Özellik çıkarımı
        x = self.spec(x)  # (B, 64, T)
        x = x.unsqueeze(1)  # (B, 1, 64, T)
        
        # CNN
        x = self.encoder(x)  # (B, 32, 16, T//4)
        b, c, f, t = x.size()
        
        # RNN için reshape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T//4, 32, 16)
        x = x.view(b, t, -1)  # (B, T//4, 512)
        
        # RNN
        x, _ = self.rnn(x)
        
        # Sınıflandırma
        x = self.classifier(x)
        return x.log_softmax(dim=-1)

# ========== CTC Decoder ==========
def ctc_greedy_decoder(prediction, blank=0):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy().tolist()
    decoded = []
    prev = blank
    for p in prediction:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded



def decode_indices(indices, idx2char):
    return ''.join([idx2char.get(i, '') for i in indices])



def wer(reference, hypothesis):
    ref = reference.split()
    hyp = hypothesis.split()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)

    return d[-1][-1] / len(ref) if ref else 1.0



# ========== Training ==========

def train():

    train_tsv = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/train.tsv'
    audio_folder = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/clips'

    train_dataset = CommonVoiceDataset(train_tsv, audio_folder, augment=True)  # Data Augmentation
    vocab_size = len(train_dataset.char2idx)

    train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=3,  # CPU çekirdek sayısı kadar
    pin_memory=False,  # CPU'da True yapma!
    persistent_workers=True,
    prefetch_factor=2,  # Veriyi avans çek
    collate_fn=collate_fn
)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASRModel(vocab_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # Learning rate scheduler
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    num_epochs = 5
    loss_list = []
    accuracy_list = []

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        num_samples = 0

        for batch_idx, (waveforms, targets, input_lengths, target_lengths) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            targets = targets.to(device)
            
            outputs = model(waveforms)
            
            outputs = outputs.permute(1, 0, 2)  # (T, B, C)

            output_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, targets, output_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            decoded_batch = torch.argmax(outputs, dim=2).transpose(0, 1)  # (B, T)
            for i in range(decoded_batch.size(0)):
                decoded_seq = ctc_greedy_decoder(decoded_batch[i].cpu().numpy().tolist())
                pred_text = decode_indices(decoded_seq, train_dataset.idx2char)
                target_text = decode_indices(targets[i].cpu().numpy().tolist(), train_dataset.idx2char)

                correct_chars = sum(p == t for p, t in zip(pred_text, target_text))
                char_acc = correct_chars / max(len(target_text), 1)
                total_accuracy += char_acc
                num_samples += 1

            print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}", end="")

        # Learning rate scheduler
        scheduler.step(total_loss / len(train_loader))

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_accuracy / num_samples
        loss_list.append(avg_loss)
        accuracy_list.append(avg_acc)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Finished | Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_acc:.3f}")

        # Model checkpointing
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            torch.save(model.state_dict(), 'best_asr_model.pth')
            print("Saved best model.")

    # === Grafik ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title("Loss")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_list)
    plt.title("Char Accuracy")
    plt.grid()

    plt.tight_layout()
    plt.show()

# ========== Run ==========

if __name__ == '__main__':
    train()
