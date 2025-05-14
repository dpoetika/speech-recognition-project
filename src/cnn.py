import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio  # type: ignore
import time
from matplotlib import pyplot as plt
# ========== Dataset ==========

class CommonVoiceDataset(Dataset):
    def __init__(self, tsv_path, audio_dir):
        self.data = pd.read_csv(tsv_path, sep='\t')
        self.audio_dir = audio_dir
        self.char2idx = self._build_vocab()

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
        if "C:/Users" in row['path']:
            mp3_path = row['path']
        else:
            mp3_path = os.path.join(self.audio_dir, row['path'])
        transcript = row['sentence']

        waveform, sample_rate = torchaudio.load(mp3_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        transcript_encoded = self.encode_transcript(transcript)

        return waveform.squeeze(0), transcript_encoded  # (waveform, transcript)

# ========== Collate Function ==========

def collate_fn(batch):
    waveforms, transcripts = zip(*batch)

    # Pad audio
    max_len = max([w.shape[0] for w in waveforms])
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w

    # Pad transcripts
    max_tlen = max([t.shape[0] for t in transcripts])
    padded_targets = torch.full((len(transcripts), max_tlen), fill_value=0, dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in transcripts], dtype=torch.long)
    for i, t in enumerate(transcripts):
        padded_targets[i, :t.shape[0]] = t

    input_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)

    return padded_waveforms, padded_targets, input_lengths, target_lengths  # (B, T)

# ========== Model ==========

class SimpleASRModel(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleASRModel, self).__init__()
        n_mels = 80
        self.spec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=400)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # → (n_mels / 2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (n_mels / 4)
            nn.ReLU(),
        )

        self.rnn = nn.GRU(64 * (n_mels // 4), 512, batch_first=True, bidirectional=True)

        self.classifier = nn.Linear(512 * 2, vocab_size + 1)  # +1 for blank

    def forward(self, x):
        x = self.spec(x)  # (B, n_mels, T)
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.encoder(x)  # (B, 64, n_mels/4, T/4)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T/4, 64 * 32)
        x, _ = self.rnn(x)  # (B, T/4, 1024)
        x = self.classifier(x)  # (B, T/4, vocab+1)
        x = x.log_softmax(dim=-1)
        return x


# ========== Training ==========

def train():
    # Paths
    print("1")
    train_tsv = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/train.tsv'
    audio_folder = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/clips'

    # Dataset and Loader
    train_dataset = CommonVoiceDataset(train_tsv, audio_folder)
    vocab_size = len(train_dataset.char2idx)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Model and Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASRModel(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    print("1")
    # Training loop
    num_epochs = 5
    x = []
    y = []
    Y = 0
    toplamsure = 0

    for epoch in range(num_epochs):
        print("1")
        model.train()
        total_loss = 0
        for batch_idx,(waveforms, targets, input_lengths, target_lengths) in enumerate(train_loader):
            baslangic = time.time()
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            outputs = model(waveforms)  # (B, T, C)
            outputs = outputs.permute(1, 0, 2)  # (T, B, C) for CTC

            # CTC requires time lengths of output and target
            adjusted_input_lengths = torch.full(size=(waveforms.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            optimizer.zero_grad()
            loss = criterion(outputs, targets, adjusted_input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            toplamsure += time.time() - baslangic
            bitis = toplamsure/(batch_idx +1)
            tahmin = bitis * (len(train_loader)-batch_idx-1) * (num_epochs - epoch - 1) / 3600
            
            if batch_idx % 100 == 0:

                x.append(loss.item())
                Y += 1
                y.append(Y)
            print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f} Geçen süre : {bitis:.2f} Tahmini Bitiş : {tahmin:.2f} ", end="")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'asr_model.pth')
    print("Training completed and model saved as 'asr_model.pth'.")
    plt.figure(figsize = (10,5))
    plt.plot(range(1,Y+1), x,label = "Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")  
    plt.legend()
    plt.grid()
    plt.show()


# ========== Run ==========


def evaluate():
    # Paths
    test_tsv = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/test.tsv'
    audio_folder = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/clips'

    # Dataset and Loader
    test_dataset = CommonVoiceDataset(test_tsv, audio_folder)
    vocab_size = len(test_dataset.char2idx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASRModel(vocab_size).to(device)
    model.load_state_dict(torch.load('C:/Users/nebul/Desktop/python/audio/speech-recognition-project/models/asr_model.pth'))
    model.eval()

    total_correct = 0
    total_words = 0
    total_wer = 0

    with torch.no_grad():
        for waveforms, targets, input_lengths, target_lengths in test_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            outputs = model(waveforms)  # (B, T, C)
            outputs = outputs.permute(1, 0, 2)  # (T, B, C) for CTC

            # Decode the output using argmax (CTC decoding)
            _, predicted_ids = outputs.max(dim=-1)

            # Calculate Word Error Rate (WER)
            predicted_transcript = ''.join([test_dataset.char2idx[c] for c in predicted_ids.cpu().numpy()])
            target_transcript = ''.join([test_dataset.char2idx[c] for c in targets.cpu().numpy()])

            # Calculate WER here (this is a placeholder for your WER calculation function)
            total_wer += wer(predicted_transcript, target_transcript)

            total_correct += (predicted_ids == targets).sum().item()
            total_words += len(targets)

    accuracy = total_correct / total_words
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
    print(f"Word Error Rate: {total_wer / len(test_loader):.4f}")

# Helper function for Word Error Rate (WER)
def wer(pred, target):
    # Calculate WER between predicted and target transcription
    pred_words = pred.split()
    target_words = target.split()
    # WER calculation logic here (edit distance)
    # Placeholder implementation
    return 0.0


if __name__ == '__main__':
    train()
    evaluate()
