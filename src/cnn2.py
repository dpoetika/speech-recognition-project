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
        self.idx2char = {i: c for c, i in self.char2idx.items()}

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
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        transcript_encoded = self.encode_transcript(transcript)
        return waveform.squeeze(0), transcript_encoded


# ========== Collate Function ==========

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
        n_mels = 80
        self.spec = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=400)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(64 * (n_mels // 4), 512, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(512 * 2, vocab_size + 1)

    def forward(self, x):
        x = self.spec(x)  # (B, n_mels, T)
        x = x.unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.encoder(x)  # (B, 64, n_mels/4, T/4)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.log_softmax(dim=-1)
        return x


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


# ========== WER Calculation ==========

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

    train_dataset = CommonVoiceDataset(train_tsv, audio_folder)
    vocab_size = len(train_dataset.char2idx)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASRModel(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    num_epochs = 5
    loss_list = []
    wer_list = []
    confidence_list = []
    accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_wer = 0
        num_samples = 0
        total_confidence = 0
        total_accuracy = 0
        batch_count = 0

        for batch_idx, (waveforms, targets, input_lengths, target_lengths) in enumerate(train_loader):
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            outputs = model(waveforms)  # (B, T, C)
            outputs = outputs.permute(1, 0, 2)  # (T, B, C)

            output_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, targets, output_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # === Ek Metrikler ===
            probs = outputs.detach().exp()  # (T, B, C)
            batch_confidence = probs.max(dim=-1)[0].mean().item()
            total_confidence += batch_confidence

            # Character accuracy hesapla
            decoded_batch = outputs.argmax(dim=-1).permute(1, 0)  # (B, T)
            for i in range(decoded_batch.size(0)):
                decoded_seq = ctc_greedy_decoder(decoded_batch[i].cpu().numpy().tolist())
                pred_text = decode_indices(decoded_seq, train_dataset.idx2char)
                target_text = decode_indices(targets[i].cpu().numpy().tolist(), train_dataset.idx2char)

                # Karakter bazlı accuracy
                correct_chars = sum(p == t for p, t in zip(pred_text, target_text))
                char_acc = correct_chars / max(len(target_text), 1)
                total_accuracy += char_acc

            total_loss += loss.item()

            decoded_batch = torch.argmax(outputs, dim=2).transpose(0, 1)  # (B, T)
            for i in range(decoded_batch.size(0)):
                pred_indices = ctc_greedy_decoder(decoded_batch[i])
                pred_text = decode_indices(pred_indices, train_dataset.idx2char)

                target_indices = targets[i][:target_lengths[i]].cpu().numpy().tolist()
                target_text = decode_indices(target_indices, train_dataset.idx2char)

                total_wer += wer(target_text, pred_text)
                num_samples += 1

            batch_count += decoded_batch.size(0)

            if batch_idx % 1 == 0:

                print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Confidence: {batch_confidence:.3f}, WER: {total_wer / num_samples} ", end="")
                loss_list.append(loss.item())
                confidence_list.append(batch_confidence)

        avg_loss = total_loss / len(train_loader)
        avg_conf = total_confidence / len(train_loader)
        avg_acc = total_accuracy / batch_count
        accuracy_list.append(avg_acc)

        print(f"\nEpoch [{epoch+1}/{num_epochs}] Finished | Avg Loss: {avg_loss:.4f} | Avg Confidence: {avg_conf:.3f} | Char Accuracy: {avg_acc:.3f}")

    # === Kaydet ===
    torch.save(model.state_dict(), 'asr_model.pth')
    print("Model saved as 'asr_model.pth'.")

    # === Grafik ===
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_list)
    plt.title("Loss")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(confidence_list)
    plt.title("Confidence")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(accuracy_list)
    plt.title("Char Accuracy")
    plt.grid()

    plt.tight_layout()
    plt.show()


# ========== Evaluation ==========

def evaluate():
    test_tsv = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/test.tsv'
    audio_folder = 'C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/clips'

    test_dataset = CommonVoiceDataset(test_tsv, audio_folder)
    vocab_size = len(test_dataset.char2idx)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleASRModel(vocab_size).to(device)
    model.load_state_dict(torch.load('asr_model.pth'))
    model.eval()

    total_wer = 0.0

    with torch.no_grad():
        for waveforms, targets, input_lengths, target_lengths in test_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            outputs = model(waveforms)
            outputs = outputs.permute(1, 0, 2).squeeze(1)

            decoded_ids = ctc_greedy_decoder(outputs)
            predicted_text = decode_indices(decoded_ids, test_dataset.idx2char)

            target_indices = targets.squeeze(0).cpu().numpy().tolist()
            target_text = decode_indices(target_indices, test_dataset.idx2char)

            total_wer += wer(target_text, predicted_text)

    avg_wer = total_wer / len(test_loader)
    print(f"Average WER: {avg_wer:.4f}")


# ========== Run ==========

if __name__ == '__main__':
    train()
    evaluate()
