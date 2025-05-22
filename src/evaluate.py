import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio  # type: ignore
import time
from matplotlib import pyplot as plt
from cnn3 import CommonVoiceDataset, collate_fn,SimpleASRModel,ctc_greedy_decoder,decode_indices,wer

def evaluate(model_path, tsv_path, audio_dir):

    import json
    with open('vocab.json', 'r', encoding='utf-8') as f:
        char2idx = json.load(f)

    # Test dataset'i bu vocab ile oluştur
    eval_dataset = CommonVoiceDataset(tsv_path, audio_dir, augment=False)
    eval_dataset.char2idx = {k: int(v) for k, v in char2idx.items()}
    eval_dataset.idx2char = {i: c for c, i in eval_dataset.char2idx.items()}




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Dataloader
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=16,
        num_workers=2,
        collate_fn=collate_fn
    )

    vocab_size = len(eval_dataset.char2idx)
    print(vocab_size)
    model = SimpleASRModel(vocab_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    total_loss = 0
    total_accuracy = 0
    total_wer = 0
    num_samples = 0

    with torch.no_grad():
        for waveforms, targets, input_lengths, target_lengths in eval_loader:
            waveforms = waveforms.to(device)
            targets = targets.to(device)

            outputs = model(waveforms)
            outputs = outputs.permute(1, 0, 2)  # (T, B, C)

            output_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, targets, output_lengths, target_lengths)
            total_loss += loss.item()

            decoded_batch = torch.argmax(outputs, dim=2).transpose(0, 1)  # (B, T)

            for i in range(decoded_batch.size(0)):
                decoded_seq = ctc_greedy_decoder(decoded_batch[i].cpu().numpy().tolist())
                pred_text = decode_indices(decoded_seq, eval_dataset.idx2char)
                target_text = decode_indices(targets[i].cpu().numpy().tolist(), eval_dataset.idx2char)
                print(pred_text)
                print(target_text)
                print("--")
                correct_chars = sum(p == t for p, t in zip(pred_text, target_text))
                char_acc = correct_chars / max(len(target_text), 1)
                total_accuracy += char_acc

                total_wer += wer(target_text, pred_text)
                num_samples += 1

    avg_loss = total_loss / len(eval_loader)
    avg_accuracy = total_accuracy / num_samples
    avg_wer = total_wer / num_samples

    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average Char Accuracy: {avg_accuracy:.3f}")
    print(f"Average WER: {avg_wer:.3f}")

    return avg_loss, avg_accuracy, avg_wer



if __name__ == '__main__':
    evaluate(
        model_path='C:/Users/nebul/Desktop/python/audio/speech-recognition-project/checkpoint.pth',
        tsv_path='C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/train.tsv',
        audio_dir='C:/Users/nebul/Desktop/python/audio/speech-recognition-project/data/raw/clips'
    )
