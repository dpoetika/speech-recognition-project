import torch
import torchaudio  # type: ignore
import json
from cnn3 import SimpleASRModel, ctc_greedy_decoder, decode_indices
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import os
import numpy as np

def record_audio(duration=30, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")

    audio = np.squeeze(audio)  # (samples,)
    
    # Geçici WAV dosyasına kaydet
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_wav.name, sample_rate, audio)
    return temp_wav.name

def run_inference(audio_path=None, checkpoint_path='best_asr_model.pth', vocab_path='vocab.json', use_microphone=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Vocab Yükle ===
    with open(vocab_path, 'r', encoding='utf-8') as f:
        char2idx = json.load(f)
    idx2char = {int(i): c for c, i in char2idx.items()}
    vocab_size = len(char2idx)

    # === Model Yükle ===
    model = SimpleASRModel(vocab_size)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # === Ses Al ===
    if use_microphone:
        audio_path = record_audio()
    
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.to(device)

    with torch.no_grad():
        output = model(waveform)  # (B=1, T, C)
        output = output.squeeze(0).cpu()  # (T, C)
        prediction = torch.argmax(output, dim=-1).numpy().tolist()
        decoded_seq = ctc_greedy_decoder(prediction)
        predicted_text = decode_indices(decoded_seq, idx2char)

    return predicted_text

if __name__ == "__main__":
    text = run_inference(use_microphone=True)
    print("Predicted Text:", text)
