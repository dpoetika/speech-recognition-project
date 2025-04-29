# Speech Recognition Project

Bu proje, Türkçe dilinde konuşma tanıma (ASR - Automatic Speech Recognition) modelini sıfırdan eğitmek amacıyla geliştirilmiştir. Model, ses verilerini giriş olarak alır ve transkripsiyon (yazıya döküm) çıktısı üretir. Eğitim sırasında CTC kaybı ve `Adam` optimizasyonu kullanılmaktadır.

## Özellikler

- Türkçe karakter seti (29 harf + boşluk + `<blank>` token)
- CTC tabanlı model eğitimi
- `Adam` optimizasyonu
- PyTorch ile model eğitimi
- CNN + RNN mimarisi desteklenebilir
- `matplotlib` ile loss görselleştirmesi

## Proje Yapısı

```
speech-recognition-project/
│
├── data/
│   └── processed/
│       └── train/             # .npy ve .txt veri dosyaları (örnek: 0_features.npy, 0_transcription.txt)
│
├── models/                    # Eğitilen modellerin kaydedildiği klasör
│
├── src/
│   ├── train.py               # Modeli eğitmek için ana dosya
│   ├── model.py               # SpeechModel tanımı (örneğin CNN+RNN mimarisi)
│   └── ...                    # Diğer yardımcı modüller
│
├── requirements.txt           # Gerekli Python kütüphaneleri
├── README.md                  # Bu dosya
└── .gitignore                 # Git'e dahil edilmeyecek dosyalar
```

## Gereksinimler

Aşağıdaki paketleri yüklemek için:

```bash
pip install -r requirements.txt
```


## Veri Formatı

Eğitim verileri `data/raw/validated.tsv` ve `data/raw/clips/seskaydi.mp3`  şeklinde olmalıdır.

```bash
python src/split.py
```

```bash
python src/preprocess.py
```

Eğitim ve test verisini ayırıp önişlemeden geçirdikten sonra `data/processed/train/` klasöründe aşağıdaki formatta olmalıdır:

- `0_features.npy`: Sesin özellik temsili (örn. Mel spectrogram)
- `0_transcription.txt`: Sesin yazıya dökülmüş hali

Her ses örneği için eşleşen bir `.npy` ve `.txt` dosyası bulunmalıdır.

## Eğitim

Aşağıdaki komut ile modeli eğitebilirsiniz:

```bash
python src/train.py
```

Eğitim sonunda `models/model.pth` dosyasına model kaydedilir ve loss grafiği çizilir.

## Lisans

MIT Lisansı