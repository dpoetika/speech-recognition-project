"""import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, num_features=80, num_classes=30):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
        
    
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.LSTM(64 * num_features, 256, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(3, 0, 1, 2).reshape(x.size(3), x.size(0), -1)
      
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x"
"""
import torch
import torch.nn as nn

class SpeechModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Dinamik RNN input boyutu hesapla
        self._init_rnn(num_features)
        
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(256, num_classes)  # 128*2 (bidirectional)

    def _init_rnn(self, num_features):
        # CNN çıkış boyutunu hesapla
        dummy_input = torch.randn(1, 1, num_features, 80)  # (batch, channel, height, width)
        with torch.no_grad():
            dummy_output = self.cnn(dummy_input)
            self.rnn_input_size = dummy_output.size(1) * dummy_output.size(2)  # (kanal * yükseklik)

    def forward(self, x):
        # Boyutları düzenle: (Zaman, Batch, Kanal, Özellik) -> (Batch, Kanal, Zaman, Özellik)
        x = x.permute(1, 2, 0, 3)  # [287, 16, 1, 80] -> [16, 1, 287, 80]
        x = self.cnn(x)  # Örnek çıkış: [16, 64, 71, 20]
        
        # RNN için boyutları düzenle
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # [16, 64, 71, 20] -> [16, 71, 20, 64]
        x = x.reshape(batch_size, -1, self.rnn_input_size)  # [16, 71*20=1420, 64*20=1280]
        
        # RNN
        x, _ = self.rnn(x)  # Çıkış: [16, 1420, 256]
        x = self.fc(x)       # Çıkış: [16, 1420, num_classes]
        return x