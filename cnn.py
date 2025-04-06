import torch.nn as nn
import torch.nn.functional as F
import torch

class ConfidenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5) 
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 61, 64) #32 channels × 61 time steps → result of conv/pool layers
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x): #Shape of x: [batch_size, channels, time], e.g., [32, 4, 256]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Confidence score
        return x

model = ConfidenceCNN()

# Example usage
eeg_window = torch.randn(4, 1, 64)  # 4 channels, 1 time step, 64 features
model(eeg_window)