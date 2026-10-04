import torch
import torch.nn as nn
import torch.optim as optim

EMBEDDING_DIM = 784

"""
    models.py
    All models should receive the same type of tokenization, and the same embedding size at least once.
        - e.g MLP struggles on larger embedding sizes so its shortened to improve performance
    
    For MLP and CNN's inability to deal with irregular sequence lengths, their embeddings will be averaged down the columns.

"""


class MLP(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        """
        Multilayer Perceptron

        Args:
            vocab_size: The number of unique tokens.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.linear1 = nn.Linear(EMBEDDING_DIM, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        x = torch.mean(x, dim=1)  # Average down columns to get consistent input size.

        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        
        x = self.linear3(x)
        x = self.relu(x)
        
        x = self.linear4(x)
        x = self.sigmoid(x)

        return x

class RNN(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        """
        Recurrent Nerual Network

        Args:
            vocab_size: The number of unique tokens.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.rnn = nn.RNN(EMBEDDING_DIM, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)

        output, states = self.rnn(x)
        x = self.fc1(output)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

class LSTM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        """
        Long Short Term Memory Nerual Network

        Args:
            vocab_size: The number of unique tokens.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(EMBEDDING_DIM, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, vocab_size)

    def forward(self, x):
        output, states = self.lstm(x)
        x = self.fc1(output)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

class CNN(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        """
        Convolutional Nerual Network

        Args:
            vocab_size: The number of unique tokens.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(EMBEDDING_DIM, out_channels=1024, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool1d(kernel_size=2)
        )

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # New shape: (batch size, EMBEDDING_DIM, sequence_length)

        x = self.conv_layers(x)

        x = torch.mean(x, dim=2)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.sigmoid(x)

        return x

lstm = LSTM(30_000)
rnn = RNN(30_000)
mlp = MLP(30_000)
cnn = CNN(30_000)

models = [lstm, rnn, mlp, cnn]

for m in models:
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Total Parameters: {total_params:,}")