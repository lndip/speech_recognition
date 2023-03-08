from asr_train_test import asr_train, asr_test
from data_loader import data_preprocessing
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchaudio
import os
import torch.utils.data as data_utils
import torch
from data_loader import load_data
from pathlib import Path
from torch.utils.data import random_split

__author__ = "Diep Luong"


def main():
    # Check on device
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    train_dataset = torchaudio.datasets.LIBRISPEECH(Path('speech_recognition', 'data', 'train'), url='train-clean-100', download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(Path('speech_recognition', 'data', 'test'), url="test-clean", download=True)
    train_size = int(len(train_dataset)*0.2)
    test_size = int(len(test_dataset)*0.2)
    [train_dataset, _] = random_split(train_dataset, [train_size, len(train_dataset)-train_size])
    [test_dataset, _] = random_split(test_dataset, [test_size, len(test_dataset)-test_size])

    # Training paraneters
    epochs = 100
    n_features = 128

    # Get dataloader 
    train_loader = data.DataLoader(
        dataset = train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda x: data_preprocessing(x, 'train')
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        collate_fn=lambda x: data_preprocessing(x, 'test')
    )

    # Training and testing
    asr_train(n_features, epochs, device, train_loader)

    asr_test(n_features, device, test_loader)


if __name__ == "__main__":
    main()