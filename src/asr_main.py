from asr_train_test import asr_train, asr_test
from data_loader import data_preprocessing
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchaudio
import os
import torch.utils.data as data_utils
import torch
from data_loader import load_data

__author__ = "Diep Luong"


def main():
    # Check on device
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    train_dataset = load_data("./train-clean-100")
    test_dataset = load_data("./test-clean")
    

    # Training paraneters
    epochs = 200
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