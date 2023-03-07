from asr_train_test import asr_train, asr_test
# from getting_and_init_the_data import get_dataloder

import torch

__author__ = "Diep Luong"

def main():
    # Check on device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Training paraneters
    epochs = 200
    n_features = None # the spectrogram will have the shape of (features x n_frames), n_features = features

    # Get dataloader
    train_dataloader = None
    test_dataloader = None # for test_dataloader, batch_size = len(test_dataset)

    asr_train(n_features, epochs, device, train_dataloader)

    asr_test(n_features, device, test_dataloader)


if __name__ == "__main__":
    main()