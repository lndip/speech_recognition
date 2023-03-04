from asr_train_test import asr_train, asr_test
# from getting_and_init_the_data import get_dataloder

import torch

def main():
    # Check on device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Training paraneters
    epochs = 200
    n_features = None

    # Get dataloader
    train_dataloader = None
    test_dataloader = None

    asr_train(n_features, epochs, device, train_dataloader)

    asr_test(n_features, device, test_dataloader)


if __name__ == "__main__":
    main()