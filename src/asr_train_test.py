from forward_backward_pass import forward_backward_pass
from speech_recognition_model import SpeechRecognitionModel

import torch
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader

__author__ = "Diep Luong"

def asr_train(n_features:int,
              epochs:int,
              device:str,
              dataloader:DataLoader):
    """ASR training loop

    :param n_features: Number of features input to CNN
    :type n_features: int
    :param epochs: Number of epochs for training
    :type epochs: int
    :param device: Device (cpu|cuda)
    :type device: str
    :param dataloader: Dataloder
    :type dataloader: DataLoader
    """
    asr_model = SpeechRecognitionModel(rnn_dim=512,
                                       n_classes=29, # 29 classes including the blank
                                       n_cnn_features=n_features)

    optimizer = Adam(params=asr_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        asr_model, train_loss, _, _ = forward_backward_pass(dataloader,
                                                    asr_model=asr_model,
                                                    optimizer=optimizer,
                                                    device=device)

        print(f'Epoch: {epoch:03d} | '
              f'Mean loss:{train_loss:7.4f}')

    torch.save(asr_model.state_dict(), Path('speech_recognition', 'src', 'asr_state_dict.pt'))


def asr_test(n_features:int,
             device:str,
             dataloader:DataLoader):
    asr_model = SpeechRecognitionModel(rnn_dim=128,
                                    n_classes=29,
                                    n_cnn_features=n_features)

    asr_model.load_state_dict(torch.load(Path('speech_recognition', 'src', 'asr_state_dict.pt'), map_location=device))

    asr_model, test_loss, pred_wer, pred_cer = forward_backward_pass(dataloader,
                                            asr_model=asr_model,
                                            optimizer=None,
                                            device=device)

    print(f'Mean loss:{test_loss:7.4f} | WER: {pred_wer:7.4f} | CER: {pred_cer:7.4f}')
