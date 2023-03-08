import torch
import numpy as np
from torch.nn import CTCLoss, Module
from torchmetrics import WordErrorRate, CharErrorRate
from torch.utils.data import DataLoader
from decoder import GreedyDecoder

__author__ = "Diep Luong"

def forward_backward_pass(dataloader:DataLoader,
                          asr_model:Module,
                          optimizer:torch.optim.Optimizer,
                          device:str):
    """Forward backward pass for training and testing

    :param dataloader: Dataloader
    :type dataloader: DataLoader
    :param asr_model: ASR model
    :type asr_model: Module
    :param optimizer: Model optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: Device (cpu|cuda)
    :type device: str
    :return: ASR model, mean loss, wer, cer 
    :rtype: Tuple
    """

    if optimizer is not None:
        asr_model.train()
    else:
        asr_model.eval()

    loss_function = CTCLoss(blank=28)
    iterartion_losses = []

    # Evaluation metrics
    wer = WordErrorRate()
    cer = CharErrorRate()

    for batch in dataloader:
        # Zero the gradients
        if optimizer is not None:
            optimizer.zero_grad()

        # Get the batches    
        spectrograms, labels, input_lengths, label_lengths = batch
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        predictions = asr_model(spectrograms) # (batch, time, n_classes)
        predictions = torch.nn.functional.log_softmax(predictions, dim=-1)

        loss = loss_function(predictions.transpose(0,1), labels, input_lengths, label_lengths) # transpose to match the required dimension of CTC
        
        # Back propagation
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        # Log the loss
        iterartion_losses.append(loss.item())

        # Evaluate by using word error rate and char error rate
        # The decoder return sequences of text in form ["text1", "text2",...]
        decoded_preds, decoded_labels = GreedyDecoder(predictions, labels, label_lengths)
        pred_wer = wer(decoded_preds, decoded_labels)
        pred_cer = cer(decoded_preds, decoded_labels)

    return asr_model, \
           np.mean(iterartion_losses), \
           pred_wer, \
           pred_cer

