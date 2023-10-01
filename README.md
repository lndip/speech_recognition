# Introduction
This project focuses on automatic speech recognition task, specifically speech transcription, using Deep Neural Network (DNN) architecture. The model was trained and test on 10% of `train-clean-100` and `test-clean` from Librispeech.

# Model architecture
This project employs CRNN structure with convolutional and GRU blocks to process the input spectrogram. The model output the prediction probabilities of the letters over the time steps.
![image](https://github.com/lndip/speech_recognition/assets/65665546/20cdbf3b-9b81-4d59-89ba-cf268b40cd5a)

# Installation
To run the code, you need `python`, `pytorch`, and `numpy`

# How to run
`asr_main.py` incorperates the training loop and the testing stage of the speech transcription model

# Authors
<ul>
  <li>Diep Luong
  <li>Fareeda Mohammad


