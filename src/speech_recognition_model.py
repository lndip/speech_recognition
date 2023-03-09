import torch
from torch.nn import Module, Conv2d, GRU, Dropout, BatchNorm2d, ReLU, Sequential, Linear
from pytorch_model_summary import summary 

__author__ = "Diep Luong"

class ResidualCNNBlock(Module):
    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                kernel:int, 
                stride:int, 
                dropout:float):
        """Residual block for ASR system

        :param in_channels: number of input channels 
        :type in_channels: int
        :param out_channels: number of output channels 
        :type out_channels: int
        :param kernel: kernel size
        :type kernel: int
        :param stride: stride
        :type stride: int
        :param dropout: dropout rate
        :type dropout: float
        """
        super().__init__()

        self.batchnorm1 = BatchNorm2d(num_features=out_channels)
        self.dropout1 = Dropout(dropout)
        self.cnn1 = Conv2d(in_channels, 
                           out_channels,
                           kernel,
                           stride,
                           padding=kernel//2)
        self.batchnorm2 = BatchNorm2d(num_features=out_channels)
        self.dropout2 = Dropout(dropout)
        self.cnn2 = Conv2d(out_channels,
                           out_channels,    
                           kernel,
                           stride,
                           padding=kernel//2)

        self.relu = ReLU()

    def forward(self, X):
        residual = X  # (batch, channel, feature, time)
        X = self.batchnorm1(X)
        X = self.relu(X)
        X = self.dropout1(X)
        X = self.cnn1(X)

        X = self.batchnorm2(X)
        X = self.relu(X)
        X = self.dropout2(X)
        X = self.cnn2(X)
        X += residual
        return X # (batch, channel, feature, time)


class GRUBlock(Module):
    def __init__(self, 
                in_features:int,
                hidden_size:int,
                dropout:float) -> None:
        """Bidirectional GRU block for ASR system

        :param in_features: number of input features 
        :type in_features: int
        :param hidden_size: number of hidden features  
        :type hidden_size: int
        """
        super().__init__()

        self.bigru = GRU(input_size=in_features,
                        hidden_size=hidden_size,
                        batch_first=True,
                        bidirectional=True)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self,X):
        X = self.relu(X)
        X, _ = self.bigru(X)
        X = self.dropout(X)
        return X


class SpeechRecognitionModel(Module):
    def __init__(self,
                rnn_dim:int,
                n_classes:int,
                n_cnn_features:int,
                kernel=3,
                stride=2,
                dropout=0.1) -> None:
        """Speech Recognition system based on CNN-RNN model

        :param rnn_dim: RNN dimision
        :type rnn_dim: int
        :param n_classes: Number of classes (28 character symbols and 1 blank symbol)
        :type n_classes: int
        :param n_cnn_features: Number of input CNN features (CNN block has size of batch x channels x features x time)
        :type n_cnn_features: int
        :param kernel: Kernel size, defaults to 3
        :type kernel: int, optional
        :param stride: Stride, defaults to 2
        :type stride: int, optional
        :param dropout: Dropout rate, defaults to 0.1
        :type dropout: float, optional
        """
        super().__init__()

        n_cnn_features = n_cnn_features//2 # after the first cnn layer, num_features/=2

        self.cnn = Conv2d(in_channels=1, 
                          out_channels=32,
                          kernel_size=kernel,
                          stride=stride,
                          padding=kernel//2)

        self.rescnn_blocks = Sequential(
            ResidualCNNBlock(32,32,3,1,dropout),
            ResidualCNNBlock(32,32,3,1,dropout),
        )

        self.fc = Linear(in_features=n_cnn_features*32, out_features=rnn_dim)

        self.bigru_blocks = Sequential(
            GRUBlock(in_features=rnn_dim, hidden_size=rnn_dim, dropout=dropout),
            GRUBlock(in_features=rnn_dim*2, hidden_size=rnn_dim, dropout=dropout),
            GRUBlock(in_features=rnn_dim*2, hidden_size=rnn_dim, dropout=dropout),
        )

        self.classifier = Sequential(
            Linear(in_features=rnn_dim*2, out_features=rnn_dim),
            ReLU(),
            Dropout(dropout),
            Linear(in_features=rnn_dim, out_features=n_classes)
        )

    def forward(self,X):
        X = self.cnn(X)
        X = self.rescnn_blocks(X)
        shape = X.size()
        # print(shape)
        X = X.view(-1, shape[1]*shape[2], shape[3]).transpose(1,2) # (batch, time, features)
        X = self.fc(X) # (batch, time, rnn_dim)
        X = self.bigru_blocks(X)  # (batch, time, rnn_dim*2)
        pred = self.classifier(X) # (batch, time, n_classes)

        return pred


def main():
    model = SpeechRecognitionModel(128, 29, 64)
    print(summary(model, torch.rand(4, 1, 64, 124)))


if __name__ == "__main__":
    main()