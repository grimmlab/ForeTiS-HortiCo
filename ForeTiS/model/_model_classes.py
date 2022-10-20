import torch


# Class that can be build in to print the passed data between layers for debugging
class PrintLayer(torch.nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x)
        print(x.shape)
        return x


# Class to only get the first output of a lstm layer
class GetOutputZero(torch.nn.Module):
    def __init__(self):
        super(GetOutputZero, self).__init__()

    def forward(self, x):
        lstm_out, (hn, cn) = x
        return lstm_out


# Class to reshape the data suitable for lstm layer
class PrepareForlstm(torch.nn.Module):
    def __init__(self):
        super(PrepareForlstm, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1)


# Class to reshape data suitable for dropout layer
class PrepareForDropout(torch.nn.Module):
    def __init__(self):
        super(PrepareForDropout, self).__init__()

    def forward(self, lstm_out):
        return lstm_out[:, -1, :]
