import sys

sys.path.insert(0, '../forward_mode_tensorized_src')
import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.num_layers = layers
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)

        if layers >= 4:
            self.fc3 = nn.Linear(100, 100)
        if layers == 5:
            self.fc4 = nn.Linear(100, 100)

        self.fc_final = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        if self.num_layers >= 4:
            x = torch.sigmoid(self.fc3(x))
        if self.num_layers == 5:
            x = torch.sigmoid(self.fc4(x))

        x = self.fc_final(x)
        return x


def FCNBig():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 1024),
        nn.Sigmoid(),
        nn.Linear(1024, 1024),
        nn.Sigmoid(),
        nn.Linear(1024, 1024),
        nn.Sigmoid(),
        nn.Linear(1024, 1024),
        nn.Sigmoid(),
        nn.Linear(1024, 10)
    )


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.tensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class Conv(nn.Module):
    def __init__(self, device, input_size, conv_layers, fc_layers, dropout=0.9):
        super(Conv, self).__init__()

        self.input_size = input_size

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                       nn.Sigmoid()]
            prev_channels = n_channels
            img_dim = (img_dim - kernel_size + 2 * padding) // stride + 1
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 2 < len(fc_layers):
                layers += [nn.Dropout(dropout)]
            if i + 1 < len(fc_layers):
                layers += [nn.Sigmoid()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
