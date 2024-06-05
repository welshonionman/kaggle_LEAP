import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super(FFNN, self).__init__()

        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
