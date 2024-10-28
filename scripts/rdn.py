import torch
import torch.nn as nn
import torch.optim as optim

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=64, num_layers=6):
        super(ResidualDenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.ModuleList(layers)
        self.local_feature_fusion = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, 1))
            inputs.append(out)
        return x + self.local_feature_fusion(torch.cat(inputs, 1))

class RDN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_rdb_blocks=4, growth_rate=64, num_layers=6):
        super(RDN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.rdb_blocks = nn.ModuleList([ResidualDenseBlock(num_features, growth_rate, num_layers) for _ in range(num_rdb_blocks)])
        self.global_feature_fusion = nn.Conv2d(num_features * num_rdb_blocks, num_features, kernel_size=1)
        self.final_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        rdb_outputs = [rdb(x) for rdb in self.rdb_blocks]
        x = self.global_feature_fusion(torch.cat(rdb_outputs, 1)) + x
        return self.final_conv(x)