"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn

__all__ = ['VGG16', 'VGG19', 'VGG19BN']

config = {
    16: [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    19: [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}


def make_layers(config, batch_norm=False, in_channels=3):
    layer_blocks = nn.ModuleList()
    activation_blocks = nn.ModuleList()
    poolings = nn.ModuleList()

    kwargs = dict()
    conv = nn.Conv2d
    bn = nn.BatchNorm2d

    in_channels = in_channels
    for sizes in config:
        layer_blocks.append(nn.ModuleList())
        activation_blocks.append(nn.ModuleList())
        for channels in sizes:
            layer_blocks[-1].append(conv(in_channels, channels, kernel_size=3, padding=1, **kwargs))
            if batch_norm:
                layer_blocks[-1].append(bn(channels, **kwargs))
            activation_blocks[-1].append(nn.ReLU(inplace=True))
            in_channels = channels
        poolings.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layer_blocks, activation_blocks, poolings


class VGGBase(nn.Module):
    def __init__(self, num_classes, depth=16, batch_norm=False, in_channels=3, in_classifier_dim=512,
                 hidden_classifier_dim=512, lastpooling=True):
        super(VGGBase, self).__init__()
        layer_blocks, activation_blocks, poolings = make_layers(config[depth], batch_norm, in_channels=in_channels)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings
        self.lastpooling = lastpooling
        self.layers_len = len(self.layer_blocks)

        self.classifier = nn.ModuleList()

        classifier_layers = [nn.Dropout(), #0
                            nn.Linear(in_classifier_dim, hidden_classifier_dim), #1
                            nn.ReLU(inplace=True),  #2
                            nn.Dropout(), #3
                            nn.Linear(hidden_classifier_dim, hidden_classifier_dim), #4
                            nn.ReLU(inplace=True), #5
                            nn.Linear(hidden_classifier_dim, num_classes),] #6
        self.classifier.extend(classifier_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, N=-1):
        depth = 0
        for j, (layers, activations, pooling) in enumerate(zip(self.layer_blocks, self.activation_blocks,
                                                           self.poolings)):
            for i, (layer, activation) in enumerate(zip(layers, activations)):
                if depth == N:
                    return x
                depth += 1
                x = layer(x)
                x = activation(x)
            x = pooling(x)

        x = x.view(x.size(0), -1)

        for i in range(7):
            if i in [1, 4, 6]:
                if depth == N:
                    return x
                depth += 1
            x = self.classifier[i](x)

        return x


class VGG16:
    base = VGGBase
    kwargs = {
        'depth': 16,
        'batch_norm': False
    }


class VGG19:
    base = VGGBase
    kwargs = {
        'depth': 19,
        'batch_norm': False
    }


class VGG19BN:
    base = VGGBase
    kwargs = {
        'depth': 19,
        'batch_norm': True
    }
