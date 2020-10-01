import math
import torch.nn as nn

__all__ = [
    'ConvFC',
    'ConvFCNoBias',
    'Conv2FC1',
]


class ConvBase(nn.Module):
    def __init__(self, in_ch, out_ch, bias):
        super(ConvBase, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=bias),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2))

    def forward(self, input):
        return self.conv(input)


class FCBase(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super(FCBase, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias),
            nn.ReLU(True))

    def forward(self, input):
        return self.fc(input)


class ConvFCDepthBase(nn.Module):
    def __init__(self, num_classes, conv_depth=3, fc_depth=3, bias=True):
        super(ConvFCDepthBase, self).__init__()
        self.conv_part = nn.ModuleList()
        self.fc_part = nn.ModuleList()
        self.conv_depth = conv_depth
        self.fc_depth = fc_depth

        self.conv_part += [ConvBase(3, 32, bias)]

        dim = 32
        for _ in range(conv_depth-1):
            self.conv_part += [ConvBase(dim, dim*2, bias)]
            dim *= 2

        self.fc_part += [FCBase(1152, 1000, bias)]

        for _ in range(fc_depth-2):
            self.fc_part += [FCBase(1000, 1000, bias)]

        self.fc_part += [nn.Linear(1000, num_classes, bias=bias)]

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if bias:
                    m.bias.data.zero_()

    def forward(self, x, N=-1):
        depth = 0

        for layer in self.conv_part:
            if depth == N:
                return x
            depth += 1
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.fc_part:
            if depth == N:
                return x
            depth += 1
            x = layer(x)

        return x


class Conv2FC1(nn.Module):
    def __init__(self, num_classes, conv_depth=2, bias=True):
        super().__init__()
        self.conv_part = nn.ModuleList()
        self.fc_part = nn.ModuleList()
        self.conv_depth = conv_depth

        self.conv_part += [ConvBase(3, 32, bias)]

        dim = 32
        for _ in range(conv_depth-1):
            self.conv_part += [ConvBase(dim, dim*2, bias)]
            dim *= 2

        self.fc_part += [nn.Linear(3136, num_classes, bias=bias)]

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if bias:
                    m.bias.data.zero_()

    def forward(self, x, N=-1):
        depth = 0

        for layer in self.conv_part:
            if depth == N:
                return x
            depth += 1
            x = layer(x)

        x = x.view(x.size(0), -1)

        for layer in self.fc_part:
            if depth == N:
                return x
            depth += 1
            x = layer(x)

        return x


class ConvFCBase(nn.Module):
    def __init__(self, num_classes):
        super(ConvFCBase, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(1152, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

        # Initialize weights
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


class ConvFC:
    base = ConvFCBase
    kwargs = {}


class ConvFCNoBias:
    base = ConvFCDepthBase
    kwargs = {'bias': False}


class Conv2FC1:
    base = Conv2FC1
    kwargs = {'bias': True}
