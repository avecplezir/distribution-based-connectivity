import torch.nn as nn

__all__ = [
    'Linear', 'LinearMNIST', 'LinearOneLayer', 'LinearOneLayer2000', 'LinearNoBias',
    'Linear3NoBias', 'LinearOneLayer60k', 'Linear2NoBias', 'Linear3NoBiasW',
    'LinearOneLayer100', 'LinearOneLayer500', 'LinearOneLayer1000',
    'Linear5NoBias', 'Linear7NoBias',
]


class LinearBase(nn.Module):
    def __init__(self, num_classes, in_dim, middle_dim,  bias=True):
        super(LinearBase, self).__init__()

        self.dims = [in_dim] + list(middle_dim) + [num_classes]
        self.linear_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()

        for i in range(len(self.dims)-2):
            self.linear_layers.append(nn.Linear(self.dims[i], self.dims[i+1], bias=bias))
            self.relu_layers.append(nn.ReLU(True))

        self.linear_layers.append(nn.Linear(self.dims[-2], self.dims[-1], bias=bias))

    def last_layers(self, x, N=-1):

        for i in range(N, len(self.dims) - 2):
            x = self.linear_layers[i](x)
            x = self.relu_layers[i](x)

        x = self.linear_layers[-1](x)

        return x

    def forward(self, x, N=-1):
        x = x.view(x.size(0), -1)
        for i in range(len(self.dims) - 2):
            x = self.linear_layers[i](x)
            x = self.relu_layers[i](x)
            if N == i:
                return x

        x = self.linear_layers[-1](x)

        return x


class LinearOneLayerBase(nn.Module):
    def __init__(self, num_classes, in_dim, middle_dim):
        super(LinearOneLayerBase, self).__init__()
        self.in_dim = in_dim
        self.middle_dim = middle_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, middle_dim),
            nn.ReLU(True),
            nn.Linear(middle_dim, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Linear:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': True,
        'middle_dim': [2*3072, 2*3072, 1152, 1000, 1000],
    }


class LinearNoBias:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2 * 3072, 1152, 1000, 1000],
    }


class Linear3NoBias:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000],
    }


class Linear5NoBias:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000, 1000, 1000],
    }


class Linear7NoBias:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2000, 1000, 1000, 1000, 1000],
    }



class Linear3NoBiasW:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072, 2 * 3072],
    }


class Linear2NoBias:
    base = LinearBase
    kwargs = {
        'in_dim': 3072,
        'bias': False,
        'middle_dim': [2 * 3072],
    }


class LinearMNIST:
    base = LinearBase
    kwargs = {
        'in_dim': 784,
        'bias': True,
        'middle_dim': [2 * 3072, 2 * 3072, 1152, 1000, 1000],
    }


class LinearOneLayer:
    base = LinearOneLayerBase
    kwargs = {
        'in_dim': 784,
        'middle_dim': 2000
    }


class LinearOneLayer100:
    base = LinearOneLayerBase
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 100
    }


class LinearOneLayer500:
    base = LinearOneLayerBase
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 500
    }


class LinearOneLayer1000:
    base = LinearOneLayerBase
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 1000
    }


class LinearOneLayer60k:
    base = LinearOneLayerBase
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 60000
    }


class LinearOneLayer2000:
    base = LinearOneLayerBase
    kwargs = {
        'in_dim': 3072,
        'middle_dim': 2000
    }