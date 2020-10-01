"""
These methods construct paths between ReLU Dense networks without bias except PointFinderSimultaneous.
PointFinderSimultaneous works for any net.
"""

__all__ = ['PointFinderSimultaneous',
           'PointFinderSimultaneousData',
           'PointFinderStepWiseButterfly',
           'PointFinderStepWiseInverse',
           'PointFinderStepWiseTransportation',
           'PointFinderStepWiseInverseOT',
            ]

import os

import numpy as np
import torch
import ot
from tqdm import tqdm
from glob import glob

from connector import Connector


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def next_layer(W, data):
    funcs = np.maximum(data @ W.T, 0)
    return funcs


def get_model_from_weights(W, architecture):
    model_sampled = architecture.base(num_classes=10, **architecture.kwargs)
    for parameter, w in zip(model_sampled.parameters(), W):
        parameter.data.copy_(torch.from_numpy(w))
    return model_sampled


def get_models_path(path, name, max=100):
    file_paths = glob(os.path.join(path, '**'), recursive=True)
    models_path = sorted(list(filter(lambda x: '.pt' in x and name in x, file_paths)))
    real_len = len(models_path)
    print('len', real_len)
    return sorted(models_path)[:max]


class PointFinderSimultaneous:
    """
    corresponds to Linear and Arc in the paper (Table 2)
    """
    def __init__(self, model1, model2, architecture, *args):
        self.architecture = architecture
        self.model1 = model1
        self.model2 = model2
        self.weights_model1 = self.get_model_weights(model1)
        self.weights_model2 = self.get_model_weights(model2)
        self.depth = len(self.weights_model1)

    @staticmethod
    def get_model_weights(model):
        p = [list(model.parameters())[i].data.cpu().numpy() for i in range(len(list(model.parameters())))]
        return p

    def find_point(self, t, method='arc_connect'):
        weights_model_new = []
        assert 0 <= t <= 1, 't must be between 0 and 1'
        for W1, W2 in zip(self.weights_model1, self.weights_model2):
            Wn = getattr(Connector(W1, W2), method)(t=t)[1]
            weights_model_new.append(Wn)

        m = get_model_from_weights(weights_model_new, self.architecture)
        # m.cuda();
        return m


class PointFinderSimultaneousData(PointFinderSimultaneous):
    def __init__(self, model1, model2, architecture, loader):
        super().__init__(model1, model2, architecture)
        print('getting data')
        self.data = self.get_data(loader)
        print('train len', len(self.data))

    def get_data(self, loader):
        data = []
        for X, y in loader:
            data.append(X.view(X.shape[0], -1).cpu().data.numpy())
        data = np.concatenate(data)
        return data


class PointFinderStepWiseButterfly(PointFinderSimultaneousData):
    """
    corresponds to Linear + B-fly and Arc + B-fly in the paper (Table 2 FC3)
    """
    def __init__(self, model1, model2, architecture, loaders):
        super().__init__(model1, model2, architecture, loaders)
        self.funcs1 = self.find_feature_maps(self.weights_model1, self.data)
        self.funcs2 = self.find_feature_maps(self.weights_model2, self.data)
        self.weights_adjusted = self.adjust_all_weights()

    def find_feature_maps(self, weights_model, data):
        """find feature maps for 2, 3 ,..., N-2 layers of network"""
        print('finding feature maps')
        funcs_list = []
        funcs = data
        for W in tqdm(list(weights_model)[:-2]):
            funcs = next_layer(W, data=funcs)
            funcs_list.append(funcs)
        return funcs_list

    def connect_butterflies(self, W10, W20, W11, W11b2,
                            t=0.5, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        Wn1 = getattr(Connector(W11.T, W11b2.T), method)(t=t)[1].T
        return Wn0, Wn1

    def adjust_weights(self, f1, f2, W):
        f_inv2 = np.linalg.pinv(f2.T)
        Wb2 = W @ f1.T @ f_inv2
        return Wb2

    def adjust_all_weights(self, ):
        """find intermidiate weights between \Theta^A and \Theta^B (see the the paper for the notation) """
        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1,
                                                 self.funcs2,
                                                 self.weights_model1[1:-1]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-1])
        return Wb2_list

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        W11b2 = self.weights_adjusted[layer + 1]
        Wn0, Wn1 = self.connect_butterflies(W10, W20, W11, W11b2,
                                            t=t, method=method)
        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        m.cuda();
        return m


class PointFinderStepWiseInverse(PointFinderStepWiseButterfly):
    """
    corresponds to Linear + WA and Arc + WA in the paper (Table 2 FC3)
    """
    def __init__(self, model1, model2, architecture, loaders):
        super().__init__(model1, model2, architecture, loaders)

    def find_feature_maps(self, weights_model, data):
        """find feature maps of functions \theta_2^AB, ..., \theta_{N-1}^AB
        (see the the paper for the notation)"""

        print('finding feature maps')
        funcs_list = []
        funcs = data
        funcs_list.append(funcs)
        for W in tqdm(list(weights_model)[:-1]):
            funcs = next_layer(W, data=funcs)
            funcs_list.append(funcs)
        return funcs_list

    def connect(self, W10, W20, t, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        return Wn0

    def adjust_all_weights(self, ):
        """adjust weights of the first model (model1) according to feature maps of model1, model2
        in a way that resulting model will have the output of the model1 """

        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1[1:],
                                                 self.funcs2[1:],
                                                 self.weights_model1[1:]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-1])
        return Wb2_list

    def find_intermediate_point(self, t, layer, method):
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        Wn0 = self.connect(W10, W20, t=t, method=method)
        f1 = self.funcs1[layer + 1]
        f2 = next_layer(Wn0, data=self.funcs2[layer])
        Wn1 = self.adjust_weights(f1, f2, W11)
        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]
        return weights_model_t

    def last_layer_interpolation(self, t, layer, method):
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        Wn0 = self.connect(W10, W20, t=t, method=method)
        weights_model_t = self.weights_model2[:layer] + [Wn0]
        return weights_model_t

    def find_point(self, t, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        assert layer <= self.depth + 0.1, 'the network is shot for this t value'

        if layer == self.depth:
            layer -= 1
            t = 1
        if layer == self.depth - 1:
            weights_model_t = self.last_layer_interpolation(t, layer, method=method)
        else:
            weights_model_t = self.find_intermediate_point(t, layer, method=method)
        m = get_model_from_weights(weights_model_t, self.architecture)
        m.cuda();
        return m


class PointFinderStepWiseTransportation(PointFinderStepWiseButterfly):
    """
    corresponds to OT + B-fly in the paper (Table 2 FC3)
    """
    def __init__(self, model1, model2, architecture, loaders):
        super().__init__(model1, model2, architecture, loaders)
        self.GO = []
        self.M = []

    def solve_optimal_transport_problem(self, weights1, weights2):
        n = len(weights1)
        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
        # loss matrix
        M = ot.dist(weights1, weights2)
        M /= M.max()
        self.M.append(M)
        GO = ot.emd(a, b, M)
        self.GO.append(GO)
        return GO

    def butterfly_weights(self, W1, W2):
        samples = np.hstack([W1, W2.T])
        return samples

    def unbutterfly_weights(self, samples, l1):
        Wn0 = samples[:, :l1]
        Wn1 = samples[:, l1:].T
        return Wn0, Wn1

    def find_point(self, t=0.5, method='lin_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        W11b2 = self.weights_adjusted[layer + 1]
        weights1 = self.butterfly_weights(W10, W11)
        weights2 = self.butterfly_weights(W20, W11b2)
        GO = self.solve_optimal_transport_problem(weights1, weights2)
        # TODO: make an assertion that there are only zero and ones in GO
        indices = np.argmax(GO, axis=-1)

        weights2_permuted = weights2[indices]
        weights_model_new = getattr(Connector(weights1, weights2_permuted), method)(t=t)[1]
        Wn0, Wn1 = self.unbutterfly_weights(weights_model_new, len(W10.T))
        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]

        m = get_model_from_weights(weights_model_t, self.architecture)
        m.cuda();
        return m


class PointFinderStepWiseInverseOT(PointFinderStepWiseTransportation):
    """
    corresponds to OT + WA in the paper (Table 2 FC3)
    """
    def __init__(self, model1, model2, architecture, loaders):
        super().__init__(model1, model2, architecture, loaders)
        self.funcs1 = self.find_feature_maps(self.weights_model1, self.data)
        self.funcs2 = self.find_feature_maps(self.weights_model2, self.data)
        self.weights_adjusted = self.adjust_all_weights()

    def find_feature_maps(self, weights_model, data):
        """find feature maps of functions \theta_2^AB, ..., \theta_{N-1}^AB
        (see the the paper for the notation)"""

        print('finding feature maps')
        funcs_list = []
        funcs = data
        funcs_list.append(funcs)
        for W in tqdm(list(weights_model)[:-1]):
            funcs = next_layer(W, data=funcs)
            funcs_list.append(funcs)
        return funcs_list

    def connect(self, W10, W20, t, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        return Wn0

    def adjust_all_weights(self, ):
        """adjust weights of the first model (model1) according to feature maps of model1, model2
        in a way that resulting model will have the output of the model1 """

        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1[1:],
                                                 self.funcs2[1:],
                                                 self.weights_model1[1:]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-1])
        return Wb2_list

    def last_layer_interpolation(self, t, layer, method):
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]
        Wn0 = self.connect(W10, W20, t=t, method=method)
        weights_model_t = self.weights_model2[:layer] + [Wn0]
        return weights_model_t

    def find_intermediate_point(self, t, layer, method):
        W11 = self.weights_model1[layer + 1]
        W20 = self.weights_model2[layer]
        W10 = self.weights_adjusted[layer]

        GO = self.solve_optimal_transport_problem(W10, W20)
        indices = np.argmax(GO, axis=-1)
        W20_permuted = W20[indices]
        Wn0 = getattr(Connector(W10, W20_permuted), method)(t=t)[1]
        f1 = self.funcs1[layer + 1]
        f2 = next_layer(Wn0, data=self.funcs2[layer])
        Wn1 = self.adjust_weights(f1, f2, W11)

        weights_model_t = self.weights_model2[:layer] + [Wn0, Wn1] + self.weights_model1[layer + 2:]
        return weights_model_t

    def find_point(self, t=0.5, method='lin_connect'):

        layer = int(t // 1)
        t = t - layer
        assert layer <= self.depth + 0.1, 'the network is shot for this t value'

        if layer == self.depth:
            layer -= 1
            t = 1
        if layer == self.depth - 1:
            weights_model_t = self.last_layer_interpolation(t, layer, method=method)
        else:
            weights_model_t = self.find_intermediate_point(t, layer, method=method)
        m = get_model_from_weights(weights_model_t, self.architecture)
        m.cuda();
        return m
