"""
These methods construct paths between VGG networks
"""

__all__ = ['PointFinderStepWiseButterflyConvWBiasOT',
           'PointFinderStepWiseButterflyConvWBiasOTWA',
            ]

import ot
import torch
import numpy as np
from copy import deepcopy

from .point_finders_models import get_model_from_weights
from .point_finders_conv import PointFinderStepWiseButterflyConvWBias
from connector import Connector

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class PointFinderStepWiseButterflyConvWBiasOT(PointFinderStepWiseButterflyConvWBias):
    """
    corresponds to OT + B-fly in the paper
    """
    def __init__(self, model1, model2, architecture, loader, padding=1, kernel_size=3, stride=1):
        super().__init__(model1, model2, architecture, loader,
                         padding=padding, kernel_size=kernel_size, stride=stride)
        self.GO = []
        self.M = []

    def solve_optimal_transport_problem(self, weights1, weights2):
        n = len(weights1)
        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples
        # loss matrix
        M = ot.dist(weights1.reshape(weights1.shape[0], -1), weights2.reshape(weights2.shape[0], -1))
        M /= M.max()
        self.M.append(M)
        GO = ot.emd(a, b, M)
        self.GO.append(GO)
        return GO

    def butterfly_weights(self, W1, B, W2):
        samples = np.hstack([W1.reshape(W1.shape[0], -1), B[:, None], self.transpose(W2).reshape(W1.shape[0], -1)])
        return samples

    def unbutterfly_weights(self, weights_butterflyed, W1_shape, W2_shape):
        print('unbutterfly_weights', weights_butterflyed.shape, W1_shape, W2_shape)
        hidden_dim = np.prod(W1_shape[1:])
        print('hidden_dim', hidden_dim)
        W1 = weights_butterflyed[:, :hidden_dim].reshape(*W1_shape)
        B = weights_butterflyed[:, hidden_dim]
        W2 = self.transpose(weights_butterflyed[:, hidden_dim+1:].reshape(W2_shape[1], W2_shape[0], *W2_shape[2:]))
        print('W1 B W2 shape', W1.shape, B.shape, W2.shape)
        return W1, B, W2

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W10 = deepcopy(self.weights_adjusted[layer])
        W20 = deepcopy(self.weights_model2[0::2][layer])

        B10 = deepcopy(self.weights_model1[1::2][layer])
        B20 = deepcopy(self.weights_model2[1::2][layer])

        W11 = deepcopy(self.weights_model1[0::2][layer + 1])
        W11b2 = deepcopy(self.weights_adjusted[layer + 1])

        W1_butterflyed = self.butterfly_weights(W10, B10, W11)
        W2_butterflyed = self.butterfly_weights(W20, B20, W11b2)
        GO = self.solve_optimal_transport_problem(W1_butterflyed, W2_butterflyed)
        indices = np.argmax(GO, axis=-1)
        W2_butterflyed_permuted = W2_butterflyed[indices]

        W20, B20, W11b2 = self.unbutterfly_weights(W2_butterflyed_permuted, W20.shape, W11b2.shape)
        Wn0, Bn0, Wn1 = self.connect_butterflies(W10, W20, B10, B20, W11, W11b2,
                                                 t=t, method=method)
        weights_model_t = self.weights_model2[:2*layer] + [Wn0, Bn0, Wn1] + self.weights_model1[2*layer + 3:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        return m


class PointFinderStepWiseButterflyConvWBiasOTWA(PointFinderStepWiseButterflyConvWBiasOT):
    """
    correspond to OT + WA  in the paper
    """
    def butterfly_weights(self, W1, B):
        samples = np.hstack([W1.reshape(W1.shape[0], -1), B[:, None]])
        return samples

    def unbutterfly_weights(self, weights_butterflyed, W1_shape):
        print('unbutterfly_weights', weights_butterflyed.shape, W1_shape)
        hidden_dim = np.prod(W1_shape[1:])
        print('hidden_dim', hidden_dim)
        W1 = weights_butterflyed[:, :hidden_dim].reshape(*W1_shape)
        B = weights_butterflyed[:, hidden_dim]
        print('W1 B W2 shape', W1.shape, B.shape)
        return W1, B

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W10 = deepcopy(self.weights_adjusted[layer])
        W20 = deepcopy(self.weights_model2[0::2][layer])

        B10 = deepcopy(self.weights_model1[1::2][layer])
        B20 = deepcopy(self.weights_model2[1::2][layer])

        W11 = deepcopy(self.weights_model1[0::2][layer + 1])
        # W11b2 = deepcopy(self.weights_adjusted[layer + 1])

        W1_butterflyed = self.butterfly_weights(W10, B10)
        W2_butterflyed = self.butterfly_weights(W20, B20)
        GO = self.solve_optimal_transport_problem(W1_butterflyed, W2_butterflyed)
        indices = np.argmax(GO, axis=-1)
        W2_butterflyed_permuted = W2_butterflyed[indices]

        W20, B20 = self.unbutterfly_weights(W2_butterflyed_permuted, W20.shape)

        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        Bn0 = getattr(Connector(B10, B20), method)(t=t)[1]

        # beg new
        weights_model_till_n1 = self.weights_model2[:2*layer] + [Wn0, Bn0] + self.weights_model1[2*layer + 2:]
        m_till_n1 = get_model_from_weights(weights_model_till_n1, self.architecture)
        features_new = self.get_funcs(m_till_n1, self.loader, layer+1,
                                      padding=self.padding, kernel_size=self.kernel_size, stride=self.stride)
        Wn1 = self.adjust_weights(self.funcs1[layer], features_new, W11,)
        # end new

        weights_model_t = self.weights_model2[:2*layer] + [Wn0, Bn0, Wn1] + self.weights_model1[2*layer + 3:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        return m