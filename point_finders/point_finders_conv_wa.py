"""
The method constructs paths between VGG networks
"""

__all__ = ['PointFinderStepWiseButterflyConvWBiasWA',
            ]

import torch

from .point_finders_models import get_model_from_weights
from .point_finders_conv import PointFinderStepWiseButterflyConvWBias
from connector import Connector

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class PointFinderStepWiseButterflyConvWBiasWA(PointFinderStepWiseButterflyConvWBias):
    """
    corresponds to Linear + WA and Arc + WA in the paper
    """
    def __init__(self, model1, model2, architecture, loader, padding=1, kernel_size=3, stride=1):
        super().__init__(model1, model2, architecture, loader, padding=padding, kernel_size=kernel_size, stride=stride)

    @staticmethod
    def connect_butterflies(W10, W20, B10, B20,
                            t=0.5, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        Bn0 = getattr(Connector(B10, B20), method)(t=t)[1]
        return Wn0, Bn0

    def find_point(self, t=0.5, method='arc_connect'):

        layer = int(t // 1)
        t = t - layer
        if layer >= self.depth - 1:
            layer = self.depth - 2
            t = 1

        assert layer < self.depth, 'the network is shot for this t value'
        W11 = self.weights_model1[0::2][layer + 1]
        W20 = self.weights_model2[0::2][layer]
        W10 = self.weights_adjusted[layer]

        B10 = self.weights_model1[1::2][layer]
        B20 = self.weights_model2[1::2][layer]

        Wn0, Bn0 = self.connect_butterflies(W10, W20, B10, B20,
                                            t=t, method=method)

        weights_model_till_n1 = self.weights_model2[:2*layer] + [Wn0, Bn0] + self.weights_model1[2*layer + 2:]
        m_till_n1 = get_model_from_weights(weights_model_till_n1, self.architecture)
        features_new = self.get_funcs(m_till_n1, self.loader, layer+1,
                                      padding=self.padding, kernel_size=self.kernel_size, stride=self.stride)
        Wn1 = self.adjust_weights(self.funcs1[layer], features_new, W11,)

        weights_model_t = self.weights_model2[:2*layer] + [Wn0, Bn0, Wn1] + self.weights_model1[2*layer + 3:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        return m