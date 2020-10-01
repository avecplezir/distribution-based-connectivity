"""
The method constructs paths between VGG networks
"""

__all__ = ['PointFinderStepWiseButterflyConvWBias',
            ]

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from connector import Connector
from .point_finders_models import PointFinderSimultaneousData, get_model_from_weights

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class PointFinderStepWiseButterflyConvWBias(PointFinderSimultaneousData):
    """
    corresponds to Linear + B-fly and Arc + B-fly in the paper
    """
    def __init__(self, model1, model2, architecture, loader, padding=1, kernel_size=3, stride=1):
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        super().__init__(model1, model2, architecture, loader)
        self.depth = self.depth // 2
        self.loader = loader
        self.funcs1 = self.find_feature_maps(self.model1, loader, self.depth)
        self.funcs2 = self.find_feature_maps(self.model2, loader, self.depth)
        self.weights_adjusted = self.adjust_all_weights()

    @staticmethod
    def get_data(loader):
        data = []
        for X, y in loader:
            data.append(X.cpu().data.numpy())
        data = np.concatenate(data)
        return data

    @staticmethod
    def get_funcs(m, loader, layer, padding=1, kernel_size=3, stride=1):

        m.eval()
        functions = []

        with torch.no_grad():
            for it, (X, y) in enumerate(loader):
                funcs = m(X, N=layer)
                if len(funcs.shape) == 4:
                    if it == 0:
                        print('funcs', funcs.shape)
                    # TODO: make random features map reduction
                    # TODO: more efficient use of memory?
                    every = max(funcs.shape[-1] // 2, 1)
                    stride = every
                    funcs = F.pad(funcs, (padding, padding, padding, padding))
                    batch, chanels, width, high = funcs.shape
                    funcs2save = []
                    for i in range(0, width - kernel_size + 1, stride):
                        for j in range(0, high - kernel_size + 1, stride):
                            funcs2save.append(funcs[:, :, i:i + kernel_size, j:j + kernel_size])
                    funcs2save = torch.cat(funcs2save, 0)
                    funcs2save = funcs2save.view(funcs2save.size(0), -1)
                    lin_flag = False
                if len(funcs.shape) == 2:
                    every = 1
                    lin_flag = True
                    funcs2save = funcs

                functions.extend(funcs2save.cpu().data.numpy())

        if lin_flag:
            out = np.array(functions)
        else:
            out = np.array(functions[::2])
        if every > 1:
            print('reduce to {} length feature map'.format(out.shape[0]))
        del functions

        return out

    def find_feature_maps(self, model, loader, depth):
        """find feature maps for 2, 3 ,..., N-2 layers of network"""
        print('finding feature maps')
        funcs_list = []
        print('depth', depth)
        for i in range(depth - 1):
            print('current depth', i+1)
            functions = self.get_funcs(model, loader, layer=i+1,
                                       padding=self.padding, kernel_size=self.kernel_size, stride=self.stride)
            funcs_list.append(functions)
        return funcs_list

    @staticmethod
    def transpose(W):
        arange = tuple(np.arange(len(W.shape)))
        W = np.transpose(W, (1, 0) + arange[2:])
        return W

    @staticmethod
    def connect_butterflies(W10, W20, B10, B20, W11, W11b2,
                            t=0.5, method='arc_connect'):
        Wn0 = getattr(Connector(W10, W20), method)(t=t)[1]
        Bn0 = getattr(Connector(B10, B20), method)(t=t)[1]
        Wn1 = getattr(Connector(PointFinderStepWiseButterflyConvWBias.transpose(W11), PointFinderStepWiseButterflyConvWBias.transpose(W11b2)), method)(t=t)[1]
        Wn1 = PointFinderStepWiseButterflyConvWBias.transpose(Wn1)
        return Wn0, Bn0, Wn1

    def adjust_weights(self, f1, f2, W):
        target_shape = W.shape
        print('target_shape', target_shape)
        if len(target_shape) == 4:
            print('conv')
            print('prod', np.prod(target_shape[1:]))
            W = W.reshape(target_shape[0], np.prod(target_shape[1:]))
        else:
            print('lin')

        print('W, f1', W.shape, f1.shape)
        f_inv2 = np.linalg.pinv(f2.T)
        print('f_inv', f_inv2.shape)
        Wb2 = W @ f1.T @ f_inv2

        if len(target_shape) == 4:
            Wb2 = Wb2.reshape(target_shape)

        return Wb2

    def adjust_all_weights(self, ):
        """find intermidiate weights between \Theta^A and \Theta^B (see the the paper for the notation) """
        print('adjusting weights')
        Wb2_list = []
        Wb2_list.append(self.weights_model1[0])
        for i, (f1, f2, W) in tqdm(enumerate(zip(self.funcs1,
                                                 self.funcs2,
                                                 self.weights_model1[::2][1:-1]))):
            Wb2 = self.adjust_weights(f1, f2, W)
            Wb2_list.append(Wb2)
        Wb2_list.append(self.weights_model2[-2])
        return Wb2_list

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
        W11b2 = self.weights_adjusted[layer + 1]
        B10 = self.weights_model1[1::2][layer]
        B20 = self.weights_model2[1::2][layer]
        Wn0, Bn0, Wn1 = self.connect_butterflies(W10, W20, B10, B20, W11, W11b2,
                                            t=t, method=method)
        weights_model_t = self.weights_model2[:2*layer] + [Wn0, Bn0, Wn1] + self.weights_model1[2*layer + 3:]
        m = get_model_from_weights(weights_model_t, self.architecture)
        return m