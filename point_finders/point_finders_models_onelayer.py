"""
These methods construct paths between ReLU One Hidden layer networks
"""

__all__ = ['PointFinderWithBias',
           'PointFinderTransportation',
           'PointFinderInverseWithBias',
           'PointFinderInverseWithBiasOT',
            ]

import torch
import numpy as np
import ot

from connector import Connector
from .point_finders_models import PointFinderSimultaneousData

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class PointFinderWithBias:
    """
    corresponds to Linear and Arc in the paper (Table 1)
    """
    def __init__(self, model1, model2, architecture, loader):
        self.architecture = architecture
        self.model1 = model1
        self.model2 = model2
        self.weights_model1 = self.get_model_weights(model1)
        self.weights_model2 = self.get_model_weights(model2)
        self.depth = len(self.weights_model1)

    def get_model_from_weights(self, W, B, architecture):
        model_sampled = architecture.base(num_classes=10, **architecture.kwargs)
        model_samples = np.array(W)  # .cpu().data.numpy()
        SIZE = model_sampled.middle_dim

        offset = 0
        for parameter in list(model_sampled.parameters())[:-1]:
            size = int(np.prod(parameter.size()) / SIZE)
            value = model_samples[:, offset:offset + size]
            if size == 10 or size == 1:
                value = value.T
            value = value.reshape(parameter.size())
            parameter.data.copy_(torch.from_numpy(value))
            offset += size

        list(model_sampled.parameters())[-1].data.copy_(torch.tensor(B))

        return model_sampled

    def get_b(self, model1, model2):
        B = []
        B.append(list(model1.parameters())[-1].cpu().data.numpy())
        B.append(list(model2.parameters())[-1].cpu().data.numpy())
        B = torch.tensor(np.array(B))
        return B

    def get_model_weights(self, model):
        p1 = list(model.parameters())[0].data.cpu().numpy()
        p2 = list(model.parameters())[1].data.cpu().numpy()
        p3 = list(model.parameters())[2].transpose(0, 1).data.cpu().numpy()
        samples = np.hstack([p1, p2[:, None], p3])
        return samples

    def find_point(self, t, method='arc_connect'):
        assert 0 <= t <= 1, 't must be between 0 and 1'
        weights_model_new = getattr(Connector(self.weights_model1, self.weights_model2), method)(t=t)[1]
        B = self.get_b(self.model1, self.model2)
        B = getattr(Connector(B[:1], B[1:]), method)(t=t)[1]
        m = self.get_model_from_weights(weights_model_new, B[0], self.architecture)
        m.cuda();
        return m


class PointFinderTransportation(PointFinderWithBias):
    """
    corresponds to OT in the paper (Table 1)
    """
    def __init__(self, model1, model2, architecture, loader):
        super().__init__(model1, model2, architecture, loader)
        self.solve_optimal_transport_problem()
        # find bijection
        self.indices = np.argmax(self.G0, axis=-1)
        self.weights_model2_permuted = self.weights_model2[self.indices]

    def solve_optimal_transport_problem(self, ):
        self.n = len(self.weights_model1)
        self.a, self.b = np.ones((self.n,)) / self.n, np.ones((self.n,)) / self.n  # uniform distribution on samples
        # loss matrix
        self.M = ot.dist(self.weights_model1, self.weights_model2)
        self.M /= self.M.max()
        self.G0 = ot.emd(self.a, self.b, self.M)

    def find_point(self, t, method='arc_connect'):
        assert 0 <= t <= 1, 't must be between 0 and 1'
        weights_model_new = getattr(Connector(self.weights_model1, self.weights_model2_permuted), method)(t=t)[1]
        B = self.get_b(self.model1, self.model2)
        B = getattr(Connector(B[:1], B[1:]), method)(t=t)[1]
        m = self.get_model_from_weights(weights_model_new, B[0], self.architecture)
        m.cuda();
        return m


class PointFinderInverseWithBias(PointFinderSimultaneousData):
    """
    corresponds to Linear + WA and Arc + WA in the paper (Table 1)
    """
    def __init__(self, model1, model2, architecture, loaders):
        super().__init__(model1, model2, architecture, loaders)
        self.architecture = architecture
        self.W10 = list(model1.parameters())[0].data.cpu().numpy()
        self.W20 = list(model2.parameters())[0].data.cpu().numpy()

        self.b10 = list(model1.parameters())[1].data.cpu().numpy()
        self.b20 = list(model2.parameters())[1].data.cpu().numpy()

        self.W11 = list(model1.parameters())[2].data.cpu().numpy()
        self.W21 = list(model2.parameters())[2].data.cpu().numpy()

        self.b11 = list(model1.parameters())[3].data.cpu().numpy()
        self.b21 = list(model2.parameters())[3].data.cpu().numpy()

        self.funcs11 = self.next_layer(self.W10, self.b10, data=self.data)
        self.funcs21 = self.next_layer(self.W20, self.b20, data=self.data)

        self.f_inv21 = np.linalg.pinv(self.funcs21.T)
        self.W11b2 = self.W11 @ self.funcs11.T @ self.f_inv21

        self.W10b = np.concatenate([self.W10, np.expand_dims(self.b10, axis=1)], axis=1)
        self.W20b = np.concatenate([self.W20, np.expand_dims(self.b20, axis=1)], axis=1)

    def next_layer(self, W, b, data):
        pred = np.maximum(data @ W.T + b, 0)
        return pred

    def get_model_from_weights(self, Wn1, bn1, Wn0, bn0, architecture):
        model_sampled = architecture.base(num_classes=10, **architecture.kwargs)
        model_samples = np.array([Wn0, bn0, Wn1, bn1])  # .cpu().data.numpy()
        for parameter, value in zip(model_sampled.parameters(), model_samples):
            parameter.data.copy_(torch.from_numpy(value))
        return model_sampled

    def find_point(self, t=0.5, method='arc_connect'):
        assert 0 <= t <= 2, 't is not in [0, 2]'
        if 0 <= t <= 1:
            s = t
            # go from model1 basic to model2 basis weight in 2th layer
            Wn0 = getattr(Connector(self.W10, self.W20), method)(t=s)[1]
            bn0 = getattr(Connector(self.b10, self.b20), method)(t=s)[1]

            funcs = self.next_layer(Wn0, bn0, data=self.data)
            f_inv = np.linalg.pinv(funcs.T)
            Wn1 = self.W11 @ self.funcs11.T @ f_inv
            m = self.get_model_from_weights(Wn1, self.b11, Wn0, bn0, self.architecture)
        elif 1 < t <= 2:
            s = t - 1
            Wn1 = getattr(Connector(self.W11b2, self.W21), method)(t=s)[1]
            bn1 = getattr(Connector(self.b11, self.b21), method)(t=s)[1]
            m = self.get_model_from_weights(Wn1, bn1, self.W20, self.b20, self.architecture)
        m.cuda()
        return m


class PointFinderInverseWithBiasOT(PointFinderInverseWithBias):
    """
    corresponds to OT + WA in the paper (Table 1)
    """
    def __init__(self, model1, model2, architecture, loaders):
        super().__init__(model1, model2, architecture, loaders)
        self.architecture = architecture
        self.W10 = list(model1.parameters())[0].data.cpu().numpy()
        self.W20 = list(model2.parameters())[0].data.cpu().numpy()
        self.solve_optimal_transport_problem(self.W10, self.W20)
        # find bijection
        self.indices = np.argmax(self.G0, axis=-1)
        self.W20 = self.W20[self.indices]

        self.b10 = list(model1.parameters())[1].data.cpu().numpy()
        self.b20 = list(model2.parameters())[1].data.cpu().numpy()[self.indices]

        self.W11 = list(model1.parameters())[2].data.cpu().numpy()
        self.W21 = list(model2.parameters())[2].data.cpu().numpy()[:, self.indices]

        self.b11 = list(model1.parameters())[3].data.cpu().numpy()
        self.b21 = list(model2.parameters())[3].data.cpu().numpy()

        self.funcs11 = self.next_layer(self.W10, self.b10, data=self.data)
        self.funcs21 = self.next_layer(self.W20, self.b20, data=self.data)

        self.f_inv21 = np.linalg.pinv(self.funcs21.T)
        self.W11b2 = self.W11 @ self.funcs11.T @ self.f_inv21

    def solve_optimal_transport_problem(self, W1, W2):
        self.n = len(W1)
        self.a, self.b = np.ones((self.n,)) / self.n, np.ones((self.n,)) / self.n  # uniform distribution on samples
        # loss matrix
        self.M = ot.dist(W1, W2)
        self.M /= self.M.max()
        self.G0 = ot.emd(self.a, self.b, self.M)

    def test_zeroing(self, ):
        self.W20[0] = 0
        m = self.get_model_from_weights(self.W21, self.b21, self.W20 , self.b20, self.architecture)
        m.cuda();
        return m
