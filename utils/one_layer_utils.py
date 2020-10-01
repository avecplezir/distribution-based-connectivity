import numpy as np
import torch
import pickle


def get_model(W, B, architecture):
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

    list(model_sampled.parameters())[-1].data.copy_(B.mean(0))

    return model_sampled


def get_b(model1, model2):
    B = []
    B.append(list(model1.parameters())[-1].cpu().data.numpy())
    B.append(list(model2.parameters())[-1].cpu().data.numpy())
    B = torch.tensor(np.array(B))
    return B


def samples(model):
    p1 = list(model.parameters())[0].data.cpu().numpy()
    p2 = list(model.parameters())[1].data.cpu().numpy()
    p3 = list(model.parameters())[2].transpose(0, 1).data.cpu().numpy()
    samples = np.hstack([p1, p2[:, None], p3])
    return samples


def make_dataset(model_dir, dataset, architecture, N_models=20, check=30):

    ind = 1
    S = []
    B = []

    m = architecture.base(num_classes=10, **architecture.kwargs)
    while ind < N_models:
        ckpt = model_dir + str(ind) + '/checkpoint-'+str(check)+'.pt'
        checkpoint = torch.load(ckpt)
        m.load_state_dict(checkpoint['model_state'])

        S.append(samples(m))
        B.append(list(m.parameters())[-1].data.numpy())
        ind += 1

    S = np.concatenate(S)
    save_path = 'data/'+dataset+'.pickle'
    print('saving to ', save_path)

    with open(save_path, 'wb') as handle:
        pickle.dump([S, B], handle)

    return S, torch.tensor(np.array(B))

