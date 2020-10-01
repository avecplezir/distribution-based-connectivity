import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
# import matplotlib.pyplot as plt

import data
import models
from utils import train_utils
from utils.train_utils import get_models_path

parser = argparse.ArgumentParser(description='Ensemble evaluation')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--dir', type=str, default='experiments/model_name/', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--model_paths', type=str, default='/home/ivan/dnn-mode-connectivity/curves/LinearOneLayer100/', metavar='MDIR',
                    help='training directory (default: /tmp/eval)')
parser.add_argument('--name', type=str, default='400')
parser.add_argument('--layer', type=int, default=-2)
parser.add_argument('--layer_ind', type=int, default=-2)
parser.add_argument('--base', type=str, default='train')
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

device_id = 'cuda:' + str(args.device)
torch.cuda.set_device(device_id)

torch.backends.cudnn.benchmark = True

file_paths = get_models_path(args.model_paths, args.name)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    train_random=False,
    shuffle_train=False,
)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

model.cuda()


def get_funcs(m, data_type='train', layer=0, padding=1, kernel_size=3, stride=1):
    m.cuda()
    m.eval()
    functions = []

    with torch.no_grad():
        for it, (X, y) in enumerate(loaders[data_type]):
            funcs = m(X.cuda(), N=layer)
            if len(funcs.shape) == 4:
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
            if len(funcs.shape) == 2:
                every = 1
                funcs2save = funcs

            functions.extend(funcs2save.cpu().data.numpy())

    # every = max(len(functions) // 150_000, 1)
    # out = np.array(functions[::every])
    out = np.array(functions)
    if every > 1:
        print('reduce to {} length feature map'.format(out.shape[0]))
    del functions

    return out

def adjust_weights(f, fbase_inv, W):
    target_shape = W.shape
    print('target_shape', target_shape)
    if len(target_shape) == 4:
        print('conv')
        print('prod', np.prod(target_shape[1:]))
        W = W.reshape(target_shape[0], np.prod(target_shape[1:]))
    else:
        print('lin')

    print('W, f, f_inv', W.shape, f.shape, fbase_inv.shape)
    Wb2 = W @ f.T @ fbase_inv

    if len(target_shape) == 4:
        Wb2 = Wb2.reshape(target_shape)

    return Wb2


length = len(loaders['test'].dataset)

predictions_sum = np.zeros((length, num_classes))
predictions_sum_adjusted = np.zeros((length, num_classes))

ensemble_accuracy = []
one_model_accuracy = []
ensemble_accuracy_adjusted = []
one_model_accuracy_adjusted = []
base_model = None

for path in tqdm(file_paths):
    print(path)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])

    predictions, targets = train_utils.predictions(loaders['test'], model)
    acc = 100.0 * np.mean(np.argmax(predictions, axis=1) == targets)
    one_model_accuracy.append(acc)

    predictions_sum += predictions
    ens_acc = 100.0 * np.mean(np.argmax(predictions_sum, axis=1) == targets)
    ensemble_accuracy.append(ens_acc)

    print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))

    if base_model is None:
        print('loading base model for weight adjustment ensemble')
        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
        base_model.cuda()
        checkpoint = torch.load(file_paths[0])
        base_model.load_state_dict(checkpoint['model_state'])
        print('getting base feature maps')
        fbase = get_funcs(model, layer=args.layer, data_type=args.base)
        print('computing pseudo inverse')
        fbase_inv = np.linalg.pinv(fbase.T)
    else:
        print('performing weight adjustment')
        feature_map = get_funcs(model, layer=args.layer, data_type=args.base)
        Ws = list(model.parameters())[args.layer_ind].data.cpu().numpy()
        W = adjust_weights(feature_map, fbase_inv, Ws)
        list(base_model.parameters())[args.layer_ind].data.copy_(torch.from_numpy(W))
        for param_base, param_new in \
            zip(list(base_model.parameters())[args.layer_ind+1:],
                list(model.parameters())[args.layer_ind+1:]):
            param_base.data.copy_(param_new)

    predictions, targets = train_utils.predictions(loaders['test'], base_model)
    acc = 100.0 * np.mean(np.argmax(predictions, axis=1) == targets)
    one_model_accuracy_adjusted.append(acc)

    predictions_sum_adjusted += predictions
    ens_acc = 100.0 * np.mean(np.argmax(predictions_sum_adjusted, axis=1) == targets)
    ensemble_accuracy_adjusted.append(ens_acc)

    print('Weight Adjusted Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (acc, ens_acc))

os.makedirs(args.dir, exist_ok=True)

np.savez(
    os.path.join(args.dir, 'ens_stat.npz'),
    ensemble_accuracy=ensemble_accuracy,
    one_model_accuracy=one_model_accuracy,
    ensemble_accuracy_adjusted=ensemble_accuracy_adjusted,
    one_model_accuracy_adjusted=one_model_accuracy_adjusted,
)


# plt.plot(ensemble_accuracy_adjusted, label='ensemble adjusted', c='b')
# plt.plot(ensemble_accuracy, label='ensemble', c='g')
# plt.plot(one_model_accuracy_adjusted, 'bo', c='b')
# plt.plot(one_model_accuracy, 'bo', c='g')
# plt.legend()
# plt.xlabel('N models in ensemble')
# plt.ylabel('accuracy')
# # plt.title(args.model)
# plt.savefig(os.path.join(args.dir, 'ensemble.png'))

print('finished!')
