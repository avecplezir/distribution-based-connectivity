import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import time

import data as dateset
import models
from utils import train_utils as utils
import point_finders


parser = argparse.ArgumentParser(description='DNN curve evaluation')
parser.add_argument('--point_finder', type=str, default='gd', help='PointFinder', metavar='POINTFINDER')
parser.add_argument('--method', type=str, default=None, help='submethod to apply in PointFinder', metavar='METHOD')
parser.add_argument('--beg_time', type=int, default=0, help='time when a path starts', metavar='BEGTIME')
parser.add_argument('--end_time', type=int, default=1, help='time when a path reaches the --end checkpoint', metavar='ENDTIME')

parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                    help='training directory (default: /tmp/eval)')

parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--cuda', action='store_true')

parser.add_argument('--start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

device_id = 'cuda:' + str(args.device)
torch.cuda.set_device(device_id)

os.makedirs(args.dir, exist_ok=True)

if args.cuda:
    torch.backends.cudnn.benchmark = True

loaders, num_classes = dateset.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    train_random=False,
    shuffle_train=False,
)

num_classes = int(num_classes)

architecture = getattr(models, args.model)

print('connecting {} models via {} {} method'.format(args.model, args.point_finder, args.method))

beg_time = time.time()
print('getting loaders')
finder_loaders, _ = dateset.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    0,
    args.transform,
    train_random=False,
    shuffle_train=False)

PointFinder = getattr(point_finders, args.point_finder)
model1 = architecture.base(num_classes=num_classes, **architecture.kwargs)
model2 = architecture.base(num_classes=num_classes, **architecture.kwargs)
model1.load_state_dict(torch.load(args.start)['model_state'])
model2.load_state_dict(torch.load(args.end)['model_state'])
if args.cuda:
    model1.cuda()
    model2.cuda()
pointfinder = PointFinder(model1, model2, architecture, finder_loaders['train'])

regularizer = None

criterion = F.cross_entropy

T = args.num_points
ts = np.linspace(args.beg_time, args.end_time, T)
tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)
dl = np.zeros(T)

previous_weights = None

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

if args.cuda:
    t = torch.FloatTensor([0.0]).cuda()
else:
    t = torch.FloatTensor([0.0])


def get_weights(model):
    return np.concatenate([w.detach().cpu().numpy().ravel() for w in model.parameters()])


for i, t_value in enumerate(ts):
    t.data.fill_(t_value)
    model = pointfinder.find_point(t_value, method=args.method)
    weights = get_weights(model)
    if previous_weights is not None:
        dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
    previous_weights = weights.copy()

    tr_res = utils.test(loaders['train'], model, criterion, regularizer, cuda=args.cuda)
    te_res = utils.test(loaders['test'], model, criterion, regularizer, cuda=args.cuda)

    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]

    values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

print('Length: %.2f' % np.sum(dl))
print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))

computation_time = time.time() - beg_time
print(f'computation_time {computation_time} seconds')

np.savez(
    os.path.join(args.dir, 'curve.npz'),
    ts=ts,
    dl=dl,
    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_loss_int=tr_loss_int,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_nll_int=tr_nll_int,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    tr_err_int=tr_err_int,
    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_loss_int=te_loss_int,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_nll_int=te_nll_int,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
    te_err_int=te_err_int,
    computation_time=computation_time,
)
