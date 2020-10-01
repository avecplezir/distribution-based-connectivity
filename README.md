# Distribution-Based Mode Connectivity

This repository contains a PyTorch implementation of the curve-finding methods and WA-ensembling procedure from the paper
 
## [Low-loss connection of weight vectors: distribution-based approaches](https://arxiv.org/abs/2008.00741)
 
by Ivan Anokhin and  Dmitry Yarotsky  (ICML 2020).

Please cite our work if you find it useful in your research:
```latex
@article{anokhin2020low,
  title={Low-loss connection of weight vectors: distribution-based approaches},
  author={Anokhin, Ivan and Yarotsky, Dmitry},
  journal={arXiv preprint arXiv:2008.00741},
  year={2020}
}
```

# Dependencies

Before usage go to the project directory: ```cd distribution_connector```, install requirements: ```pip install -r requirements.txt``` and export PYTHONPATH: ```export PYTHONPATH=$(pwd)```.

# Usage

The code in this repository implements the curve-finding procedure for the various methods for Dense ReLU nets and VGG16, and the Ensembling procedure with Weight Adjusment as discribed in the paper.

## Curve Finding


### Training the endpoints 

To run the curve-finding procedure or the ensembling procedure, you first need to train two or more networks that will serve as the end-points of the curve or as input to the WA ensembling procedure. 
You can train the endpoints using the following command

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --seed=<SEED>
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [MNIST/CIFAR10] 
* ```DATA_PATH``` &mdash; path to the data directory
* ```MODEL``` &mdash; DNN model name:
    - for MNIST dataset:
        - LinearOneLayer 
    - for CIFAR10: 
        - LinearOneLayer100, LinearOneLayer500, LinearOneLayer1000, LinearOneLayer2000 
        - Linear3NoBias, Linear5NoBias, Linear7NoBias
        - VGG16/
        - PreResNet110
* ```EPOCHS``` &mdash; number of training epochs 
* ```LR_INIT``` &mdash; initial learning rate
* ```WD``` &mdash; weight decay 
* ```SEED``` &mdash; use different seeds to get different end-points

For example, use the following commands to train LinearOneLayer on MNIST and LinearOneLayer100, Linear3NoBias, VGG16 on CIFAR10:
```bash
#LinearOneLayer
python3 train.py --dir=checkpoints/LinearOneLayer/chp1 --dataset=MNIST  --data_path=data --model=LinearOneLayer  --epochs=30 --seed=1 --cuda
#LinearOneLayer100
python3 train.py --dir=checkpoints/LinearOneLayer100/chp1 --dataset=CIFAR10  --data_path=data  --model=LinearOneLayer100 --epochs=400 --seed=1 --cuda
#Linear3NoBias
python3 train.py --dir=checkpoints/Linear3NoBias/chp1 --dataset=CIFAR10  --data_path=data  --model=Linear3NoBias --epochs=400 --seed=1 --cuda
#VGG16
python3 train.py --dir=checkpoints/VGG16/chp1 --dataset=CIFAR10 --data_path=data --model=VGG16 --epochs=200 --seed=1 --cuda

```

### Evaluating the curves

To evaluate the methods to connect the endpoints, you can use the following command
```bash
python3 eval_curve.py --dir=<DIR> \
                 --point_finder=<POINTFINDER> \
                 --method=<METHOD>\
                 --end_time=<ENDTIME>\
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --start=<START> \
                 --end=<END> \
                 --num_points=<NUM_POINTS>
```
Parameters
* ```POINTFINDER``` &mdash; algorithm that proposes samples of distribution to connect and may do some additional routine to preserve output of the network [PointFinderWithBias/PointFinderInverseWithBias/PointFinderTransportation/PointFinderInverseWithBiasOT/PointFinderSimultaneous/PointFinderStepWiseButterfly/PointFinderStepWiseInverse/PointFinderStepWiseTransportation/PointFinderStepWiseInverseOT]
* ```METHOD``` &mdash; method that connects proposed by POINTFINDER samples [lin_connect/arc_connect]; lin_connect and arc_connect refer to Eq. 5 and Eq. 6 in the paper respectively.
- ```POINTFINDER``` and ```METHOD``` together determine the curve-finding procedures we examine in the paper. For example, in Table 1 in the paper PointFinderWithBias lin_connect refers to the `Linear`, PointFinderWithBias arc_connect refers to  `Arc`, PointFinderInverseWithBias lin_connect refers to  `Linear + Weight Adjustment`,  PointFinderInverseWithBias arc_connect refers to  `Arc + Weight Adjustment`, PointFinderTransportation lin_connect refers to  `OT`,  PointFinderInverseWithBiasOT lin_connect refers to  `OT + Weight Adjustment`. 
Also, in Table 2 in the paper PointFinderSimultaneous lin_connect refers to `Linear`, PointFinderSimultaneous arc_connect refers to `Arc`, PointFinderStepWiseButterfly lin_connect refers to `Linear + B-fly`, PointFinderStepWiseButterfly arc_connect refers to `Arc + B-fly`,  PointFinderStepWiseInverse lin_connect refers to `Linear + WA`, PointFinderStepWiseInverse arc_connect refers to `Arc + WA`,  PointFinderStepWiseTransportation lin_connect refers to `OT + B-fly`, PointFinderStepWiseInverseOT lin_connect to `OT + WA`, 
* ```START``` &mdash; path to the first checkpoint saved by `train.py`
* ```END``` &mdash; path to the second checkpoint saved by `train.py`
* ```NUM_POINTS``` &mdash; number of points along the curve to use for evaluation
* ```ENDTIME``` &mdash; `POINTFINDER` and `MODEL` dependent time (parametrization of the curve) when the curve reaches the endpoint

`eval_curve.py` outputs the statistics on train and test loss and error along the curve. It also saves a `.npz` file containing more detailed statistics at `<DIR>`.


For example, use the following commands to evaluate the paths on CIFAR10:
```bash
#PointFinderWithBias lin_connect for LinearOneLayer100 model
python3 eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderWithBias --point_finder=PointFinderWithBias --method=lin_connect --model=LinearOneLayer100 --end_time=1 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
#PointFinderInverseWithBias arc_connect for LinearOneLayer100 model
python3 eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderInverseWithBias --point_finder=PointFinderInverseWithBias --method=arc_connect --model=LinearOneLayer100 --end_time=2 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
#PointFinderTransportation lin_connect for LinearOneLayer100 model
python3 eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderTransportation --point_finder=PointFinderTransportation --method=lin_connect --model=LinearOneLayer100 --end_time=1 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda
#PointFinderInverseWithBiasOT lin_connect for LinearOneLayer100 model
python3 eval_curve.py --dir=experiments/eval/LinearOneLayer100/PointFinderInverseWithBiasOT --point_finder=PointFinderInverseWithBiasOT --method=lin_connect --model=LinearOneLayer100 --end_time=2 --data_path=data --num_points=21 --start=checkpoints/LinearOneLayer100/chp1/checkpoint-400.pt  --end=checkpoints/LinearOneLayer100/chp2/checkpoint-400.pt --cuda

#PointFinderSimultaneous lin_connect for Linear3NoBias model
python3 eval_curve.py --dir=experiments/eval/Linear3NoBias/PointFinderSimultaneous --point_finder=PointFinderSimultaneous --method=lin_connect --model=Linear3NoBias --end_time=1 --data_path=data --num_points=21 --start=checkpoints/Linear3NoBias/chp1/checkpoint-400.pt  --end=checkpoints/Linear3NoBias/chp2/checkpoint-400.pt --cuda
#PointFinderStepWiseButterfly arc_connect for Linear3NoBias model
python3 eval_curve.py --dir=experiments/eval/Linear3NoBias/PointFinderStepWiseButterfly --point_finder=PointFinderStepWiseButterfly --method=arc_connect --model=Linear3NoBias --end_time=2 --data_path=data --num_points=21 --start=checkpoints/Linear3NoBias/chp1/checkpoint-400.pt  --end=checkpoints/Linear3NoBias/chp2/checkpoint-400.pt --cuda
#PointFinderStepWiseInverse lin_connect for Linear3NoBias model
python3 eval_curve.py --dir=experiments/eval/Linear3NoBias/PointFinderStepWiseInverse --point_finder=PointFinderStepWiseInverse --method=lin_connect --model=Linear3NoBias --end_time=3 --data_path=data --num_points=31 --start=checkpoints/Linear3NoBias/chp1/checkpoint-400.pt  --end=checkpoints/Linear3NoBias/chp2/checkpoint-400.pt --cuda
#PointFinderStepWiseTransportation lin_connect for Linear3NoBias model
python3 eval_curve.py --dir=experiments/eval/Linear3NoBias/PointFinderStepWiseTransportation --point_finder=PointFinderStepWiseTransportation --method=lin_connect --model=Linear3NoBias --end_time=2 --data_path=data --num_points=21 --start=checkpoints/Linear3NoBias/chp1/checkpoint-400.pt  --end=checkpoints/Linear3NoBias/chp2/checkpoint-400.pt --cuda
#PointFinderStepWiseInverseOT lin_connect for Linear3NoBias model
python3 eval_curve.py --dir=experiments/eval/Linear3NoBias/PointFinderStepWiseInverseOT --point_finder=PointFinderStepWiseInverseOT --method=lin_connect --model=Linear3NoBias --end_time=3 --data_path=data --num_points=31 --start=checkpoints/Linear3NoBias/chp1/checkpoint-400.pt  --end=checkpoints/Linear3NoBias/chp2/checkpoint-400.pt --cuda
#PointFinderStepWiseButterflyConvWBiasOT lin_connect for VGG16
python3 eval_curve.py --dir=experiments/eval/VGG16lin/PointFinderStepWiseButterflyConvWBiasOT/12 --point_finder=PointFinderStepWiseButterflyConvWBiasOT --method=lin_connect --model=VGG16 --end_time=15 --data_path=data --num_points=61 --start=checkpoints/VGG16/chp1/checkpoint-400.pt  --end=checkpoints/VGG16/chp2/checkpoint-400.pt 

```

##  Ensembling with Weight Adjustment

To evaluate results of Ensembling with Weight Adjustment you can use the following command

```bash
python3 eval_ensemble.py --dir=<DIR> \
                 --data_path=<PATH> \
                 --model=<MODEL> \
                 --name=<NAME> \
                 --layer=<LAYER>\
                 --layer_ind=<LAYERIND>\
                 --model_paths=<MPATHS>
```
 
Parameters
* ```NAME``` &mdash; substring that is in all checkpoint's names you want to ensemble. For example, specify NAME=400 if you want to ensemble checkpoints trained 400 epochs.
* ```LAYER``` &mdash; index of the layer in pytorch network implementation after which Weight Adjusment procedure is performed
* ```LAYERIND``` &mdash; index of the layer in parameter space on which Weight Adjusment procedure is performed
* ```MPATHS``` &mdash; path to the directory where checkpoints for ensembling are stored

For example, use the following commands to evaluate the WA(n) Ensembling (please see Section 6 in the paper for WA(n)): 

```bash            
#Linear3NoBias WA(1) 
python3 eval_ensemble.py --dir=experiments/eval_ensemble/Linear3NoBias/ --data_path=data --model=Linear3NoBias --name=400  --layer=1 --layer_ind=2 --model_paths=checkpoints/Linear3NoBias/
#Linear3NoBias WA(2) 
python3 eval_ensemble.py --dir=experiments/eval_ensemble/Linear3NoBias/ --data_path=data --model=Linear3NoBias --name=400  --layer=0 --layer_ind=1 --model_paths=checkpoints/Linear3NoBias/
#Linear5NoBias WA(1)
python3 eval_ensemble.py --dir=experiments/eval_ensemble/Linear5NoBias/ --data_path=data --model=Linear5NoBias --name=400  --layer=3 --layer_ind=4 --model_paths=checkpoints/curves/Linear5NoBias/
#Linear7NoBias WA(1)
python3 eval_ensemble.py --dir=experiments/eval_ensemble/Linear7NoBias/ --data_path=data --model=Linear7NoBias --name=400  --layer=5 --layer_ind=6 --model_paths=checkpoints/Linear7NoBias/
#Linear7NoBias WA(3)
python3 eval_ensemble.py --dir=experiments/eval_ensemble/Linear7NoBias/ --data_path=data --model=Linear7NoBias --name=400  --layer=3 --layer_ind=4 --model_paths=checkpoints/Linear7NoBias/
#VGG16 WA(9)
#python3 eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w9/ --data_path=data --model=VGG16 --name=200  --layer=9 --layer_ind=-14 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#VGG16 WA(10)
#python3 eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w10/ --data_path=data --model=VGG16 --name=200  --layer=10 --layer_ind=-12 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
#VGG16 WA(3)
#python3 eval_ensemble.py --dir=experiments/eval_ensemble/VGG16cifar100w3/ --data_path=data --model=VGG16 --name=200  --layer=3 --layer_ind=-26 --model_paths=checkpoints/cifar100/VGG16 --dataset=CIFAR100
```                 

`eval_ensemble.py` outputs the statistics on ensembling. It also saves a `.npz` file and a `.png` plot containing more details at `<DIR>`.


## Other Relevant Papers
 
 * [Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026) by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov and Andrew Gordon Wilson
 * [Essentially No Barriers in Neural Network Energy Landscape](https://arxiv.org/abs/1803.00885) by Felix Draxler, Kambis Veschgini, Manfred Salmhofer, Fred A. Hamprecht
 * [Topology and Geometry of Half-Rectified Network Optimization](https://arxiv.org/abs/1611.01540) by C. Daniel Freeman, Joan Bruna
 * [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407) by Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
