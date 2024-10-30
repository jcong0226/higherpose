# Structure-Aware Transformer for Enhanced Low-Resolution Human Pose Estimation
Official implementation of paper(Structure-Aware Transformer for Enhanced Low-Resolution Human Pose Estimation(HigherPose)).

## Environment

The code is conducted under the following environment:

* Ubuntu 22.04
* Python 3.8.18
* PyTorch 1.8.1
* CUDA 10.2

## Training from scratch
For the training stage, you should run：
```bash
python tools/dist_train.py --cfg experiments/coco/higher_hrnet/w32_256_adam_lr1e-3.yaml
```
For the testing stage, you should run:
```bash
python tools/valid.py --cfg experiments/coco/higher_hrnet/w32_256_adam_lr1e-3.yaml
```