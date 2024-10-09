# SATPose: Enhancing Low-Resolution Human Pose Estimation with Structure-Aware Transformer
This is the readme file for the code release of "SATPose: Enhancing Low-Resolution Human Pose Estimation with Structure-Aware Transformer" on PyTorch platform.

## Training from scratch
For the training stage, you should run：
```bash
python tools/dist_train.py --cfg experiments/coco/higher_hrnet/w32_256_adam_lr1e-3.yaml
```
For the testing stage, you should run:
```bash
python tools/valid.py --cfg experiments/coco/higher_hrnet/w32_256_adam_lr1e-3.yaml
```