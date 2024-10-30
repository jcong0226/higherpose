# Structure-Aware Transformer for Enhanced Low-Resolution Human Pose Estimation
Official implementation of paper(Structure-Aware Transformer for Enhanced Low-Resolution Human Pose Estimation(HigherPose)).

## Environment

The code is conducted under the following environment:

* Ubuntu 22.04
* Python 3.10
* PyTorch 2.1.1
* CUDA 11.8

## Main Results

[//]: # (### Results on COCO test-dev2017 and top-down methods)

[//]: # (| Arch            | Input size | #Params | GFLOPs | AP   | Ap .5 | AP .75 | AP &#40;M&#41; | AP &#40;L&#41; |)

[//]: # (|-----------------|------------|---------|--------|------|-------|--------|--------|--------|)

[//]: # (| HRNet_W32       | 256x192    | 28.5M   | 7.1    | 73.4 | 89.5  | 80.7   | 70.2   | 80.1   |)

[//]: # (| HRNet_W48       | 256x192    | 63.6M   | 14.6   | 75.1 | 90.6  | 82.2   | 71.5   | 81.8   |)

[//]: # (| TransPose-H-S   | 256x192    | 8.0M    | 10.2   | 73.4 | 91.6  | 81.1   | 70.1   | 79.3   |)

[//]: # (| TransPose-H-A6  | 256x192    | 17.5M   | 21.8   | 75.0 | 92.2  | 82.3   | 71.3   | 81.1   |)

[//]: # (| TokenPose-S-v2  | 256x192    | 6.2M    | 11.6   | 73.1 | 91.4  | 80.7   | 69.7   | 79.0   |)

[//]: # (| TokenPose-L/D24 | 256x192    | 27.5M   | 11.0   | 75.1 | 92.1  | 82.5   | 71.7   | 81.1   |)

[//]: # (| HRNet-W32+UDP   | 256x192    | -       | -      | 75.2 | 92.4  | 82.9   | 72.0   | 80.8   |)

[//]: # (| HRNet-W48+UDP   | 256x192    | -       | -      | 75.7 | 92.4  | 83.3   | 72.5   | 81.4   |)

[//]: # (| **higher_pose** | 256x256    | 16.3M   | 5.64   | 73.9 | 91.7  | 82.0   | 70.5   | 79.5   |)

[//]: # ()
[//]: # (### Results on COCO test-dev2017 and bottom-up methods)

[//]: # (| Arch           | Input size | #Params | GFLOPs | AP   | Ap .5 | AP .75 | AP &#40;M&#41; | AP &#40;L&#41; |)

[//]: # (|----------------|------------|---------|--------|------|-------|--------|--------|--------|)

[//]: # (| higherHRNet_W32| 512        | 28.6M   | 47.9   | 66.4 | 87.5  | 72.8   | 61.2   | 74.2   |)

[//]: # (| +UDP           | 512        | -       | -      | 69.1 | 89.1  | 75.8   | 64.4   | 75.5   |)

[//]: # (| higherHRNet_W48| 640        | 63.8M   | 154.3  | 68.4 | 88.2  | 75.1   | 64.4   | 74.2   |)

[//]: # (| +UDP           | 640        | -       | -      | 70.5 | 89.4  | 77.0   | 66.8   | 75.4   |)

[//]: # (| LitePose-S     | 448x448    | 2.7M    | 5      | 56.7 | -     | -      | -      | -      |)

[//]: # (| LitePose-XS    | 256x256    | 1.7M    | 11.6   | 37.8 | -     | -      | -      | -      |)

[//]: # (| **higher_pose**| 512x512    | 29.06M  | 54.9   | 57.9 | 82.6  | 62.9   | 50.4   | 68.5   |)

[//]: # (| **higher_pose**| 256x256    | 29.06M  | 12.01  | 43.4 | 70.6  | 45.2   | 30.3   | 62.1   |)

### Results on COCO test-dev2017 and bottom-up methods
| Arch            | Input size | #Params | AP   | Ap .5 | AP .75 | AP (M) | AP (L) |
|-----------------|------------|---------|------|-------|--------|--------|--------|
| HigherHRNet_W32 | 512x512    | 28.6M   | 56.9 | 81.2  | 61.9   | 49.0   | 68.7   |
| LitePose_S      | 448x448    | 2.7M    | 56.7 | -     | -      | -      | -      |
| HigherHRNet_W32 | 256x256    | 29.0M   | 43.4 | 70.6  | 45.2   | 30.3   | 62.1   |
| LitePose_XS     | 256x256    | 1.7M    | 37.8 | -     | -      | -      | -      |
| higher_pose     | 512x512    | 29.0M   | 57.9 | 82.6  | 62.9   | 50.4   | 68.5   |
| higher_pose     | 256x256    | 29.0M   | 43.4 | 70.6  | 45.2   | 30.3   | 62.1   |

### Results on CrowdPose test dataset and bottom-up methods
| Arch            | Input size | #Params | AP   | Ap .5 | AP .75 |
|-----------------|------------|---------|------|-------|--------|
| HigherHRNet_W32 | 512x512    | 28.6M   | 54.7 | 80.3  | 57.7   |
| LitePose_S      | 448x448    | 2.7M    | 58.3 | 81.1  | 61.8   |
| HigherHRNet_W32 | 256x256    | 28.6M   | 38.5 | 64.9  | 38.3   |
| LitePose_XS     | 256x256    | 1.7M    | 49.5 | -     | -      |
| higher_pose     | 512x512    | 29.0M   | 49.5 | 74.7  | 51.4   |
| higher_pose     | 256x256    | 29.0M   | 49.1 | 75.3  | 51.3   |

## Training from scratch
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.

3. Training and Testing
   For the training stage on COCO Dataset, you should run：
   ```bash
   python tools/dist_train.py --cfg experiments/coco/higher_pose/w32_256_adam_lr1e-3.yaml
   ```
   For the testing stage on COCO Dataset, you should run:
   ```bash
   python tools/valid.py --cfg experiments/coco/higher_pose/w32_256_adam_lr1e-3.yaml
   ```
    For the training stage on CrowdPose Dataset, you should run：
   ```bash
   python tools/dist_train.py --cfg experiments/crowd_pose/higher_pose/w32_512_adam_lr1e-3.yaml
   ```
   For the testing stage on COCO Dataset, you should run:
   ```bash
   python tools/valid.py --cfg experiments/crowd_pose/higher_pose/w32_512_adam_lr1e-3.yaml
   ```
   
### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{14cheng2020higherhrnet,
  title={Higherhrnet: Scale-aware representation learning for bottom-up human pose estimation},
  author={Cheng, Bowen and Xiao, Bin and Wang, Jingdong and Shi, Honghui and Huang, Thomas S and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5386--5395},
  year={2020}
}