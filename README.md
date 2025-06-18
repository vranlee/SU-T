# When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking

The official implementation of the paper：
>  [**When Trackers Date Fish: A Benchmark and Framework for Underwater Multiple Fish Tracking**](https://github.com/vranlee/SU-T)  
>  Weiran Li, Yeqiang Liu, Qiannan Guo, Yijie Wei, Hwa Liang Leo, Zhenbo Li*
>  [**\[Paper\]**](https://github.com/vranlee/SU-T) [**\[Code\]**](https://github.com/vranlee/SU-T)

<div align="center">
<img src="assets/Fig.PNG" width="900"/>
</div>

Contact: vranlee@cau.edu.cn or weiranli@u.nus.edu. Any questions or discussion are welcome!

-----

## Updates
+ [2025.04] We have released the MFT25 dataset and codes of SU-T!
-----

## Abstract
Multiple object tracking (MOT) technology has made significant progress in terrestrial applications, but underwater tracking scenarios remain underexplored despite their importance to marine ecology and aquaculture. We present Multiple Fish Tracking Dataset 2025 (MFT25), the first comprehensive dataset specifically designed for underwater multiple fish tracking, featuring 15 diverse video sequences with 408,578 meticulously annotated bounding boxes across 48,066 frames. Our dataset captures various underwater environments, fish species, and challenging conditions including occlusions, similar appearances, and erratic motion patterns. Additionally, we introduce Scale-aware and Unscented Tracker (SU-T), a specialized tracking framework featuring an Unscented Kalman Filter (UKF) optimized for non-linear fish swimming patterns and a novel Fish-Intersection-over-Union (FishIoU) matching that accounts for the unique morphological characteristics of aquatic species. Extensive experiments demonstrate that our SU-T baseline achieves state-of-the-art performance on MFT25, with 34.1 HOTA and 44.6 IDF1, while revealing fundamental differences between fish tracking and terrestrial object tracking scenarios. MFT25 establishes a robust foundation for advancing research in underwater tracking systems with important applications in marine biology, aquaculture monitoring, and ecological conservation.

## Contributions

+ We introduce MFT25, the first comprehensive multiple fish tracking dataset featuring 15 diverse video sequences with 408,578 meticulously annotated bounding boxes across 48,066 frames, capturing various underwater environments, fish species, and challenging conditions including occlusions, rapid direction changes, and visually similar appearances.
    
+ We propose SU-T, a specialized tracking framework featuring an Unscented Kalman Filter (UKF) optimized for non-linear fish swimming patterns and a novel Fish-Intersection-over-Union (FishIoU) matching that accounts for the unique morphological characteristics and erratic movement behaviors of aquatic species.
    
+ We conduct extensive comparative experiments demonstrating that our tracker achieves state-of-the-art performance on MFT25, with 34.1 HOTA and 44.6 IDF1. Through quantitative analysis, we highlight the fundamental differences between fish tracking and land-based object tracking scenarios.

## Installation Guide

### Prerequisites
- CUDA >= 10.2
- Python >= 3.7
- PyTorch >= 1.7.0
- Ubuntu 18.04 or later (Windows is also supported but may require additional setup)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vranlee/SU-T.git
   cd SU-T
   ```

2. **Create and Activate Conda Environment**
   ```bash
   # Create environment from yaml file
   conda env create -f conda_env.yaml
   
   # Activate the environment
   conda activate su_t
   ```

3. **Download Required Resources**
   - Download pretrained models from [BaiduYun (Password: 9uqc)](https://pan.baidu.com/s/1AkIuViwXCPz5l5Oo-UgtaQ?pwd=9uqc)
   - Download MFT25 dataset from [BaiduYun (Password: wrbg)](https://pan.baidu.com/s/11TkRqNIq4poNAU5dyoL5hA?pwd=wrbg)

4. **Organize the Directory Structure**
   ```
   SU-T/
   ├── pretrained/
   │   └── Checkpoint.pth.tar
   ├── MFT25/
   │   ├── train/
   │   └── test/
   └── ...
   ```

## Usage Guide

### Training

1. **Basic Training Command**
   ```bash
   python tools/train.py \
       -f exps/SU-T.py \        # Base model configuration
       -d 8 \                   # Number of GPUs
       -b 48 \                  # Batch size
       --fp16 \                 # Enable mixed precision training
       -o \                     # Enable occupy GPU memory
       -c pretrained/Checkpoint.pth.tar  # Path to pretrained weights
   ```

2. **Training with ReID Module**
   ```bash
   python tools/train.py \
       -f exps/SU-T-ReID.py \   # ReID model configuration
       -d 8 \
       -b 48 \
       --fp16 \
       -o \
       -c pretrained/Checkpoint.pth.tar
   ```

### Testing

1. **Basic Testing Command**
   ```bash
   python tools/su_tracker.py \
       -f exps/SU-T.py \        # Model configuration
       -b 1 \                   # Batch size
       -d 1 \                   # Number of GPUs
       --fp16 \                 # Enable mixed precision
       --fuse \                 # Enable model fusion
       --expn your_exp_name     # Experiment name
   ```

2. **Testing with ReID Module**
   ```bash
   python tools/su_tracker.py \
       -f exps/SU-T-ReID.py \   # ReID model configuration
       -b 1 \
       -d 1 \
       --fp16 \
       --fuse \
       --expn your_exp_name
   ```

### Additional Configuration Options

- **Model Configuration**: Edit `exps/SU-T.py` or `exps/SU-T-ReID.py` to modify:
  - Learning rate
  - Training epochs
  - Data augmentation parameters
  - Model architecture settings

- **Training Parameters**:
  ```bash
  # Additional training options
  --cache        # Cache images in RAM
  --resume       # Resume from a specific checkpoint
  --trt          # Export TensorRT model
  ```

- **Testing Parameters**:
  ```bash
  # Additional testing options
  --tsize        # Test image size
  --conf         # Confidence threshold
  --nms          # NMS threshold
  --track_thresh # Tracking threshold
  ```

## Tracking Performance

### Comparisons on MFT25 dataset

| Method | Class | Year | HOTA↑ | IDF1↑ | MOTA↑ | AssA↑ | DetA↑ | IDs↓ | IDFP↓ | IDFN↓ | Frag↓ |
|--------|-------|------|-------|-------|-------|-------|-------|------|-------|-------|-------|
| FairMOT | JDE | 2021 | 22.226 | 26.867 | 47.509 | 13.910 | 35.606 | 939 | 58198 | 113393 | 3768 |
| CMFTNet | JDE | 2022 | 22.432 | 27.659 | 46.365 | 14.278 | 35.452 | 1301 | 64754 | 111263 | 2769 |
| TransTrack | TF | 2021 | 30.426 | 35.215 | 68.983 | 18.525 | _50.458_ | 1116 | 96045 | 93418 | 2588 |
| TransCenter | TF | 2023 | 27.896 | 30.278 | 68.693 | **30.255** | 30.301 | 807 | 101223 | 101002 | 1992 |
| TrackFormer | TF | 2022 | 30.361 | 35.285 | **74.609** | 17.661 | **52.649** | 718 | 89391 | 94720 | 1729 |
| TFMFT | TF | 2024 | 25.440 | 33.950 | 49.725 | 17.112 | 38.059 | 719 | 63125 | 102378 | 3251 |
| SORT | SDE | 2016 | 29.063 | 34.119 | 69.038 | 16.952 | 50.195 | 778 | 88928 | 96815 | _1726_ |
| ByteTrack | SDE | 2022 | 31.758 | 40.355 | _69.586_ | 20.392 | 49.712 | **489** | 80765 | 87866 | **1555** |
| BoT-SORT | SDE | 2022 | 26.848 | 36.847 | 49.108 | 19.446 | 37.241 | _500_ | 57581 | 99181 | 2704 |
| OC-SORT | SDE | 2023 | 25.017 | 34.620 | 46.706 | 17.783 | 35.369 | 550 | **52934** | 103495 | 3651 |
| Deep-OC-SORT | SDE | 2023 | 24.848 | 34.176 | 46.721 | 17.537 | 35.373 | 550 | _53478_ | 104024 | 3659 |
| HybridSORT | SDE | 2024 | 32.258 | 38.421 | 68.905 | 20.936 | 49.992 | 613 | 85924 | 90022 | 1931 |
| HybridSORT† | SDE | 2024 | 32.705 | _41.727_ | 69.167 | 21.701 | 49.697 | 562 | 79189 | 85830 | 1963 |
| **SU-T (Ours)** | SDE | 2025 | _33.351_ | 41.717 | 68.450 | 22.425 | 49.943 | 607 | 83111 | _84814_ | 2006 |
| **SU-T† (Ours)** | SDE | 2025 | **34.067** | **44.643** | 68.958 | _23.594_ | 49.531 | 544 | 76440 | **81304** | 2011 |

*Note:  † indicates the integration of ReID module, **Bold** indicates the best performance, _italics_ indicate the second-best performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller input resolution
   - Enable mixed precision training

2. **Installation Failures**
   - Ensure CUDA toolkit matches PyTorch version
   - Try creating environment with `pip` if conda fails
   - Check system CUDA compatibility

3. **Training Issues**
   - Verify dataset path and structure
   - Check GPU memory usage
   - Monitor learning rate and loss curves

## Acknowledgement
A large part of the code is borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [OC_SORT](https://github.com/noahcao/OC_SORT), and [HybridSORT](https://github.com/ymzis69/HybridSORT). Thanks for their wonderful works!

## Citation
The citation format will be given after the manuscript is accepted. Using this repo citation if needed now.

## License
This project is released under the [MIT License](LICENSE).
