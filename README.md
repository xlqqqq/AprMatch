# AprMatch

Adaptive Patch Replacement Match

<img width="7538" height="2256" alt="apr框架图" src="https://github.com/user-attachments/assets/f8d7ee30-5719-4d13-bfa3-2ca1347afb11" />

## Getting Started

### Pre-trained Encoders

```
├── ./pretrained
    ├── mit_b2.pth
```

### Datasets

- CHN6-CUG

- WHU-RuRp

Please modify your dataset path in configuration files.

## Training

### AprMatch

```
sh scripts/train.sh <num_gpu> <port>
```

To train on other datasets or splits, please modify `dataset` and `split` in `train.sh` 
