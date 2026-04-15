# AprMatch

Adaptive Patch Replacement Match

![apr框架图](C:\Users\toby\Desktop\小论文\图片\apr框架图.jpg)

## Getting Started

### Pre-trained Encoders

├── ./pretrained
    ├── mit_b2.pth

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
