# H-Net
This repository is a tensorflow implementation for

> Liu, Weiquan, Xuelun Shen, Cheng Wang, Zhihong Zhang, Chenglu Wen, and Jonathan Li. "H-Net: Neural Network for Cross-domain Image Patch Matching." In *IJCAI*, pp. 856-863. 2018. 



If you use this code in your research, please cite [the paper](https://www.ijcai.org/proceedings/2018/0119.pdf).

![architecture](/examples/Figure3.jpg)



## Environment

This code is based on Python3 and tensorflow  with CUDA 9.0.



## Pretrained models

1. Download the Pretrained [model](https://drive.google.com/open?id=1oO91kqGwZGBE64ypzyFPoTAggbQTstnz). 
2. Extract them to the current folder so that they fall under `model` for example.



## Cross Domain Dataset

There are 160 pair full size [images](https://drive.google.com/open?id=1ZwBkYRB9C0-rWvFtKpglM9UwQQBoXxgM) for training. You can process it as you wish.

Download [test data](https://drive.google.com/open?id=1u1Z00nwd0DbFWmbJrtWaLItd0QkV8gPg)  and extract it. (Remember to change the `data_base_path` in file `trainHNet.py`)

