#  Stair Detection

## Description

This project is my course project of "Pattern Recognition and Machine Learning"

## Setup

The code relies on the deep learning framework of Pytorch

### Requirements

- Anaconda
- numpy
- torch
- torchvision
- tqdm 
- matplotlib
- Pillow
- torchcam

## Dataset

Please download the stair detection dataset and unzip it under your working directory. You are recommand to rename the dataset directory into `stair`.

## Pre-Trained Models

You can download the pretrained model from [here](https://cloud.tsinghua.edu.cn/d/0f6fb8f474d8412dabff/).

You are recommanded to put the pretrained model under the  `pretrained_models/` directory.

## Train

Run the following command to train on ConvNet:
```
python train.py --epochs 100 --image_dir /path/to/your/dataset
```
For example:
```
python train.py --epochs 100 --image_dir stair/public
```
Or you can train on ResNet101 or DenseNet121 with the folowing command:
```
python train.py --model resnet101 --epochs 20 --image_dir /path/to/your/dataset
```
```
python train.py --model densenet --epochs 20 --image_dir /path/to/your/dataset
```
During training, you can supervise the process with tensorboard. The best model on validation set will be recorded under the `logs` directory.

Please refer to the `options.py` for detailed parameters of training.

## Inference

Run the following command to inference with the trained model on tsinghua dataset:
```
python inference.py --pretrained_model /path/to/your/model --image_dir /path/to/your/dataset
```
For example:
```
python inference.py --pretrained_model pretrained_models/conv_net.pth --image_dir stair/tsinghua/scene1
```
After inference, you can see the classfication precision and recall value on terminal. Also, please refer to the `options.py` for detailed parameters of inferencing.

# Feature map visualizatoin

Run the following command to get the visualized feature map of the trained model:
```
python visualization.py --pretrained_model pretrained_models/conv_net.pth --image_dir /path/to/your/dataset
```
You can specifify network layer for creating the heatmap by adding the `--layer` options like:
```
python visualization.py --pretrained_model pretrained_models/conv_net.pth --image_dir /path/to/your/dataset --layer conv_layer_0
```
The visualization output will be recorded under the `results` directory.

# Operator fusion

Run the `operator_fusion.py` to fuse the convolution layers and batch norm layers:
```
python operator_fusion.py
```

# SVM detection

Run the `svm.py` to train with the SVM model:
```
python SVM.py
```
