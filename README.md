# Deep-Weed-Segmentation

# Introduction
This repository offers an implementation of diverse segmentation models designed for classifying weeds into four distinct categories. The provided method allows the integration of different networks and backbones to create a combination of choices.

# ToDoList
  -Blank visualizer problem
  -UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples

# Features

  ## Training/Evaluation
```
Flags				       Usage										            Available
--network			     Define network (Default: custom)			custom, unet, segnet, linknet, pspnet
--backbone		     Define backbone	(Default: None)			vgg16, resnet34(not for segnet), resnet50(only for segnet), inceptionv3,                                                                    densenet121, mobilenetv2
--patch_size		   Define patch size (Default:256)
--weight_path		   Set path to model weights
--data_path 		   Set path to data
--epoch				     Set number of epochs (Default: 50)
--verbose 			   Set verbose (Default: 1)
--batch_size		   Set Batch size (Default: 8)
--validation_size  Set Validation size (Default: 0.1)
--test_split		   Set test size (Default: 0.2)
--visualizer		   Enable visualizer (Default: Not enabled)
--score				     Enable score calculation after training (Default: Not enabled)
--test				     Enable testing after training (Default: Not enabled)
--binary			     Enable class 2 training (Default: Not enabled)

for pspnet image size must be divisible by 48, the image size will be adjusted accordingly.
```
# Installation
  ## Requirements
    -Python3
    -Cuda
    ```
    1. git clone
    2. pip install -r requirements.txt 
    ```
# Training 

  > Training is set to early stopping
 ```
      1. python services/train.py --network unet --backbone vgg16 --patch_size 128 --batch_size 4 --epoch 20 --score --data_path /content            /drive/MyDrive/data/CoFly-WeedDB 
 ```
# Models

  Trained models are saved in ./models/

# Dataset

  root of the dataset by default is ./data/CoFlyWeed-DB/

# Evaluation

 A model must be trained and saved in ./models/ folder first
 ```
      1. python services/eval.py --network unet --backbone vgg16
 ```

# Third-Party Implementations
 1. keras implementation
 2. segmentation model implementations: https://github.com/qubvel/segmentation_models

