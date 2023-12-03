# Deep-Weed-Segmentation

## Introduction
This repository offers an implementation of diverse segmentation models designed for classifying weeds into four distinct categories. The provided method allows the integration of different networks and backbones to create a combination of choices.

## Task List
  - [x] Blank visualizer problem (Only occurs when saved model is loaded)
  - [x] UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples (Only occurs when saved model is loaded)


## Features

  ### Training/Evaluation

| Flags  | Usage |
| ------------- | ------------- |
| ```--network``` | Define network (Default: custom)  | 
| ```--backbone```  | Define backbone	(Default: None)  |                                                                   
| ```--patch_size```  | Define patch size (Default:256) |
| ```--weight_path```  | Set path to model weights  | 
| ```--data_path```  | Set path to data  | 
| ```--epoch```  | Set number of epochs (Default: 50)  |
| ```--verbose```  | Set verbose (Default: 1)  |
| ```--batch_size```  | Set Batch size (Default: 8)  |
| ```--validation_size```  | Set Validation size (Default: 0.1)  |
| ```-test_split```  | Set test size (Default: 0.2)  |
| ```--visualizer```  | Enable visualizer (Default: Not enabled)  |
| ```--score```  | Enable score calculation after training (Default: Not enabled)  |
| ```--test```  | Enable testing after training (Default: Not enabled)  |
| ```--binary```  | Enable class 2 training (Default: Not enabled)  |
| ```--augment```  | Enable Augmentation (Default: Not enabled) **_WARNING!_ May cause system to crash** |
| ```--threshold```  | Set threshold value (Default: 0.03)  |

| Network  | BackBone |
| ------------- | ------------- |
| ```custom``` |```None``` |
| ```unet``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```,                                                            ```densenet121```, ```mobilenetv2``` |
| ```linknet``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```,                                                            ```densenet121```, ```mobilenetv2``` |
| ```pspnet``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```,                                                            ```densenet121```, ```mobilenetv2``` |
| ```segnet``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```,                                                             ```densenet121```, ```mobilenetv2``` |
| ```deeplabv3``` | ```vgg16```, ```resnet50```, ```inceptionv3```, ```efficientnetb0```,                                                             ```densenet121```, ```mobilenetv2``` |

for ```pspnet``` image size must be divisible by 48, the image size will be adjusted accordingly.

## Installation
  ### Requirements
    -Python3
    -Cuda

  ### Install
    1. git clone https://github.com/dahalsweekar/Deep-Weed-Segmentation.git
    2. pip install -r requirements.txt 
    
## Training 

  > Training is set to early stopping
 ```
 python services/train.py --network unet --backbone vgg16 --patch_size 128 --batch_size 4 --epoch 20 --score --data_path /content/drive/MyDrive/data/CoFly-WeedDB 
 ```
## Models

  > Trained models are saved in ./models/

## Dataset

  > Root of the dataset, by default, is ./data/CoFlyWeed-DB/
```
|
|
|__./data/CoFlyWeed-DB/
	|
	|___/images
		|
		|__*.jpg .png*
	|
	|___/labels_1d
		|
		|__*.jpg .png*
```
## Evaluation

 > A model must be trained and saved in ./models/ folder first
 ```
 python services/eval.py --network unet --backbone vgg16
 ```

## Third-Party Implementations
 - Keras implementation
 - Segmentation model implementations: https://github.com/qubvel/segmentation_models
 - Advance Segmentation models: https://github.com/JanMarcelKezmann/TensorFlow-Advanced-Segmentation-Models

