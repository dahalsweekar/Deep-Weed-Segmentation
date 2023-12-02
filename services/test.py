import sys

sys.path.append("/content/drive/MyDrive/Deep-Weed-Segmentation/")

import numpy as np
import random
import matplotlib.pyplot as plt
from scripts.prepare_dataset import Prepare_Dataset
import argparse
from scripts.model import Models
from tensorflow import keras
import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm
import os


def main():
    parser = argparse.ArgumentParser(description='Testing Model')
    parser.add_argument('--network', help='Network Model type', default='custom')
    parser.add_argument('--backbone', help='Backbone', default='None')
    parser.add_argument('--weight_path', help='path to saved weights', default='./models')
    parser.add_argument('--image_path', help='path to image', default='./images')
    parser.add_argument('--patch_size', help='Patch size', default=256)
    parser.add_argument('--binary', help='Enable Binary Segmentation', default=False, action='store_true')
    parser.add_argument('--data_path', help='root path to dataset',
                        default='./CoFly-WeedDB')

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    Test(args['network'], args['backbone'], args['weight_path'], args['image_path'], args['patch_size'],
         args['data_path'], args['binary']).test()


class Test:
    def __init__(self, network, backbone, weight_path, input_image, PATCH_SIZE, data_path, binary):
        self.network = network
        self.backbone = backbone
        self.PATCH_SIZE = self.size_(PATCH_SIZE)
        self.weights_path = weight_path
        self.input_image = input_image
        (self.Y_train_cat, self.Y_test_cat, self.X_train, self.Y_test, self.X_test, self.p_weights,
         self.n_classes) = Prepare_Dataset(self.PATCH_SIZE, binary=binary, backbone=backbone, data_path=data_path).prepare_all()
        self.total_loss = Prepare_Dataset(self.PATCH_SIZE).get_loss(p_weights=self.p_weights)
        self.binary = binary

    def size_(self, PATCH_SIZE):
        if self.network == 'pspnet':
            if PATCH_SIZE % 48 != 0:
                print('Image size must be divisible by 48')
                PATCH_SIZE = int(PATCH_SIZE / 48) * 48 + 48
                print(f'New image size: {PATCH_SIZE}x{PATCH_SIZE}x3')
        return PATCH_SIZE

    def test(self):
        print('Testing...')

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        LR = 0.0001
        optim = keras.optimizers.Adam(LR)

        print('Building Model...')
        # Initialize model
        if self.network == 'custom':
            model = Models(self.n_classes, self.PATCH_SIZE, IMG_CHANNELS=3, model_name=self.network,
                           backbone=self.backbone).simple_unet_model()
        elif self.network == 'segnet':
            model, self.backbone = Models(self.n_classes, self.PATCH_SIZE, IMG_CHANNELS=3, model_name=self.network,
                                          backbone=self.backbone).segnet_architecture()
        elif self.network == 'unet' or self.network == 'linknet' or self.network == 'pspnet':
            model = Models(self.n_classes, self.PATCH_SIZE, IMG_CHANNELS=3, model_name=self.network,
                           backbone=self.backbone).segmented_models()
        elif self.network == 'deeplabv3':
            base_model, layers, layer_names = Models(self.n_classes, self.PATCH_SIZE, IMG_CHANNELS=3,
                                                     model_name=self.network,
                                                     backbone=self.backbone).deeplabv3(name=self.backbone,
                                                                                       weights='imagenet',
                                                                                       height=self.PATCH_SIZE,
                                                                                       width=self.PATCH_SIZE)
            model = tasm.DeepLabV3plus(n_classes=self.n_classes, base_model=base_model, output_layers=layers,
                                       backbone_trainable=False)
            model.build((None, self.PATCH_SIZE, self.PATCH_SIZE, 3))
        else:
            print(f'{self.network} network not available.')
            quit()
        # Compilation
        model.compile(optimizer=optim,
                      loss=self.total_loss,
                      metrics=[metrics])
        print(model.summary())
        path_to_model = os.path.join(self.weights_path,
                                     self.network + '_' + self.backbone + '_' + str(
                                         self.PATCH_SIZE) + '_binary_' + str(self.binary) +
                                     '.hdf5')
        try:
            model.load_weights(path_to_model)
        except Exception as err:
            print(err)
            quit()

        counter = 0
        for img in range(len(self.X_test)):
            counter = counter + 1
            test_img = self.X_test[img]
            ground_truth = self.Y_test[img]
            # test_img_norm=test_img[:,:,0][:,:,None]
            test_img_input = np.expand_dims(test_img, 0)
            prediction = (model.predict(test_img_input))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]

            plt.figure(figsize=(12, 8))
            plt.subplot(231)
            plt.title('Testing Image')
            plt.imshow(test_img)
            plt.subplot(232)
            plt.title('Testing Label')
            plt.imshow(ground_truth, cmap='jet')
            plt.subplot(233)
            plt.title('Prediction on test image')
            plt.imshow(predicted_img, cmap='jet')
            plt.savefig(f'./plots/figure_{counter}_.png')
            print(f'{(counter/len(self.X_test))*100}% done.')


if __name__ == '__main__':
    main()
