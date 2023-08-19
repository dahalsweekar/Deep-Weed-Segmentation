import tensorflow as tf
import os
import argparse

os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm
import sys

sys.path.append("/content/drive/MyDrive/UAVSegmentation/")

from scripts.model import Models
from scripts.prepare_dataset import Prepare_Dataset
from scripts.visualizer import Visualizer
from scripts.scores import Score
from scripts.test_ import Test
import warnings

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description='Training Model')
    parser.add_argument('--network', help='Network Model type', default='custom')
    parser.add_argument('--backbone', help='Backbone', default='None')
    parser.add_argument('--patch_size', type=int, help='Patch size', default=256)
    parser.add_argument('--augment', help='Enable Augmentation', default=False, action='store_true')
    parser.add_argument('--weight_path', help='Saved weights path', default='./models/')
    parser.add_argument('--data_path', help='root path to dataset',
                        default='./data/CoFly-WeedDB')
    parser.add_argument('--epoch', type=int, help='number of epoches', default=50)
    parser.add_argument('--verbose', type=int, help='verbose', default=1)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--validation_split', type=float, help='Validation size', default=0.1)
    parser.add_argument('--test_split', type=float, help='Test size', default=0.2)
    parser.add_argument('--visualizer', help='Plot Curve', default=False, action='store_true')
    parser.add_argument('--score', help='print scores after completion', default=False, action='store_true')
    parser.add_argument('--test', help='test after completion', default=False, action='store_true')
    parser.add_argument('--binary', help='Enable Binary Segmentation', default=False, action='store_true')
    parser.add_argument('--threshold', type=float, help='Set threshold cutoff', default=0.03)

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    Train(args['network'], args['backbone'], args['epoch'], args['verbose'], args['batch_size'],
          args['validation_split'], args['test_split'], args['weight_path'], args['visualizer'], args['data_path'],
          args['score'], args['test'], args['binary'], args['augment'], args['threshold'], args['patch_size']).train_model()


class Train:

    def __init__(self, network, backbone, epoch, verbose, batch_size, validation_split, test_split, weight_path,
                 visualizer, data_path, score, test, binary, augment, threshold, PATCH_SIZE=256):
        self.test_size = test_split
        self.network = network
        self.backbone = backbone
        self.augment = augment
        self.threshold = threshold
        self.PATCH_SIZE = self.size_(PATCH_SIZE)
        (self.Y_train_cat, self.Y_test_cat, self.X_train, self.Y_test, self.X_test, self.p_weights,
         self.n_classes) = Prepare_Dataset(self.PATCH_SIZE, self.augment, self.threshold, binary, backbone=backbone,
                                           test_size=test_split,
                                           data_path=data_path).prepare_all()
        self.total_loss = Prepare_Dataset(self.PATCH_SIZE).get_loss(p_weights=self.p_weights)
        self.epoch = epoch
        self.verbose = verbose
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.weight_path = weight_path
        self.visualizer = visualizer
        self.score = score
        self.test = test
        self.binary = binary

    def size_(self, PATCH_SIZE):
        if self.network == 'pspnet':
            if PATCH_SIZE % 48 != 0:
                print('Image size must be divisible by 48')
                PATCH_SIZE = int(PATCH_SIZE / 48) * 48 + 48
                print(f'New image size: {PATCH_SIZE}x{PATCH_SIZE}x3')
        return PATCH_SIZE

    def train_model(self):
        print('Training...')

        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
        LR = 0.0001
        optim = keras.optimizers.Adam(LR)

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

        # Summary
        print(model.summary())

        # Callbacks
        weight_path = str(os.path.join(self.weight_path,
                                       self.network + '_' + self.backbone + '_' + str(
                                           self.PATCH_SIZE) + '_binary_' + str(self.binary) +
                                       '.hdf5'))
        checkpoint = tf.keras.callbacks.ModelCheckpoint(weight_path, verbose=self.verbose, save_best_only=True)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, monitor='val_loss')
        ]
        print(
            f'*********ARCHITECTURE*********** \n\tNetwork: {self.network}\n\tBackBone: {self.backbone}\n**************'
            f'******************')
        history = model.fit(self.X_train,
                            self.Y_train_cat,
                            batch_size=self.batch_size,
                            epochs=self.epoch,
                            verbose=self.verbose,
                            validation_split=self.validation_split,
                            callbacks=callbacks)

        Prepare_Dataset(PATCH_SIZE=self.PATCH_SIZE).save_model(model, path=weight_path)
        print(f'Model saved : {weight_path}')
        if self.visualizer:
            Visualizer(history).plot_curve()
        if self.score:
            Score(model, weight_path, self.X_test, self.Y_test, self.n_classes, self.Y_test_cat).calc_scores()
        if self.test:
            Test(model, weight_path, self.X_test, self.Y_test).test()


if __name__ == '__main__':
    main()
