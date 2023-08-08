import sys

sys.path.append("/content/drive/MyDrive/UAVSegmentation/")

from scripts.scores import Score
from scripts.prepare_dataset import Prepare_Dataset
import argparse
from scripts.model import Models
from tensorflow import keras
import segmentation_models as sm
import os


def main():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--network', help='Network Model type', default='custom')
    parser.add_argument('--backbone', help='Backbone', default='None')
    parser.add_argument('--patch_size', help='Patch size', default=256)
    parser.add_argument('--weight_path', help='path to saved weights', default='./models/')
    parser.add_argument('--binary', help='Enable Binary Segmentation', default=False, action='store_true')
    parser.add_argument('--data_path', help='root path to dataset',
                        default='./data/CoFly-WeedDB')

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    Eval(args['network'], args['backbone'], args['patch_size'], args['weight_path'], args['data_path'],
         args['binary']).eval_model()


class Eval:
    def __init__(self, network, backbone, PATCH_SIZE, path_to_model, data_path, binary):
        self.network = network
        self.backbone = backbone
        self.PATCH_SIZE = self.size_(PATCH_SIZE)
        (self.Y_train_cat, self.Y_test_cat, self.X_train, self.Y_test, self.X_test, self.p_weights,
         self.n_classes) = Prepare_Dataset(self.PATCH_SIZE, binary, data_path=data_path).prepare_all()
        self.path_to_model = path_to_model
        self.total_loss = Prepare_Dataset(self.PATCH_SIZE).get_loss(p_weights=self.p_weights)
        self.binary = binary

    def size_(self, PATCH_SIZE):
        if self.network == 'pspnet':
            if PATCH_SIZE % 48 != 0:
                print('Image size must be divisible by 48')
                PATCH_SIZE = int(PATCH_SIZE / 48) * 48 + 48
                print(f'New image size: {PATCH_SIZE}x{PATCH_SIZE}x3')
        return PATCH_SIZE

    def eval_model(self):
        print('Evaluating...')

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
        else:
            print(f'{self.network} network not available.')
            quit()
        # Compilation
        model.compile(optimizer=optim,
                      loss=self.total_loss,
                      metrics=[metrics])
        print(model.summary())
        path_to_model = str(os.path.join(self.path_to_model,
                                         self.network + '_' + self.backbone + '_' + str(
                                             self.PATCH_SIZE) + '_binary_' + str(self.binary) +
                                         '.hdf5'))
        Score(model, path_to_model, self.X_test, self.Y_test, self.n_classes, self.Y_test_cat).calc_scores()


if __name__ == '__main__':
    main()
