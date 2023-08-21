import tqdm as tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from keras.utils import normalize
import os

import sys

sys.path.append("/content/drive/MyDrive/UAVSegmentation/")

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import numpy as np
from keras.utils import to_categorical
from scripts.create_patch import Patches
from scripts.augment import Augment


class Prepare_Dataset:
    def __init__(self, PATCH_SIZE, augment=False, threshold=0.03, binary=False, backbone='None',
                 train_split_file='./data/CoFly-WeedDB/train_split1.txt',
                 test_split_file='./data/CoFly-WeedDB/test_split1.txt', IMG_CHANNELS=3,
                 test_size=0.2, data_path='./data/CoFly-WeedDB'):
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.PATCH_SIZE = PATCH_SIZE
        self.IMG_CHANNELS = IMG_CHANNELS
        self.test_size = test_size
        self.backbone = backbone
        self.data_path = data_path
        self.binary = binary
        self.augment = augment
        self.threshold = threshold

    def read_files(self, train_split_file, test_split_file):
        # read test file name and store into a list
        train_ids = []
        test_ids = []
        with open(train_split_file, 'r') as f:
            for line in f:
                # print(lines)
                train_ids.append(line.strip())

        with open(test_split_file, 'r') as f:
            for line in f:
                # print(lines)
                test_ids.append(line.strip())
        return train_ids, test_ids

    def convert_image_id_to_path(self, total_ids, path):
        img_lst = []
        msk_lst = []
        for itm in total_ids:
            image_path = path + '/images/' + itm
            img_lst.append(image_path)
            mask_path = path + '/labels_1d/' + itm
            msk_lst.append(mask_path)
        return img_lst, msk_lst

    def get_image_patches(self, im_list, msk_list):
        imgs, anns = Patches(im_list, msk_list, self.PATCH_SIZE, threshold=self.threshold).patchify_image_mask()
        return imgs, anns

    def binary_class(self, mask):
        mask = np.where(mask > 0, 1, 0)
        return mask

    def label_encoder(self, train_masks):
        labellencoder = LabelEncoder()
        n, h, w = train_masks.shape
        train_masks_reshaped = train_masks.reshape(-1, 1)
        train_masks_reshaped_encoded = labellencoder.fit_transform(train_masks_reshaped)
        train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
        return train_masks_reshaped_encoded, train_masks_encoded_original_shape

    def calc_class_weights(self, train_masks_reshaped_encoded):
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                          classes=np.unique(train_masks_reshaped_encoded),
                                                          y=train_masks_reshaped_encoded)
        print("Class weights are...:", class_weights)
        return class_weights

    def train_test_split(self, train_images, mask):
        if self.backbone == 'None':
            train_images = normalize(train_images, axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(train_images, mask, test_size=self.test_size,
                                                            random_state=0)
        return X_train, X_test, Y_train, Y_test

    def augment_dataset(self, X_train, Y_train):
        X_train, Y_train = Augment(X_train, Y_train).augment()
        return X_train, Y_train

    def covert_to_categorical(self, Y_train, Y_test, n_classes):
        Y_train_cat = to_categorical(Y_train, num_classes=n_classes)
        # Y_train_cat = train_masks_cat.reshape((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], n_classes))

        Y_test_cat = to_categorical(Y_test, num_classes=n_classes)
        # Y_test_cat = test_masks_cat.reshape((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], n_classes))
        return Y_train_cat, Y_test_cat

    def get_loss(self, p_weights):
        dice_loss = sm.losses.DiceLoss(class_weights=np.array(p_weights))
        focal_loss = sm.losses.CategoricalFocalLoss()
        total_loss = dice_loss + (1 * focal_loss)
        return total_loss

    def save_model(self, model, path):
        model.save_weights(path, overwrite=True)

    def prepare_all(self):
        print('Preparing Dataset...')
        # Path to patches
        train_ids, test_ids = self.read_files(train_split_file=self.train_split_file,
                                              test_split_file=self.test_split_file)
        total_ids = train_ids + test_ids
        print(f'Total IDs: {len(total_ids)}')
        im_path, m_path = self.convert_image_id_to_path(total_ids=total_ids,
                                                        path=self.data_path)
        image, mask = self.get_image_patches(im_list=im_path, msk_list=m_path)

        print(f'Image Shape = {image.shape}\nMask Shape = {mask.shape}')

        if self.binary:
            mask = self.binary_class(mask)

        # Reshape the masks array to have a channel dimension
        masks_reshaped_encoded, masks_encoded_original_shape = self.label_encoder(train_masks=mask)

        print(f'Train Encoded shape: {masks_reshaped_encoded.shape}')
        class_weights = self.calc_class_weights(train_masks_reshaped_encoded=masks_reshaped_encoded)

        image = np.squeeze(image)
        mask = np.expand_dims(masks_encoded_original_shape, axis=-1)

        p_weights = class_weights / sum(class_weights)
        print(f'Mean weights: {p_weights}')

        n_classes = len(np.unique(mask))
        print(f'Total Classes: {n_classes}')

        X_train, X_test, Y_train, Y_test = self.train_test_split(train_images=image,
                                                                 mask=mask)
        print(f'Train shape:{X_train.shape}\nTest shape: {X_test.shape}')

        if self.augment:
            print('Augmenting Dataset...')
            X_train, Y_train = self.augment_dataset(X_train, Y_train)
            print(f'Train shape:{X_train.shape}\nTest shape: {X_test.shape}')

        Y_train_cat, Y_test_cat = self.covert_to_categorical(Y_train=Y_train, Y_test=Y_test, n_classes=n_classes)

        return Y_train_cat, Y_test_cat, X_train, Y_test, X_test, p_weights, n_classes
