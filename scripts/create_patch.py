import math
from patchify import patchify
import numpy as np
from skimage import io as skio


class Patches:

    def __init__(self, im_list, msk_list, PATCH_SIZE, threshold=0.03):
        self.im_list = im_list
        self.msk_list = msk_list
        self.PATCH_SIZE = PATCH_SIZE
        self.threshold = threshold

    def image_to_patches(self, image, b_msk=False):
        slc_size = self.PATCH_SIZE
        x = int(math.ceil(int(image.shape[0]) / (slc_size * 1.0)))
        y = int(math.ceil(int(image.shape[1]) / (slc_size * 1.0)))
        padded_shape = (x * slc_size, y * slc_size)
        if not b_msk:
            padded_rgb_image = np.zeros((padded_shape[0], padded_shape[1], 3), dtype=np.uint8)
            padded_rgb_image[:image.shape[0], :image.shape[1]] = image
            patches = patchify(padded_rgb_image, (slc_size, slc_size, 3), step=slc_size)
        elif b_msk:
            padded_rgb_image = np.zeros((padded_shape[0], padded_shape[1]), dtype=np.uint8)
            padded_rgb_image[:image.shape[0], :image.shape[1]] = image
            patches = patchify(padded_rgb_image, (slc_size, slc_size), step=slc_size)

        return patches, slc_size

    def load_image(self, path):
        """
        loads an image based on the path
        """
        rgb_image = skio.imread(path)
        return rgb_image

    def patchify_image_mask(self):
        imgs = []
        anns = []
        AREA = self.PATCH_SIZE * self.PATCH_SIZE
        f_AREA = int(self.threshold * AREA)
        print(f'Threshold: {self.threshold} * {AREA} = {f_AREA}')
        print(f"Patchyfying images and mask...")
        for im_path, msk_path in zip(self.im_list, self.msk_list):
            patches, _ = self.image_to_patches(self.load_image(im_path))
            masks, _ = self.image_to_patches(self.load_image(msk_path), b_msk=True)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, :, :, :]
                    mask = masks[i, j, ::]
                    if mask.reshape(-1).sum() > f_AREA:
                        imgs.append(patch)
                        anns.append(mask)
        return np.array(imgs), np.array(anns)
