import numpy as np
import albumentations as A


class Augment:

    def __init__(self, X_train, Y_train):
        self.transform1 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.GridDistortion(p=1.0)
        ])
        self.transform2 = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.GridDistortion(p=1.0)
        ])
        self.X_train = X_train
        self.Y_train = Y_train

    def augment(self):
        # Create empty lists to store augmented images and masks
        augmented_images = []
        augmented_masks = []

        # Iterate over the images and masks arrays
        for i in range(len(self.X_train)):
            image = self.X_train[i]
            mask = self.Y_train[i]

            # Apply augmentation transformations to the image and mask
            augmented = self.transform1(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

            # Append the augmented image and mask to the lists
            augmented_images.append(augmented_image)
            augmented_images.append(image)
            augmented_masks.append(augmented_mask)
            augmented_masks.append(mask)
            # Second agumentation
            augmented = self.transform2(image=image, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

            augmented_images.append(augmented_image)
            augmented_masks.append(augmented_mask)
        # Convert the augmented image and mask lists to NumPy arrays
        augmented_images = np.array(augmented_images)
        augmented_masks = np.array(augmented_masks)

        # Print the shape of the augmented image and mask arrays
        print("Augmented Images shape:", augmented_images.shape)
        print("Augmented Masks shape:", augmented_masks.shape)
        return augmented_images, augmented_masks
