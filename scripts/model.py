from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
from keras.applications import VGG16, InceptionV3, MobileNetV2, DenseNet121, ResNet50
import segmentation_models as sm
import keras


class Models:

    def __init__(self, n_classes, PATCH_SIZE, IMG_CHANNELS=3, model_name='unet', backbone='None'):
        self.n_classes = n_classes
        self.PATCH_SIZE = PATCH_SIZE
        self.IMG_CHANNELS = IMG_CHANNELS
        self.model_name = model_name
        self.backbone = backbone

    # Define a multi-unet model
    def simple_unet_model(self):
        # Build the model
        inputs = Input((self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
        # s = Lambda(lambda x: x / 255)(inputs)  # No need for this if we normalize our inputs beforehand
        s = inputs

        # Contraction path
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(self.n_classes, (1, 1), activation='softmax')(c9)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    def segmented_models(self):
        model_name = self.model_name
        activation = 'softmax'
        BACKBONE1 = self.backbone
        if BACKBONE1 != 'None':
            preprocess_input1 = sm.get_preprocessing(BACKBONE1)

        # define model
        if model_name == 'unet':
            if BACKBONE1 != 'None':
                model = sm.Unet(BACKBONE1, input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS),
                                encoder_weights='imagenet',
                                encoder_freeze=False, classes=self.n_classes, activation=activation)
            else:
                model = sm.Unet(input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS),
                                encoder_weights='imagenet',
                                encoder_freeze=False, classes=self.n_classes, activation=activation)
        if model_name == 'pspnet':
            if BACKBONE1 != 'None':
                model = sm.PSPNet(BACKBONE1, input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS),
                                  encoder_weights='imagenet',
                                  encoder_freeze=False, classes=self.n_classes, activation=activation)
            else:
                model = sm.PSPNet(input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS),
                                  encoder_weights='imagenet',
                                  encoder_freeze=False, classes=self.n_classes, activation=activation)
        if model_name == 'linknet':
            if BACKBONE1 != 'None':
                model = sm.Linknet(BACKBONE1, input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS),
                                   encoder_weights='imagenet',
                                   encoder_freeze=False, classes=self.n_classes, activation=activation)
            else:
                model = sm.Linknet(input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS),
                                   encoder_weights='imagenet',
                                   encoder_freeze=False, classes=self.n_classes, activation=activation)

        return model

    def segnet_architecture(self):
        if self.backbone == 'vgg16':
            # Load VGG16 without the classification layers (include_top=False)
            weights = VGG16(weights='imagenet', include_top=False,
                            input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
            # Retrieve the VGG16's first layer
            input_m = weights.layers[0].input
            # Retrieve the VGG16's last layer (usually the block5_conv3 layer)
            m_output = weights.get_layer('block5_conv3').output
        elif self.backbone == 'inceptionv3':
            # Load inceptionV3 without the classification layers (include_top=False)
            weights = InceptionV3(weights='imagenet', include_top=False,
                                  input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
            # Retrieve the InceptionV3's first layer
            input_m = weights.layers[0].input
            # Retrieve the InceptionV3's last layer (usually mixed7 layer)
            m_output = weights.get_layer('mixed7').output
            m_output = keras.layers.ZeroPadding2D(padding=1)(m_output)
        elif self.backbone == 'mobilenetv2':
            # Load MobileNet without the classification layers (include_top=False)
            weights = MobileNetV2(weights='imagenet', include_top=False,
                                  input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
            # Retrieve the MobileNetV2's first layer
            input_m = weights.layers[0].input
            # Retrieve the MobileNetV2's last layer (usually block_13_expand layer)
            m_output = weights.get_layer('block_13_expand').output
        elif self.backbone == 'densenet121':
            # Load DenseNet without the classification layers (include_top=False)
            weights = DenseNet121(weights='imagenet', includet_op=False,
                                  input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
            # Retrieve the DenseNet121's first layer
            input_m = weights.layers[0].input
            # Retrieve the DenseNet121's last layer (usually the conv5_block16_concat layer)
            m_output = weights.get_layer('pool4_conv').output
        elif self.backbone == 'resnet50':
            # Load ResNet without the classification layers (include_top=False)
            weights = ResNet50(weights='imagenet', include_top=False,
                               input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
            # Retrieve the ResNet50's first layer
            input_m = weights.layers[0].input
            # Retrieve the ResNet50's last layer (usually the activation_49 layer)
            m_output = weights.get_layer('conv4_block6_out').output
        else:
            if self.backbone != 'None':
                print(
                    f'{self.backbone} backbone is not available. Do you want to construct SegNet encoder block instead?(Y/n)')
                choice = input().lower()
                self.backbone = 'None'
                if choice != 'y':
                    quit()
            # Encode Block
            input_m = Input(shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))

            # Encoder block 1
            conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_m)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

            # Encoder block 2
            conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
            conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            # Encoder block 3
            conv1 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
            conv2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv1)
            conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            # Encoder block 4
            conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
            conv2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv1)
            conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv2)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)

            # # Encoder block 5
            # conv1 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
            # conv2 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv1)
            # conv3 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv2)
            # pool5 = MaxPooling2D(pool_size=(2, 2))(conv3)

            m_output = pool4

        # Decoder block 1
        x = UpSampling2D(size=(2, 2))(m_output)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

        # Decoder block 2
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

        # Decoder block 3
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        # Decoder block 4
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        # Final segmentation layer
        x = Conv2D(self.n_classes, (1, 1), activation='softmax')(x)

        model = Model(inputs=input_m, outputs=x)

        return model, self.backbone
