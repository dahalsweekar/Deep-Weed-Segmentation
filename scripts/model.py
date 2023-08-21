from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda, Activation
from keras.applications import VGG16, InceptionV3, MobileNetV2, DenseNet121, ResNet50, EfficientNetB0
import segmentation_models as sm
import keras
import tensorflow as tf


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
        elif self.backbone == 'efficientnetb0':
            # Load inceptionV3 without the classification layers (include_top=False)
            weights = EfficientNetB0(weights='imagenet', include_top=False,
                                     input_shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))
            # Retrieve the InceptionV3's first layer
            input_m = weights.layers[0].input
            # Retrieve the InceptionV3's last layer (usually mixed7 layer)
            m_output = weights.get_layer('block5a_expand_activation').output
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
            weights = DenseNet121(weights='imagenet', include_top=False,
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

            print('building with segnet encoder...')
            # Encode Block
            input_m = Input(shape=(self.PATCH_SIZE, self.PATCH_SIZE, self.IMG_CHANNELS))

            # Encoder block 1
            conv1 = Conv2D(64, (3, 3), padding='same')(input_m)
            conv1 = BatchNormalization()(conv1)  # Add BatchNormalization
            conv1 = Activation("relu")(conv1)  # Add ReLU activation
            conv2 = Conv2D(64, (3, 3), padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)  # Add BatchNormalization
            conv2 = Activation("relu")(conv2)  # Add ReLU activation
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

            # Encoder block 2
            conv1 = Conv2D(128, (3, 3), padding='same')(pool1)
            conv1 = BatchNormalization()(conv1)  # Add BatchNormalization
            conv1 = Activation("relu")(conv1)  # Add ReLU activation

            conv2 = Conv2D(128, (3, 3), padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)  # Add BatchNormalization
            conv2 = Activation("relu")(conv2)  # Add ReLU activation
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            # Encoder block 3
            conv1 = Conv2D(256, (3, 3), padding='same')(pool2)
            conv1 = BatchNormalization()(conv1)  # Add BatchNormalization
            conv1 = Activation("relu")(conv1)  # Add ReLU activation
            conv2 = Conv2D(256, (3, 3), padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)  # Add BatchNormalization
            conv2 = Activation("relu")(conv2)  # Add ReLU activation
            conv3 = Conv2D(256, (3, 3), padding='same')(conv2)
            conv3 = BatchNormalization()(conv3)  # Add BatchNormalization
            conv3 = Activation("relu")(conv3)  # Add ReLU activation
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            # Encoder block 4
            conv1 = Conv2D(512, (3, 3), padding='same')(pool3)
            conv1 = BatchNormalization()(conv1)  # Add BatchNormalization
            conv1 = Activation("relu")(conv1)  # Add ReLU activation
            conv2 = Conv2D(512, (3, 3), padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)  # Add BatchNormalization
            conv2 = Activation("relu")(conv2)  # Add ReLU activation
            conv3 = Conv2D(512, (3, 3), padding='same')(conv2)
            conv3 = BatchNormalization()(conv3)  # Add BatchNormalization
            conv3 = Activation("relu")(conv3)  # Add ReLU activation
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv3)

            m_output = pool4

        # Decoder block 1
        x = UpSampling2D(size=(2, 2))(m_output)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation

        # Decoder block 2
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation

        # Decoder block 3
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation

        # Decoder block 4
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("relu")(x)  # Add ReLU activation

        # Final segmentation layer
        x = Conv2D(self.n_classes, (1, 1))(x)
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Activation("softmax")(x)  # Add ReLU activation

        model = Model(inputs=input_m, outputs=x)

        return model, self.backbone

    def deeplabv3(self, name="ResNet50", weights="imagenet", height=None, width=None,
                  channels=3, include_top=False, pooling=None, alpha=1.0,
                  depth_multiplier=1, dropout=0.001):
        if not isinstance(height, int) or not isinstance(width, int) or not isinstance(channels, int):
            raise TypeError(
                "'height', 'width' and 'channels' need to be of type 'int'")

        if channels <= 0:
            raise ValueError(
                f"'channels' must be greater of equal to 1 but given was {channels}")

        input_shape = [height, width, channels]

        if name.lower() == "densenet121":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.DenseNet121(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1/relu", "pool2_relu",
                           "pool3_relu", "pool4_relu", "relu"]
        elif name.lower() == "densenet169":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.DenseNet169(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1/relu", "pool2_relu",
                           "pool3_relu", "pool4_relu", "relu"]
        elif name.lower() == "densenet201":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.DenseNet201(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1/relu", "pool2_relu",
                           "pool3_relu", "pool4_relu", "relu"]
        elif name.lower() == "efficientnetb0":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb1":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB1(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb2":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB2(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb3":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB3(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb4":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB4(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb5":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB5(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb6":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB6(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "efficientnetb7":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.EfficientNetB7(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2a_expand_activation", "block3a_expand_activation",
                           "block4a_expand_activation", "block6a_expand_activation", "top_activation"]
        elif name.lower() == "mobilenet":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.MobileNet(include_top=include_top, weights=weights,
                                                         input_shape=input_shape,
                                                         pooling=pooling, alpha=alpha,
                                                         depth_multiplier=depth_multiplier, dropout=dropout)
            layer_names = ["conv_pw_1_relu", "conv_pw_3_relu",
                           "conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"]
        elif name.lower() == "mobilenetv2":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.MobileNetV2(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling, alpha=alpha)
            layer_names = ["block_1_expand_relu", "block_3_expand_relu",
                           "block_6_expand_relu", "block_13_expand_relu", "out_relu"]
        elif name.lower() == "mobilenetv3small":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.MobileNetV3Small(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling, alpha=alpha)
            layer_names = ["multiply", "re_lu_3",
                           "multiply_1", "multiply_11", "multiply_17"]
        elif name.lower() == "nasnetlarge":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.NASNetLarge(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["zero_padding2d", "cropping2d_1",
                           "cropping2d_2", "cropping2d_3", "activation_650"]
        elif name.lower() == "nasnetmobile":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.NASNetMobile(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["zero_padding2d", "cropping2d_1",
                           "cropping2d_2", "cropping2d_3", "activation_187"]
        elif name.lower() == "resnet50":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.ResNet50(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1_relu", "conv2_block3_out",
                           "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
        elif name.lower() == "resnet50v2":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.ResNet50V2(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1_conv", "conv2_block3_preact_relu",
                           "conv3_block4_preact_relu", "conv4_block6_preact_relu", "post_relu"]
        elif name.lower() == "resnet101":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.ResNet101(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1_relu", "conv2_block3_out",
                           "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
        elif name.lower() == "resnet101v2":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.ResNet101V2(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1_conv", "conv2_block3_preact_relu",
                           "conv3_block4_preact_relu", "conv4_block23_preact_relu", "post_relu"]
        elif name.lower() == "resnet152":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.ResNet152(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1_relu", "conv2_block3_out",
                           "conv3_block8_out", "conv4_block36_out", "conv5_block3_out"]
        elif name.lower() == "resnet152v2":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.ResNet152V2(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["conv1_conv", "conv2_block3_preact_relu",
                           "conv3_block8_preact_relu", "conv4_block36_preact_relu", "post_relu"]
        elif name.lower() == "vgg16":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.VGG16(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2_conv2", "block3_conv3",
                           "block4_conv3", "block5_conv3", "block5_pool"]
        elif name.lower() == "vgg19":
            if height <= 31 or width <= 31:
                raise ValueError(
                    "Parameters 'height' and 'width' should not be smaller than 32.")
            base_model = tf.keras.applications.VGG19(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2_conv2", "block3_conv4",
                           "block4_conv4", "block5_conv4", "block5_pool"]
        elif name.lower() == "xception":
            if height <= 70 or width <= 70:
                raise ValueError(
                    "Parameters 'height' and width' should not be smaller than 71.")
            base_model = tf.keras.applications.Xception(
                include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
            layer_names = ["block2_sepconv2_act", "block3_sepconv2_act",
                           "block4_sepconv2_act", "block13_sepconv2_act", "block14_sepconv2_act"]
        else:
            raise ValueError("'name' should be one of 'densenet121', 'densenet169', 'densenet201', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', \
                        'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7','mobilenet', 'mobilenetv2', 'mobilenetv3small', 'nasnetlarge', 'nasnetmobile', \
                        'resnet50', 'resnet50v2', 'resnet101', 'resnet101v2', 'resnet152', 'resnet152v2', 'vgg16', 'vgg19' or 'xception'.")

        layers = [base_model.get_layer(
            layer_name).output for layer_name in layer_names]

        return base_model, layers, layer_names
