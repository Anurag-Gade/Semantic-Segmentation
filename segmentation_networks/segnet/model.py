from utils import *
import_frameworks()

class SegmentationModel:

  def  __init__(self, input_dims, num_classes):
    self.input_dims = input_dims
    self.num_classes = num_classes

  def segnet_model(self):

    input_layer = layers.Input(shape=self.input_dims)

    #SegNet Encoder

    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    #SegNet Decoder

    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.UpSampling2D(size=(2,2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(self.num_classes, kernel_size=(1,1), strides=(1,1), padding="valid")(x)

    flattened = layers.Reshape(((self.input_dims[0])*(self.input_dims[1]), self.num_classes))(x)

    output_layer = layers.Activation("softmax")(x)


    model = Model(input_layer, output_layer)

    return model