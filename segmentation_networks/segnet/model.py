from utils import *
from custom_layers import *
import_frameworks()

class SegmentationModel:

  def  __init__(self, input_dims, num_classes, save_dir):
    self.input_dims = input_dims
    self.num_classes = num_classes
    self.model = self.segnet_model()
    self.save_dir = save_dir
    self.weights_save_path = os.path.join(self.save_dir, 'seg_weights.h5')

  def segnet_model(self):

    input_layer = layers.Input(shape=self.input_dims)

    #SegNet Encoder

    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p1, m1 = MaxPoolingWithArgmax2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p2, m2 = MaxPoolingWithArgmax2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p3, m3 = MaxPoolingWithArgmax2D(pool_size=(2,2))(x)

    x = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p4, m4 = MaxPoolingWithArgmax2D(pool_size=(2,2))(x)

    #SegNet Decoder

    x = MaxUnpooling2D((2,2))([p4,m4])
    x = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=512, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = MaxUnpooling2D((2,2))([x,m3])
    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = MaxUnpooling2D((2,2))([x,m2])
    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = MaxUnpooling2D((2,2))([x,m1])
    x = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.Conv2D(filters=self.num_classes, kernel_size=(1,1), strides=(1,1), padding="same")(x)
    # # x = layers.BatchNormalization()(x)
    # x = layers.Reshape((self.input_dims[0]*self.input_dims[1], self.num_classes))
    # output_layer = layers.Activation("relu")(x)

    # x = layers.Reshape((-1, self.num_classes))(x)
    x = layers.Conv2D(12, 1, 1, padding="valid")(x)
    # x = layers.BatchNormalization()(x)
    output_layer = layers.Activation("softmax")(x)

    # flattened = layers.Reshape((self.input_dims[0] * self.input_dims[1], self.num_classes),
    #       input_shape=(self.input_dims[0], self.input_dims[1], self.num_classes),
    #       )(x)

    # output_layer = layers.Activation("softmax")(x)


    model = Model(input_layer, output_layer)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]) 
    return model

  def segnet_fit(self):
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
      self.weights_save_path,
      monitor='val_loss',
      verbose=0,
      save_best_only=False,
      save_weights_only=False,
      mode='auto',
      period=50))
  
    # class_weighting = {0:0.2595, 1:0.1826, 2:4.5640, 3:0.1417, 4:0.9051, 5:0.3826, 6:9.6446, 7:1.8418, 8:6.6823, 9:6.2478, 10:3.0, 11:7.3614}
    class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
    model = self.model
    
    # model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),epochs=200,batch_size=16,shuffle=True,verbose=1,callbacks=callbacks)
