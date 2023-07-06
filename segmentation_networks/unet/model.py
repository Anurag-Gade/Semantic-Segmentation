from utils import *
import_frameworks()


class SegmentationModel:

  def __init__(self, input_dims, num_classes, save_dir):
    self.input_dims = input_dims
    self.num_classes = num_classes
    self.model = self.unet_model()
    self.save_dir = save_dir
    self.weights_save_path = os.path.join(self.save_dir, 'unet_weights.h5')

    # self.default_weights_load_path = "/content/seg_proj/weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

  def unet_model(self):

    input_layer = layers.Input(shape = self.input_dims)

    c1 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
    c1 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(pool_size=(2,2))(c1)

    c2 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(pool_size=(2,2))(c2)

    c3 = layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(pool_size=(2,2))(c3)

    c4 = layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2,2))(c4)

    c5 = layers.Conv2D(filters=1024, kernel_size=(3,3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(filters=1024, kernel_size=(3,3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(filters=512, kernel_size=(2,2), strides=(2,2), padding='same')(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(filters=256, kernel_size=(2,2), strides=(2,2), padding='same')(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(filters=128, kernel_size=(2,2), strides=(2,2), padding='same')(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(filters=64, kernel_size=(2,2), strides=(2,2), padding='same')(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(c9) 

    output_layer = layers.Conv2D(12, 1, 1, activation='softmax')(c9)

    # output_layer = layers.Activation("softmax")(x)

    model = Model(input_layer, output_layer)
    model.summary()


    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
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

    model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),epochs=200,batch_size=16,shuffle=True,verbose=1,callbacks=callbacks)
