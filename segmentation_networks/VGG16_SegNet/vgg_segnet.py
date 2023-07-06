class SegmentationModel:

  def __init__(self, input_dims, num_classes, save_dir, vgg_type):
    self.input_dims = input_dims
    self.num_classes = num_classes
    self.model = self.vgg16segnet_model()
    self.save_dir = save_dir
    self.weights_save_path = os.path.join(self.save_dir, 'vggseg_weights.h5')

    self.default_weights_load_path = "/content/seg_proj/weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

    self.vgg_type = vgg_type

  def vgg16segnet_model(self):

    input_layer = layers.Input(shape = self.input_dims)

    if int(vgg_type) == 16:
      vgg_model = VGG16(include_top = False, input_tensor = input_layer, weights = self.default_weights_load_path)(input_layer)
      vgg_model.trainable = False

    else if int(vgg_type) == 19:
      vgg_model = VGG19(include_top = False, input_tensor = input_layer, weights = self.default_weights_load_path)(input_layer)
      vgg_model.trainable = False

    x = layers.ZeroPadding2D(padding = (1,1))(vgg_model.ouptut)
    x = layers.Conv2D(filters = 512, kenrel_size = (3,3), padding = "valid")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.ZeroPadding2D(padding = (1,1))(x)
    x = layers.Conv2D(filters = 512, kernel_size = (3,3), padding = "valid")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.ZeroPadding2D(padding = (1,1))(x)
    x = layers.Conv2D(filters = 256, kernel_size = (3,3), padding = "valid")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.ZeroPadding2D(padding = (1,1))(x)
    x = layers.Conv2D(filters = 128, kernel_size = (3,3), padding = "valid")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(size = (2,2))(x)
    x = layers.ZeroPadding2D(padding = (1,1))(x)
    x = layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "valid")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(self.num_classes, kernel_size = (3,3), padding = "same")(x)
    
    output_layer = layers.Activation("softmax")(x)

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
