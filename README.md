# Semantic Segmentation Pipeline

The segmentation networks implemented in this repository include:

1 . [**SegNet**](https://github.com/Anurag-Gade/brain-labelling-MIT/tree/main/segmentation_networks/segnet)

2 . [**VGGSegNet**](https://github.com/Anurag-Gade/brain-labelling-MIT/tree/main/segmentation_networks/VGG16_SegNet)

3 . [**UNet**](https://github.com/Anurag-Gade/brain-labelling-MIT/tree/main/segmentation_networks/unet)

*More models are being added*

Place the train, validation and testing data into their respective folders in the `.data/CamVid/` directory. Chnage your working directory (using `cd`) to the segmentation model of your choice, and run `main.py`. 

***For VGGSegNet***, change 16 to 19 in `main.py` if you want to use the VGG-19 model as the encoder instead of VGG-16. By default, VGG-16 is the encoder in our VGGSegNet.

```
├── data
│   └── CamVid
│         ├── test
│         ├── test_labels
│         ├── train
│         ├── train_labels
│         ├── val
│         └── val_labels
├── segmentation_networks
|         ├── VGG16_SegNet
|         |      ├── dataloader.py
│         |      ├── main.py
│         |      ├── model.py
│         |      └── utils.py
│         ├── segnet
|         |      ├── dataloader.py
│         |      ├── main.py
│         |      ├── model.py
│         |      └── utils.py
│         └── unet
|                ├── dataloader.py
│                ├── main.py
│                ├── model.py
│                └── utils.py
├── README.md 
```

Some issues need to be addressed, in particular training the model with an IoU metric. If the accuracy metric is replaced with the IoU metric, the training does not go on, and the code runs into an error. Hence, a custom IoU function will be added as well. 

[VGG16 weights (for the VGG16 SegNet)](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)
