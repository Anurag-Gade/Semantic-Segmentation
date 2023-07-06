from utils import *
from dataloader import *
from models import *

import_frameworks()

X_train, y_train, X_val, y_val, X_test, y_test = dataloader()

segmod = SegmentationModel(input_dims, num_classes, '../weights/', 16)

segmod.segnet_fit()
