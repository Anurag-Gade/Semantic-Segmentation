from utils import *
import_frameworks()

def dataloader():
  X_train = []
  y_train = []
  X_test = []
  y_test = []

  print("---------------BEGIN DATALOADER-------------------")

  train_images_path = "../data/CamVid/train/"
  datapath = os.path.join(train_images_path,'*g') 
  files = glob.glob(datapath)
  files = natsorted(files, key = lambda x: x.lower())
  for i in files:
    
    filepath = train_images_path + i
    img = np.asarray(Image.open(i))
    img = np.resize(img, (360,480,3))
    X_train.append(img)

  print("X_train loaded!")

  train_masks_path = "../data/CamVid/train_labels/"
  datapath = os.path.join(train_masks_path,'*g') 
  files = glob.glob(datapath)
  files = natsorted(files, key = lambda x: x.lower()) 
  for i in files:

    filepath = train_images_path + i
    img = np.asarray(Image.open(i))
    # img = np.resize(img, (360,480,3))
    y_train.append(img) 

  print("y_train loaded!")

  test_images_path = "../data/CamVid/test/"
  datapath = os.path.join(test_images_path,'*g') 
  files = glob.glob(datapath)
  files = natsorted(files, key = lambda x: x.lower()) 
  for i in files:

    filepath = train_images_path + i
    img = np.array(Image.open(i))
    img = np.resize(img, (360,480,3))
    X_test.append(img) 

  print("X_test loaded!")

  test_masks_path = "../data/CamVid/test_labels/"
  datapath = os.path.join(test_masks_path,'*g') 
  files = glob.glob(datapath)
  files = natsorted(files, key = lambda x: x.lower()) 
  for i in files:

    filepath = train_images_path + i
    img = np.asarray(Image.open(i))
    # img = np.resize(img, (360,480,3))
    y_test.append(img) 

  print("y_test loaded!")

  X_train = np.array(X_train)
  y_train = np.array(y_train)
  X_test = np.array(X_test)
  y_test = np.array(y_test) 

  y_train = y_train.reshape((y_train.shape[0], X_train.shape[1]*X_train.shape[2], 12))
  y_test = y_test.reshape((y_test.shape[0], X_test.shape[1]*X_test.shape[2], 12))

  return X_train, y_train, X_test, y_test