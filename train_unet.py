from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import argparse
import tifffile as tiff
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 3
N_CLASSES = 2  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
UPCONV = True
PATCH_SZ = 480   # should divide by 16
# TRAIN_SZ = 800  # train size 4000
# VAL_SZ = 250    # validation size 2000


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_lusut_2006.h5'

trainIds = [str(i).zfill(2) for i in range(1, 25)]  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', 
                        type=str,
                        default='./data/mband',
                        help='Folder directory for train images')
    parser.add_argument('--mask', 
                        type=str,
                        default='./data/gt_mband',
                        help='Folder directory for mask images')
    parser.add_argument('-e','--epochs', 
                        type=int,
                        default=200,
                        help='Epoch to run training network')
    parser.add_argument('-b','--batch', 
                        type=int,
                        default=32,
                        help='Batch size for training network')
    parser.add_argument('--trainsize', 
                        type=int,
                        default=4000,
                        help='Batch size for training network')
    parser.add_argument('--validsize', 
                        type=int,
                        default=1000,
                        help='Batch size for training network')
    opt = parser.parse_args()
    print(opt)
    
    N_EPOCHS = opt.epochs
    BATCH_SIZE = opt.batch
    TRAIN_SZ = opt.trainsize  # train size 4000
    VAL_SZ = opt.validsize    # validation size 2000
    
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('[INFO] Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread('{}/{}.tif'.format(opt.images,img_id)).transpose([1, 2, 0]))
        mask = tiff.imread('{}/{}.tif'.format(opt.mask,img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print('*', end="", flush=True)
    print('\n[INFO] Images were read')

    def train_net():
        print("[INFO] Start train UNET")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        # K.clear_session()
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        # tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger], #callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        
        return model
    
    K.clear_session()
    train_net()
