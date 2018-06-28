# Working with dataset for smoke test_flag_detection
import numpy as np
from scipy import misc
from scipy.ndimage import imread
from os import listdir
import os.path
import sys
import random
import keras.preprocessing.image as im
import pickle
from os import walk # For getting list of files
from fileinput import filename
#from scipy.misc import pilutil
import PIL

from keras.utils.io_utils import HDF5Matrix

DATASET_PATH_TRAINING = '/home/alex/Datasets/Yuan/set1/'
DATASET_PATH_TESTING = '/home/alex/Datasets/Yuan/set2/'

LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0

# Opens the image and reads its dimensions
# Based on assumption that image is squared
def getFrameSize(filePath):
    print('Getting frame size of file {}'.format(filePath))
    img = imread(filePath)
    size = img.shape[1]
    return size

# Reads all files in the folder and returns all of them as an array
# of the 2nd parameter is true, then only image files will be added
def getListOfFiles(path, validate_image=True):
    f = []
    result = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        
    # Making the fullpath and filtering process
    for i in range(len(filenames)):
        if (
            validate_image == True
            and not (filenames[i].lower().endswith('.png') or filenames[i].lower().endswith('.jpg'))
            ):
            continue
        result.append(path + filenames[i])
        #print ("Added {}".format(path + filenames[i]))
    return result

def create_dataset(path_train='/home/alex/Datasets/Yuan/set1/', path_test='/home/alex/Datasets/Yuan/set2/', frame_size=224):
    print("*************************************")
    print("Making dataset")
    print("*************************************")
    import h5py
    
    # Get correct paths
    path_train_pos = path_train + 'smoke/'
    path_train_neg = path_train + 'non/'
    path_test_pos = path_test + 'smoke/'
    path_test_neg = path_test + 'non/'
    
    # Get list of files
    print("\nGetting list of files")
    files_train_pos = getListOfFiles(path_train_pos)
    files_train_neg = getListOfFiles(path_train_neg)
    files_test_pos = getListOfFiles(path_test_pos)
    files_test_neg = getListOfFiles(path_test_neg)
    
    # Declare containers
    labels_train = []
    labels_test = []
    data_train = []
    data_test = []
    
    # Fill containers
    print("\nFilling containers")
    need_resize = True
    file_size = getFrameSize(files_train_pos[0])
#     if file_size == frame_size:
#         need_resize = False
    
    for i in range(len(files_train_pos)):
        img = imread(files_train_pos[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_train.append(img)
        labels_train.append(1)
        
    for i in range(len(files_train_neg)):
        img = imread(files_train_neg[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_train.append(img)
        labels_train.append(0)
        
    for i in range(len(files_test_pos)):
        img = imread(files_test_pos[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_test.append(img)
        labels_test.append(1)
        
    for i in range(len(files_test_neg)):
        img = imread(files_test_neg[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_test.append(img)
        labels_test.append(0)
        
    
    print("\nConverting images to numpy arrays")
    x_train = np.zeros((len(labels_train),frame_size,frame_size,3), dtype='uint8')
    y_train = np.empty(len(labels_train))
    x_test = np.zeros((len(labels_test),frame_size,frame_size,3), dtype='uint8')
    y_test = np.empty(len(labels_test))
    
    print('Output containers\nx_train={}\ny_train={}\nx_test={}\ny_test={}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    
    for i in range(x_train.shape[0]):
        #print("i={} of {}".format(i, x_train.shape[0]))
        x_train[i,:,:,:] = data_train[i]
        y_train[i] = labels_train[i]
    
    for i in range(x_test.shape[0]):
        #print("i={} of {}".format(i, x_test.shape[0]))
        x_test[i,:,:,:] = data_test[i]
        y_test[i] = labels_test[i]
        
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #y_train = y_train.astype('float32')
    x_train /= 255
    x_test /= 255
    #y_train /= 255
     
        
    f = h5py.File('{}_{}.h5'.format(path_train[-5:-1], frame_size), 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('data', x_train.shape, dtype='f')
    X_dset[:] = x_train
    # Creating dataset to store labels
    y_dset = f.create_dataset('labels', y_train.shape, dtype='i')
    y_dset[:] = y_train
    f.close()
    
    
    f1 = h5py.File('{}_{}.h5'.format(path_test[-5:-1], frame_size), 'w')
    # Creating dataset to store features
    X_dset1 = f1.create_dataset('data', x_test.shape, dtype='f')
    X_dset1[:] = x_test
    # Creating dataset to store labels
    y_dset1 = f1.create_dataset('labels', y_test.shape, dtype='i')
    y_dset1[:] = y_test
    f1.close()


    









def loadDataset(path_train, path_test, frame_size):
    print("*************************************")
    print("Loading dataset")
    print("*************************************")
    
    
    
    
    
    
    
    # Get correct paths
    path_train_pos = path_train + 'smoke/'
    path_train_neg = path_train + 'non/'
    path_test_pos = path_test + 'smoke/'
    path_test_neg = path_test + 'non/'
    
    # Get list of files
    print("\nGetting list of files")
    files_train_pos = getListOfFiles(path_train_pos)
    files_train_neg = getListOfFiles(path_train_neg)
    files_test_pos = getListOfFiles(path_test_pos)
    files_test_neg = getListOfFiles(path_test_neg)
    
    # Declare containers
    labels_train = []
    labels_test = []
    data_train = []
    data_test = []
    

    
    # Fill containers
    print("\nFilling containers")
    need_resize = True
    file_size = getFrameSize(files_train_pos[0])
    if file_size == frame_size:
        need_resize = False
    
    for i in range(len(files_train_pos)):
        img = imread(files_train_pos[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_train.append(img)
        labels_train.append(1)
        
    for i in range(len(files_train_neg)):
        img = imread(files_train_neg[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_train.append(img)
        labels_train.append(0)
        
    for i in range(len(files_test_pos)):
        img = imread(files_test_pos[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_test.append(img)
        labels_test.append(1)
        
    for i in range(len(files_test_neg)):
        img = imread(files_test_neg[i])
        if need_resize:
            img = misc.imresize(img, (frame_size, frame_size), 'bicubic')
        data_test.append(img)
        labels_test.append(0)
        
    
    print("\nConverting images to numpy arrays")
    x_train = np.zeros((len(labels_train),frame_size,frame_size,3))
    y_train = np.empty(len(labels_train))
    x_test = np.zeros((len(labels_test),frame_size,frame_size,3))
    y_test = np.empty(len(labels_test))
    
    print('Output containers\nx_train={}\ny_train={}\nx_test={}\ny_test={}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    
    for i in range(x_train.shape[0]):
        #print("i={} of {}".format(i, x_train.shape[0]))
        x_train[i,:,:,:] = data_train[i]
        y_train[i] = labels_train[i]
    
    for i in range(x_test.shape[0]):
        #print("i={} of {}".format(i, x_test.shape[0]))
        x_test[i,:,:,:] = data_test[i]
        y_test[i] = labels_test[i]
      
    return (x_train, y_train),(x_test, y_test)

def getNumSamples(set_name):
    if (set_name == 'set1.h5'):
        return 1385
    elif (set_name == 'set2.h5'):
        return 1507
    elif (set_name == 'set3.h5'):
        return 10715
    elif (set_name == 'set4.h5'):
        return 10619
    else:
        return -1

if __name__ == "__main__":
    print("Generating datasets binaries with resizing")
    create_dataset(path_train='/home/alex/Datasets/Yuan/set1/', path_test='/home/alex/Datasets/Yuan/set2/', frame_size=224)
    create_dataset(path_train='/home/alex/Datasets/Yuan/set3/', path_test='/home/alex/Datasets/Yuan/set4/', frame_size=224)
    create_dataset(path_train='/home/alex/Datasets/Yuan/set1/', path_test='/home/alex/Datasets/Yuan/set2/', frame_size=299)
    create_dataset(path_train='/home/alex/Datasets/Yuan/set3/', path_test='/home/alex/Datasets/Yuan/set4/', frame_size=299)