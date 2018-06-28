from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py

from keras.utils import np_utils
from keras import optimizers
import Dataset_loader
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.backend.tensorflow_backend import dtype
from keras.utils.io_utils import HDF5Matrix
from Dataset_loader import getNumSamples

from scipy import misc
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
import customlayers

# Parameters
DATASET_COMMON_FOLDER = '/home/alex/Datasets/Yuan/'
NET_NAME = 'AlexNet'
BATCH_SIZE = 10
EPOCHS = 100
NUMBER_OF_CLASSES = 2
INPUT_FRAME_SIZE = 224


def AlexNet():
    model = Sequential()
    input_shape = (INPUT_FRAME_SIZE, INPUT_FRAME_SIZE, 3)
    model.add(Conv2D(96, kernel_size=(11, 11), activation='relu', input_shape=input_shape, strides=(4,4)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(customlayers.crosschannelnormalization(name="convpool_1"))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(customlayers.crosschannelnormalization(name="convpool_2"))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    return model

def generate_arrays(train_filename, batch_size, max_sample, new_size):
    batch_features = np.zeros((batch_size, new_size, new_size, 3))
    batch_labels = np.zeros((batch_size,1))
    
    current_sample_idx = 0
    combined_num = 0

    print('GENERATOR: Train file = {}, batch = {}, total samples = {}'.format(train_filename,  batch_size, max_sample))
    while 1:
        reached_end = False
        start_idx = current_sample_idx
        end_idx = batch_size + start_idx
        
        if (end_idx > max_sample):
            end_idx = batch_size
            reached_end = True
        
        print('GENERATOR: Start idx = {}, end_idx = {}, total samples = {}'.format(start_idx,  end_idx, max_sample))    
        x = HDF5Matrix(train_filename, 'data', start=start_idx, end=end_idx)
        y = HDF5Matrix(train_filename, 'labels', start=start_idx, end=end_idx)
        x = np.array(x)
        y = np.array(y)
        y = np_utils.to_categorical(y, NUMBER_OF_CLASSES)
        
        current_sample_idx = end_idx
        if reached_end:
            current_sample_idx = 0
        
        print("Shapes. x = {}, y = {}".format(x.shape, y.shape))
        
        #batch_labels = np_utils.to_categorical(batch_labels, NUMBER_OF_CLASSES)
        yield(x,y)
        
       
       
variant1 = ['set1_{}.h5'.format(INPUT_FRAME_SIZE), 'set2_{}.h5'.format(INPUT_FRAME_SIZE)]
variant2 = ['set2_{}.h5'.format(INPUT_FRAME_SIZE), 'set1_{}.h5'.format(INPUT_FRAME_SIZE)]
variant3 = ['set3_{}.h5'.format(INPUT_FRAME_SIZE), 'set4_{}.h5'.format(INPUT_FRAME_SIZE)]
variant4 = ['set4_{}.h5'.format(INPUT_FRAME_SIZE), 'set3_{}.h5'.format(INPUT_FRAME_SIZE)]
variants = []

variants.append(variant1)
variants.append(variant2)
variants.append(variant3)
variants.append(variant4)

for num_variant in range(len(variants)):

    TRAIN_SET = DATASET_COMMON_FOLDER + variants[num_variant][0]
    TEST_SET = DATASET_COMMON_FOLDER + variants[num_variant][1]
    
    print('Workings with sets: {} and {}'.format(TRAIN_SET, TEST_SET))
    
    # Load dataset
    x_tr = []
    x_tr = HDF5Matrix(TRAIN_SET, 'data')
    y_tr = HDF5Matrix(TRAIN_SET, 'labels')
    x_train = []
    x_train = np.array(x_tr)
    y_tr = np.array(y_tr)
    x_tr = []
    y_train = np_utils.to_categorical(y_tr, NUMBER_OF_CLASSES)  
    
    x_test = HDF5Matrix(TEST_SET, 'data')
    y_t = HDF5Matrix(TEST_SET, 'labels')
    x_test = np.array(x_test)
    y_t = np.array(y_t)
    
    total_samples_test = getNumSamples(variants[num_variant][1][0:4]+'.h5')
       
    y_test = np_utils.to_categorical(y_t, NUMBER_OF_CLASSES)  
    print('Test dataset loaded')
    print('Testing dataset size = ', x_test.shape)
    
    print('Loading model')
    # Create model
    model = AlexNet()
    print("Compiling model")
    rms_prop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms_prop,
                  metrics=['accuracy', 'mse'])
    print("Model loaded and compiled")
    
    # autosave best Model
    best_model_file = '{}_{}_{}_B{}_E{}_F{}.h5'.format(NET_NAME, TRAIN_SET[-11:-7], TEST_SET[-11:-7], BATCH_SIZE, EPOCHS, INPUT_FRAME_SIZE)
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
    
    print("*************************************")
    print("Fitting model")
    print("*************************************")
    
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              callbacks=[best_model], # this callback can be de-activated
              validation_data=(x_test, y_test))
  
    print("Finished fitting model")
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('All metrics', score)
    
    res = model.predict(x_test)
    res_label = np.argmax(res,1)
    print('\ntest:', sum(res_label==y_t)/float(len(y_t))*100)
     
    res = model.predict(x_train)
    res_label = np.argmax(res,1)
    print('train:', sum(res_label==y_tr)/float(len(y_tr))*100)
    
    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\nFixing the model file error")
    f = h5py.File(best_model_file, 'r+')
    del f['optimizer_weights']
    f.close()
    print("Done\n\n")
 
    print('result on best model')
    print('Loading the best model...')
    model = load_model(best_model_file)
    print('Best Model loaded!')
    res = model.predict(x_test)
    res_label = np.argmax(res,1)
    acc_test = sum(res_label==y_t)/float(len(y_t))*100
    print('test:', sum(res_label==y_t)/float(len(y_t))*100)
     
    res = model.predict(x_train)
    res_label = np.argmax(res,1)
    acc_train = sum(res_label==y_tr)/float(len(y_tr))*100
    print('train:', sum(res_label==y_tr)/float(len(y_tr))*100)
     
    results_filename = '{}_{}_{}_B{}_E{}_F{}.txt'.format(NET_NAME, TRAIN_SET[-11:-7], TEST_SET[-11:-7], BATCH_SIZE, EPOCHS, INPUT_FRAME_SIZE)
    f = open(results_filename, 'wb')
    data = 'Test accuracy = {}\r\nTrain accuracy = {}'.format(acc_test, acc_train)
    f.write(data)
    f.close()
