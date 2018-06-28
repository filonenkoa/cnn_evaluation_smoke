from __future__ import print_function
import numpy as np
import tensorflow as tf
import h5py

from keras.applications.resnet50 import ResNet50

# from keras.models import Sequential
# from keras.layers import advanced_activations
# from keras.layers.pooling import MaxPooling2D
#from keras.optimizers import RMSprop, Adadelta, Adam
#from keras.layers.convolutional import Convolution2D
#from keras.layers.core import Dense, Activation, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils
#from keras import backend as K
from keras import optimizers
#from keras.preprocessing.image import ImageDataGenerator
import Dataset_loader

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.backend.tensorflow_backend import dtype

from keras.utils.io_utils import HDF5Matrix
from Dataset_loader import getNumSamples

from scipy import misc

# Parameters
DATASET_COMMON_FOLDER = '/home/alex/Datasets/Yuan/'
NET_NAME = 'ResNet50'
BATCH_SIZE = 10
EPOCHS = 10
NUMBER_OF_CLASSES = 2
INPUT_FRAME_SIZE = 224




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
    print(y_tr[0])
    x_train = []
    #x_train = np.array(x_tr)
    #y_tr = np.array(y_tr)
    #x_tr = []
    y_train = np_utils.to_categorical(y_tr, NUMBER_OF_CLASSES)  
    
    
    x_test = HDF5Matrix(TEST_SET, 'data')
    y_t = HDF5Matrix(TEST_SET, 'labels')
    x_test = np.array(x_test)
    y_t = np.array(y_t)
    
    
    
    
    total_samples_test = getNumSamples(variants[num_variant][1][0:4]+'.h5')
    
    #x_test = np.zeros((total_samples_test, INPUT_FRAME_SIZE, INPUT_FRAME_SIZE, 3), dtype='float16')
    #y_test = np.zeros((total_samples_test,1), dtype='float16')
   
    y_test = np_utils.to_categorical(y_t, NUMBER_OF_CLASSES)  
    print('Test dataset loaded')
    
    print('Testing dataset size = ', x_test.shape)
    # Convert class vectors to binary class matrices
    
    print('Loading model')
    model = ResNet50(weights=None, include_top=True, classes=NUMBER_OF_CLASSES)
    rms_prop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rms_prop,
                  metrics=['accuracy', 'mse'])
    print("Model loaded")
    
    # autosave best Model
    best_model_file = '{}_{}_{}_B{}_E{}_F{}.h5'.format(NET_NAME, TRAIN_SET[-11:-7], TEST_SET[-11:-7], BATCH_SIZE, EPOCHS, INPUT_FRAME_SIZE)
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
    
    print("*************************************")
    print("Fitting model")
    print("*************************************")
    
    model.fit(x_tr, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              callbacks=[best_model], # this callback can be de-activated
              validation_data=(x_test, y_test))
      
#     datagen = ImageDataGenerator(
#             featurewise_center=False,  # set input mean to 0 over the dataset
#             samplewise_center=False,  # set each sample mean to 0
#             featurewise_std_normalization=False,  # divide inputs by std of the dataset
#             samplewise_std_normalization=False,  # divide each input by its std
#             zca_whitening=False,  # apply ZCA whitening
#             rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#             width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
#             height_shift_range=0,  # randomly shift images vertically (fraction of total height)
#             horizontal_flip=False,  # randomly flip images
#             vertical_flip=False)  # randomly flip images
#      
#         # Compute quantities required for feature-wise normalization
#         # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train)
        
#     total_samples_train = getNumSamples(variants[num_variant][0][0:4]+'.h5')
#     model.fit_generator(generate_arrays(TRAIN_SET,
#                                          batch_size=BATCH_SIZE,
#                                          max_sample=total_samples_train,
#                                          new_size=INPUT_FRAME_SIZE),
#                         BATCH_SIZE, EPOCHS,
#                         verbose=2,
#                         callbacks=[best_model],
#                         validation_data=(x_test, y_test))
    print("Finished fitting model")
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('All metrics', score)
    
    
    
    #x_train = HDF5Matrix(TRAIN_SET, 'data')
    #y_train = HDF5Matrix(TRAIN_SET, 'labels')
    
    
    res = model.predict(x_test)
    res_label = np.argmax(res,1)
    print('\ntest:', sum(res_label==y_t)/float(len(y_t))*100)
    
    
    res = model.predict(x_t)
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
    
    res = model.predict(x_tr)
    res_label = np.argmax(res,1)
    acc_train = sum(res_label==y_tr)/float(len(y_tr))*100
    print('train:', sum(res_label==y_tr)/float(len(y_tr))*100)
    
    results_filename = '{}_{}_{}_B{}_E{}_F{}.txt'.format(NET_NAME, TRAIN_SET[-11:-7], TEST_SET[-11:-7], BATCH_SIZE, EPOCHS, INPUT_FRAME_SIZE)
    f = open(results_filename, 'wb')
    data = 'Test accuracy = {}\r\nTrain accuracy = {}'.format(acc_test, acc_train)
    f.write(data)
    f.close()