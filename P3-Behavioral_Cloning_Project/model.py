import os
import json
from pathlib import Path

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Lambda, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Activation
from keras.callbacks import ModelCheckpoint, CSVLogger

import config as cf
from proc_data import train_validation_split, load_data
from proc_data import generate_train_data_batch, generate_valid_data


def get_model():
    
    input_shape = (cf.IN_Height, cf.IN_Width, cf.IN_Channels)
    
    init = 'glorot_uniform'
    
    model = Sequential()
    
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape))
    
    model.add(Convolution2D(16, 3, 3, border_mode='valid', init=init))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Convolution2D(32, 3, 3, border_mode='valid', init=init))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    
    model.add(Convolution2D(48, 3, 3, border_mode='valid', init=init))
    model.add(Activation('elu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, init=init))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, init=init))
    model.add(Activation('elu'))
    
    model.add(Dense(16, init=init))
    model.add(Activation('elu'))
    
    model.add(Dense(1, init=init))
    
    return model

    
# Save model to json and weights to file
def save_model(model, json_file, weight_file):
    if Path(json_file).is_file():
        os.remove(json_file)
    if Path(weight_file).is_file():
        os.remove(weight_file)
    # save model to JSON
    json_string = model.to_json()
    with open(json_file, 'w') as f:
        json.dump(json_string, f)
    # save weights to HDF5
    model.save_weights(weight_file)


def train_model():
    model_json = './model.json'
    model_weights = './model.h5'
    
    batch_size = 256
    epochs = 10
   
    data_train, data_val = train_validation_split(load_data(cf.DRIVING_LOG))

    model = get_model()
    adam = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
    model.summary()
       
 # json dump of model architecture
    with open('./logs/model.json', 'w') as f:
        f.write(model.to_json())

    # callbacks to save history and weights
    checkpointer = ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='./logs/history.csv')
    
    tain_generator = generate_train_data_batch(data_train, batch_size, bias=cf.BIAS)
    val_generator = generate_valid_data(data_val)
        
    # start training
    model.fit_generator(tain_generator,
                        samples_per_epoch= 79*batch_size,#20000
                        nb_epoch=epochs,
                        validation_data=val_generator,
                        nb_val_samples=len(data_val)),
#                        callbacks=[checkpointer, logger]) # added to log
    
    print('Model saving...')
    save_model(model, model_json, model_weights)
    print('Done')
            

if __name__ == '__main__':
    
    train_model()
