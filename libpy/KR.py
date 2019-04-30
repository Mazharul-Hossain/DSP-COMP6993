from __future__ import absolute_import, division, print_function

import datetime
import numpy, os, pandas 

import tensorflow
from tensorflow import keras

path = "./training_2/"

def build_model(train_data, units=64, depth=2):
    
    model = keras.Sequential([
        keras.layers.Dense(units, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tensorflow.nn.relu, input_shape=(train_data.shape[1],) )
    ])

    for i in range(depth) :
        model.add( keras.layers.Dense(units, kernel_regularizer=keras.regularizers.l2(0.001),
                           activation=tensorflow.nn.relu)
            )

    model.add(    keras.layers.Dense(1 ) ) # , activation=tensorflow.nn.relu) )

    optimizer = tensorflow.train.RMSPropOptimizer(0.001)
    
    # metrics=['mse', 'mae', 'mape', 'cosine']
    model.compile(loss='mse', optimizer=optimizer, metrics=[ 'mae' ])

    return model


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def get_cp_callback(checkpoint=""):

    global path
    
    ts = str( int( datetime.datetime.now().timestamp() ) )

    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = path + checkpoint + "-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create checkpoint callback
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint( checkpoint_dir, verbose=0, save_weights_only=True # )
    # Save weights, every 5-epochs.
    , period=5)
    
    return cp_callback


def get_early_stop(patience=20):
	
	# The patience parameter is the amount of epochs to check for improvement
	early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

	return early_stop



def train_model(model, train_data, train_labels, EPOCHS=500, callbacks = [] ) :
    
    # cp_callback = get_cp_callback()
    early_stop = get_early_stop()

    # callbacks=[ cp_callback, early_stop, PrintDot() ]
    callbacks.extend( [ early_stop, PrintDot() ] )
    
    # Store training stats
    history = model.fit( train_data, train_labels, epochs=EPOCHS, 
                    validation_split=0.2, verbose=0,
                    callbacks = callbacks )
    
    return history, model