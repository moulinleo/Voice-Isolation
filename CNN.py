import tensorflow.contrib.keras.api.keras as K
from tensorflow.contrib.keras.api.keras import backend as Q

# Network Parameters
dropout = 0.5  # Dropout, probability to drop a unit
# Input : 1024 x 64 x samples
input_shape = (1024, 64, 1)

def relu_advanced(x):
    return Q.relu(x, alpha=0.2)

def conv_nn():

    model = K.models.Sequential()

    # (1024x64x1)
    model.add(K.layers.Conv2D(input_shape=input_shape, filters=16, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))
    # (512x32x16)

    model.add(K.layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))
    # (256x16x32)

    model.add(K.layers.Conv2D(filters=64, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))
    # (128x8x64)

    model.add(K.layers.Conv2D(filters=128, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))
    # (64x4x128)
    
    model.add(K.layers.Conv2D(filters=256, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))
    # (32x2x256)
    
    model.add(K.layers.Conv2DTranspose(filters=128, kernel_size=(5,5), padding="same", activation="relu", strides=2))
    model.add(K.layers.Dropout(rate=dropout))
    # (64x4x128)

    model.add(K.layers.Conv2DTranspose(filters=64, kernel_size=(5,5), padding="same", activation="relu", strides=2))
    model.add(K.layers.Dropout(rate=dropout))
    # (128x8x64)

    model.add(K.layers.Conv2DTranspose(filters=32, kernel_size=(5,5), padding="same", activation="relu", strides=2))
    model.add(K.layers.Dropout(rate=dropout))
    # (256x16x32)

    model.add(K.layers.Conv2DTranspose(filters=16, kernel_size=(5,5), padding="same", activation="relu", strides=2))
    # (512x32x16)

    model.add(K.layers.Conv2DTranspose(filters=1, kernel_size=(5,5), padding="same", activation="sigmoid", strides=2))
    # (1024x64x1)
    

    model.compile(optimizer='ADAM',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    print("compiled.")

    # best result with ADAM & MSE & epochs=20 & batch_size (16,32)
    return model

def conv_nn_3lay():

	# 3 layer CNN used for single-genre music prediction
    model.add(K.layers.Conv2D(input_shape=input_shape, filters=16, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))

    model.add(K.layers.Conv2D(filters=32, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))

    model.add(K.layers.Conv2D(filters=64, kernel_size=(5,5), padding="same", activation=relu_advanced, strides=2))
    
    model.add(K.layers.Conv2DTranspose(filters=32, kernel_size=(5,5), padding="same", activation="relu", strides=2))
    model.add(K.layers.Dropout(rate=dropout))

    model.add(K.layers.Conv2DTranspose(filters=16, kernel_size=(5,5), padding="same", activation="relu", strides=2))

    model.add(K.layers.Conv2DTranspose(filters=1, kernel_size=(5,5), padding="same", activation="sigmoid", strides=2))
    
    model.compile(optimizer='ADAM',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    print("compiled.")
	return model
