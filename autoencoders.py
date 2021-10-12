import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




class autoencoder:
    "Given a sequence of dimensions, constructs a deep autoencoder with their respective dimensions and the indicated activation function. The autoencoder is useful only for the FASHION MNIST dataset"
    def __init__(self, hiddenDims, latentDim, hiddenActivation):
        self.hDims = hiddenDims # a sequence with the dimensions of the hidden layers, the middle layer dimension should be the same as latentDim
        self.lDim = latentDim # the dimension of the bottleneck layer
        self.activ = hiddenActivation #hidden activation function used
        if(hiddenDims[len(hiddenDims)/2] != latentDim):
            print("Error: bottleneck dimension does not match ")
        self.autoencoder = None
        self.encoder = None
        self.X_train = None
        self.X_val = None
        self.X_train_red = None
        self.X_val_red = None
    def construct(self):
        input_img = keras.Input(shape(784,))
        hLayers = []
        encoded = None
        for(k in range(len(self.hiddenDims))):
            if(k==0):
                firtLayer = layers.Dense(hiddenDims[0], activation = self.activ)(input_img))
                hLayers.push(firstLayer)
            else:
                nextLayer = layers.Dense(hiddenDims[k], activation = self.activ)(hLayers[-1])
                hlayers.push(nextLayer)
            if(hiddenDims[k]==self.lDim):
                encoded = hLayers[-1]
        
        output_img = keras.Dense(784, activation = "sigmoid")(hLayers[-1])
        self.encoder = keras.Model(input_img, encoded)
        self.autoencoder = keras.Model(input_img, output_img)
        self.autoencoder.compile(optimizer='adam', loss = "binary_crossentropy")


    def trainAndReduce(self, X_train, X_val):
        X_train = X_train.astype('float32'/255.0)
        X_val = X_val.astype('float32'/255.0)
        
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        X_val = X_val.reshape((len(X_val), np.prod(X_val.shape[1:])))

        self.X_train = X_train
        self.X_val = X_val
        self.autoencoder.fit(X_train,y_train, epochs=50, batch_size = 256, shuffle=True, validation_data=(X_val,y_val))

        self.X_train_red = self.autoencoder.predict(X_train)
        self.X_val_red = self.autoencoder.predict(X_val)

        
        
    def diagnose(self, y_train, y_val, n_neigh):
        knn = KNeighborsClassifier(n_neighbors = n_neigh).fit(self.X_train_red, y_train)

        preds = knn.predict(self.X_val_red)

        print("\n Accuracy score : ", accuracy_score(y_val, preds))
       


