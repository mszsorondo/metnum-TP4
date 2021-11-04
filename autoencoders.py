import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses


class autoencoder:
    "Given a sequence of dimensions, constructs a deep autoencoder with their respective dimensions and the indicated activation function. The autoencoder is useful only for the FASHION MNIST dataset"
    print("new")
    def __init__(self, hiddenDims, latentDim, hiddenActivation):
        # a sequence with the dimensions of the hidden layers, the middle layer dimension should be the same as latentDim
        self.hDims = hiddenDims
        self.lDim = latentDim  # the dimension of the bottleneck layer
        self.activ = hiddenActivation  # hidden activation function used
        if(hiddenDims[int(len(hiddenDims)/2)] != latentDim):
            print("Error: bottleneck dimension does not match ")
        self.autoencoder = None
        self.encoder = None
        self.X_train = None
        self.X_val = None
        self.X_train_red = None
        self.X_val_red = None
        self.classifierTimePerformances = []
        self.reductionTimePerformance = None
        self.classifierAccuracyPerformances = []

    def construct(self):
        input_img = keras.Input(shape=(784,))
        hLayers = []
        encoded = None
        for k in range(len(self.hDims)):
            if(k == 0):
                firstLayer = layers.Dense(
                    self.hDims[0], activation=self.activ)(input_img)
                hLayers.append(firstLayer)
            else:
                nextLayer = layers.Dense(
                    self.hDims[k], activation=self.activ)(hLayers[-1])
                hLayers.append(nextLayer)
            if(self.hDims[k] == self.lDim):
                encoded = hLayers[-1]
                #self.activ = 'sigmoid'

        output_img = layers.Dense(784, activation="sigmoid")(hLayers[-1])
        self.encoder = keras.Model(input_img, encoded)
        self.autoencoder = keras.Model(input_img, output_img)
        self.autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    def trainAndReduce(self, X_train, X_val):
        X_train = (X_train.astype('float32') / 255.)
        X_val = (X_val.astype('float32') / 255.)

        self.X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        self.X_val = X_val.reshape((len(X_val), np.prod(X_val.shape[1:])))
        train_st = time.time()
        self.autoencoder.fit(self.X_train, self.X_train, epochs=50, batch_size=256, 
        shuffle=True,validation_data=(self.X_val, self.X_val))
        train_end = time.time()
        train_dif = train_end-train_st
        print("\nautoencoder training time: ", train_dif,"\n")
        self.reductionTimePerformance = train_dif
        self.X_train_red = self.autoencoder.predict(self.X_train)
        self.X_val_red = self.autoencoder.predict(self.X_val)
        self.X_train = None
        self.X_val = None

    def diagnose(self, y_train, y_val, n_neigh):
        knn_st = time.time()
        knn = KNeighborsClassifier(n_neighbors=n_neigh).fit(
            self.X_train_red, y_train)
    
        preds = knn.predict(self.X_val_red)
        knn_end = time.time()
        knn_dif = knn_end - knn_st
        print("\n KNN training+prediction time: ", knn_dif, " , k = ",n_neigh,"\n")
        self.classifierTimePerformances.append(knn_dif)
        self.classifierAccuracyPerformances.append(accuracy_score(y_val, preds))
        print("\n Accuracy score : ", accuracy_score(y_val, preds))

