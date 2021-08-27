import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import time
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class clasificador:
    """La idea es que este objeto haga un modelo knn y lo entrene con el dataframe de FASHION MNIST utilizando KNN y PCA"""
    # Acordate de que se pueda medir la performance
    def __init__(self, df, k, alpha):
        self.X = df.iloc[:,1:]
        self.y = df["label"]
        self.k  = k
        self.alpha = alpha
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self.PCAobj = None
        self.model = KNeighborsClassifier(self.k)
    def __preprocessing(self):
        #hace el split de los datos y el PCA
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y)
        self.PCAobj = PCA(self.alpha)
 
    def __fit(self):
        self.PCAobj.fit(self.X_train)
        self.X_train = self.PCAobj.transform(self.X_train)
        self.X_test = self.PCAobj.transform(self.X_test)
        self.model.fit(self.X_train, self.y_train)


    def __predict(self):
        return self.model.predict(self.X_test)

    def diagnose(self):
        self.__preprocessing()
        self.__fit()
        report = pd.DataFrame(classification_report(self.y_test,self.__predict(), output_dict=True))
        acc = report["accuracy"][1]
        precisionAVG = report["macro avg"][0]
        recallAVG = report["macro avg"][1]
        f1ScoreAVG = report["macro avg"][2]

        message = "Accuracy: ", acc, "\nPrecision(avg): ", precisionAVG, "\nRecall: ", recallAVG, "F1avg: ", f1ScoreAVG
        
        print("\n ------------ k = ", self.k, "------------alpha = ", self.alpha, "----------\n")
        print(message)
        return([acc, precisionAVG, recallAVG,f1ScoreAVG])
        """Aca se entrena al modelo con las funciones internas y se hace un reporte entero de la performance"""
        
