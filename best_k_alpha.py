import importlib
import classifiers as cs
import pandas as pd
#moduleName = 'classifiers'
#importlib.import_module(moduleName)

df = pd.read_csv("./data/fashion-mnist_train.csv")

alphas = [i for i in range(1,8,2)]
ks = [i for i in range(1,8,2)]

dfacc = pd.DataFrame()
dfrec = pd.DataFrame()
dfprec= pd.DataFrame()
dff1  = pd.DataFrame()
for a in alphas:
    listacc = []
    listrec = []
    listprec = []
    listf1 = []
    for k in ks:
        cla = cs.clasificador(df,k,a)
        msg = cla.diagnose()
        listacc.append(msg[0])
        listrec.append(msg[2])
        listprec.append(msg[1])
        listf1.append(msg[3])
    dfacc["alpha = " + str(a)] = pd.Series(listacc)
    dfprec["alpha = " + str(a)]= pd.Series(listprec)
    dfrec["alpha = " + str(a)] = pd.Series(listrec)
    dff1["alpha = " + str(a)]  = pd.Series(listf1)

dfacc.to_csv("accuracy.csv")
dfrec.to_csv("recall.csv")
dfprec.to_csv("precision.csv")
dff1.to_csv("dff1")
