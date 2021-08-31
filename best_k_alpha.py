import importlib
import classifiers as cs
import pandas as pd
import time
#moduleName = 'classifiers'
#importlib.import_module(moduleName)

df = pd.read_csv("./data/fashion-mnist_train.csv")

alphas = [i for i in range(1,50,2)]
ks = [i for i in range(1,50,2)]


dfacc = pd.DataFrame()
dfrec = pd.DataFrame()
dfprec= pd.DataFrame()
dff1  = pd.DataFrame()
dftime = pd.DataFrame()
for a in alphas:
    
    listacc = []
    listrec = []
    listprec = []
    listf1 = []
    listtime = []
    for k in ks:
        start = time.perf_counter()
        
        cla = cs.clasificador(df,k,a)
        msg = cla.diagnose()
        
        end = time.perf_counter()

        
        listacc.append(msg[0])
        listrec.append(msg[2])
        listprec.append(msg[1])
        listf1.append(msg[3])
        listtime.append(msg[4])
        
    dfacc["alpha = " + str(a)] = pd.Series(listacc)
    dfprec["alpha = " + str(a)]= pd.Series(listprec)
    dfrec["alpha = " + str(a)] = pd.Series(listrec)
    dff1["alpha = " + str(a)]  = pd.Series(listf1)
    dftime["alpha = " + str(a)] = pd.Series(listtime)
dfacc["idx"] = ["k = " + str(i) for i in ks]
dfprec["idx"] = ["k = " + str(i) for i in ks]
dfrec["idx"] = ["k = " + str(i) for i in ks]
dff1["idx"] = ["k = " + str(i) for i in ks]
dftime["idx"] = ["k = " + str(i) for i in ks]
 


dfacc.set_index("idx", inplace=True)
dfprec.set_index("idx", inplace=True)
dfrec.set_index("idx", inplace=True)
dff1.set_index("idx", inplace=True)
dftime.set_index("idx", inplace=True)


dfacc.to_csv("accuracy.csv")
dfrec.to_csv("recall.csv")
dfprec.to_csv("precision.csv")
dff1.to_csv("f1.csv")
dftime.to_csv("time.csv")
