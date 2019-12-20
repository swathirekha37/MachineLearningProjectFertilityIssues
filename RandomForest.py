import numpy as np
import pandas as pd
import sklearn


colnames=["Season","Age","Childish diseases","Accident or serious trauma","Surgical intervention","High fevers in the last year","Frequency of alcohol consumption","smoking habit","Num of hours spent sitting per day", "Output diagnosis"]
df=pd.read_csv("C:/Users/Surface/Desktop/forest/fertility_Diagnosis.txt",names=colnames)
df.head()
df.count().isnull()

X=df.iloc[:,0:9]
X
y=df.iloc[:,9]
y
y[y=='N']=1
y[y=='O']=0
y
y=y.astype(int)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

rfc=RandomForestClassifier(n_estimators=10,criterion='gini')
rfc.fit(X,y)
y_pred=rfc.predict(X)
r2_score(y,y_pred)     #this model proves 44% suitable for future predictions

y_pred=rfc.predict([[-1,0.35,0,0,0,1,1,0,0.8]])
y_pred



