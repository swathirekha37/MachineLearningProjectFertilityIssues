# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:09:08 2019

@author: Surface
"""

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


#Linear regression test with r2-score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#checking with regression techniques
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
r2_score(y_test,y_pred)   # 14%, this preoves Linear regression is not suitable.


#below is svm checking r2-score
from sklearn.preprocessing import StandardScaler
SC_X=StandardScaler()
SC_X_train=SC_X.fit_transform(X_train)
SC_X_test=SC_X.fit_transform(X_test)
#y=SC_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(SC_X_train,y_train)
y_pred=regressor.predict(SC_X_test)
r2_score(y_test,y_pred)      #this is 23% correct predictor of future values.
#y_pred=regressor.predict([[1,.27,1,1,1,1,1,1,0.7]])

# ridge regression check r2_score
from sklearn.linear_model import Ridge
rrl=Ridge(alpha=1.0)  # with low alpha value
rrl.fit(X_train,y_train)
y_pred=rrl.predict(X_test)
r2_score(y_test,y_pred)             # 12%, with low alpha value this model is not suitable.

rrh=Ridge(alpha=100)
rrh.fit(X_train,y_train)
y_pred=rrh.predict(X_test)
r2_score(y_test,y_pred)         #2%,with high alpha value also this is not suitable.

#lasso regression
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train,y_train)
y_pred=lasso.predict(X_test)
r2_score(y_test,y_pred)         # -0.006, proves that not at all related.


from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
X_train=SC.fit_transform(X_train)
#y_train=SC.fit_transform(y_train)
X_test=SC.fit_transform(X_test)
#y_test=SC.fit_transform(y_test)

#checking with classification texhniques
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
r2_score(y_test,y_pred)     #-0.006, this proves that logistic regression not suitable

#naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax=min_max_scaler.fit_transform(X_test)
naive_bayes=MultinomialNB()
naive_bayes.fit(X_train_minmax,y_train)
y_pred=naive_bayes.predict(X_test_minmax)
r2_score(y_test,y_pred)   # -0.111, this proves that logistic regression not suitable

#Random forest classification
rfc=RandomForestClassifier(n_estimators=10,criterion='gini')
rfc.fit(X,y)
y_pred=rfc.predict(X)
r2_score(y,y_pred)     #this model proves 44% suitable for future predictions
y_pred=rfc.predict([[-1,0.67,0,0,1,0,0.6,0,0.5]])
y_pred

#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
r2_score(y_test,y_pred)     # -1.222, this model proves not suitable for future predictions

#SVC
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
r2_score(y_test,y_pred)     #-0.111, not suitable model

#knn with k=5
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
r2_score(y_test,y_pred)     # -0.1111, this model is not suitable for future predictions
#y_pred=classifier.predict([1,.27,1,1,1,1,1,1,0.7])


#k-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
#y_kmeans = kmeans.fit_predict(X)
kmeans.fit(X_train,y_train)
y_pred=kmeans.predict(X_test)
r2_score(y_test,y_pred)         #not at all suitable model























