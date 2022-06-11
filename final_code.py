# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:59:21 2021

@author: nikhi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn import metrics


df = pd.read_csv("Training Data_2021.csv")
df = pd.get_dummies(df, columns=['mvar47'],drop_first = True)


removed = ['mvar22','mvar26', 'mvar15', 'mvar35', 'mvar30', 'mvar23', 
           'mvar45', 'mvar11', 'mvar41', 'mvar31', 'mvar40']

categorical = ['mvar47_L']

df = df.drop(removed, axis = 1)

df_cat = df[categorical]
df_num = df.drop(categorical,axis = 1)

df_num = df_num.apply(pd.to_numeric, errors='coerce')
df_cat = df_cat.apply(pd.to_numeric, errors='coerce')


df_num = df_num.fillna(df_num.mean())
for column in df_cat.columns:
    df_cat[column].fillna(df_cat[column].mode()[0], inplace=True)

df_cat.nunique()
df_cat = pd.get_dummies(df_cat,columns = categorical,drop_first = True)



y = df['default_ind']

df_num = df_num.drop(['application_key','default_ind'],axis = 1)

standardizer = StandardScaler()
df_stand = standardizer.fit_transform(df_num)
df_stand = pd.DataFrame(df_stand)
df_stand.columns = df_num.columns


"""
pca = PCA(n_components=len(df_stand.columns))
principalComponents = pca.fit_transform(df_stand)


# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


comp = 5

pc = pd.DataFrame(data = principalComponents[:,:comp]) 

"""
findata = pd.merge(df_stand, df_cat, left_index=True, right_index=True)
#X_train,X_test,y_train,y_test = train_test_split(findata,y,test_size=0.0001)


models = [#LogisticRegression(),GaussianNB(),
          SGDClassifier(loss = 'modified_huber',class_weight = "balanced",shuffle = True,random_state = 101)
          #KNeighborsClassifier(n_neighbors = 10),
          #RandomForestClassifier(n_estimators=70,n_jobs = -1,random_state = 101,min_samples_leaf = 30),
          #SVC(kernel = 'linear',C=0.025,random_state = 101)
        
        
        ]

for model in models:
    #model.fit(X_train,y_train)
    model.fit(findata,y)
    #y_pred = np.around(model.predict(X_test))
    #print(str(model))
    #print(metrics.classification_report(y_test,y_pred))

df1 = pd.read_csv("Test Data_2021.csv")
df1 = pd.get_dummies(df1, columns=['mvar47'],drop_first = True)

df1 = df1.drop(removed, axis = 1)

df1_cat = df1[categorical]
df1_num = df1.drop(categorical,axis = 1)

df1_num = df1_num.apply(pd.to_numeric, errors='coerce')
df1_cat = df1_cat.apply(pd.to_numeric, errors='coerce')

df1_num = df1_num.fillna(df1_num.mean())
for column in df1_cat.columns:
    df1_cat[column].fillna(df1_cat[column].mode()[0], inplace=True)

df1_cat.nunique()
df1_cat = pd.get_dummies(df1_cat,columns = categorical,drop_first = True)


df1_num = df1_num.drop(['application_key'],axis = 1)

standardizer1 = StandardScaler()
df1_stand = standardizer1.fit_transform(df1_num)
df1_stand = pd.DataFrame(df1_stand)
df1_stand.columns = df1_num.columns

findata1 = pd.merge(df1_stand, df1_cat, left_index=True, right_index=True)


y1_pred = np.around(model.predict(findata1))

submission = df1[['application_key']].copy()
submission['default_key'] = y1_pred

submission.to_csv("Group_29_6.csv")












