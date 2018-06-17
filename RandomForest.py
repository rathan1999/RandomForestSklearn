import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import *
from sklearn.ensemble import RandomForestClassifier as RFC

def normalise(Xtrain):
    return (Xtrain - np.mean(Xtrain,axis=1,keepdims=True))/(np.max(Xtrain,axis=1,keepdims=True)-np.min(Xtrain,axis=1,keepdims=True))

def load_data():
    data=pd.read_csv("C:\\Users\\Sai Rathan\\Desktop\\train.csv")
    data['Sex'].replace(['female','male'],[1,2],inplace=True)
    data['Age'].replace(np.NaN,data['Age'].mean(),inplace=True)
    data['Pclass'].replace(np.NaN,data['Pclass'].mean(),inplace=True)
    data['Sex'].replace(np.NaN,data['Sex'].mean(),inplace=True)
    data['SibSp'].replace(np.NaN,data['SibSp'].mean(),inplace=True)
    data['Parch'].replace(np.NaN,data['Parch'].mean(),inplace=True)
    data['Fare'].replace(np.NaN,data['Fare'].mean(),inplace=True)
    data['Survived'].replace(np.NaN,data['Survived'].mean(),inplace=True)
    data['Embarked'].replace(['C','S','Q'],[1,2,3],inplace=True)
    data['Embarked'].replace(np.NaN,data['Embarked'].mean(),inplace=True)
    Datatrain=data.sample(frac=0.9,random_state=200)
    Dataval=data.drop(Datatrain.index)
    Xtrain=np.array(Datatrain[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]).T
    Xtrain = normalise(Xtrain)
    Xval=np.array(Dataval[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]).T
    Xval=normalise(Xval)
    Ytrain=np.array(Datatrain['Survived']).reshape(1,Xtrain.shape[1])
    Yval=np.array(Dataval['Survived']).reshape(1,Xval.shape[1])
    return [Xtrain.T,Ytrain.T,Xval.T,Yval.T]

def load_datatest():
    data=pd.read_csv("C:\\Users\\Sai Rathan\\Desktop\\test.csv")
    data['Sex'].replace(['female','male'],[1,2],inplace=True)
    data['Age'].replace(np.NaN,data['Age'].mean(),inplace=True)
    data['Pclass'].replace(np.NaN,data['Pclass'].mean(),inplace=True)
    data['Sex'].replace(np.NaN,data['Sex'].mean(),inplace=True)
    data['SibSp'].replace(np.NaN,data['SibSp'].mean(),inplace=True)
    data['Parch'].replace(np.NaN,data['Parch'].mean(),inplace=True)
    data['Fare'].replace(np.NaN,data['Fare'].mean(),inplace=True)
    data['Embarked'].replace(['C','S','Q'],[1,2,3],inplace=True)
    data['Embarked'].replace(np.NaN,data['Embarked'].mean(),inplace=True)
    Datatest=data
    Xtest=np.array(Datatest[['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']]).T
    Xtest=normalise(Xtest)
    Pass=np.array(data['PassengerId']).reshape(1,Xtest.shape[1])
    return [Xtest.T,Pass]

Xtrain,Ytrain,Xval,Yval=load_data()
Xtest,Pass=load_datatest()
clf=RFC(max_depth=5)
clf.fit(Xtrain,Ytrain.reshape(Ytrain.shape[0]))
pred=clf.predict(Xval)
err=(pred-Yval.reshape(Yval.shape[0]))
err=np.sum(err*err)
print((Yval.shape[0]-err)*100/Yval.shape[0])
print("PassengerId,Survived")
pred=clf.predict(Xtest)
for i in range(pred.shape[0]):
    print(Pass[0][i],pred[i],sep=",")

