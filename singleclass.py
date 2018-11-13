# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:03:05 2018

@author: Painel-16
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn import metrics
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from collections import Counter



path_train = r"C:\Users\Painel-16\Desktop\Classificadores\delegacias_pad1.csv"
#path_test = r"C:\Users\Painel-16\Desktop\Classificadores\Dease Total Geral 2016_ok.csv"
path_test = r"C:\Users\Painel-16\Desktop\Classificadores\deleg_ita.csv"



def readFile(path):
    
    df = pd.read_csv(path, delimiter = ";", encoding="utf-8")
    
    return (df)


def getCategoriesDict(df, col_cat):
    
    categories = list(set(df[col_cat]))
    labels = np.arange(len(categories))
    dic_categories = dict(zip(categories,labels))
    
    inv_dic = {v: k for k, v in dic_categories.items()}
    
    return (dic_categories, inv_dic)


def setupTarget(df, dict_categories, col_cat):
    target = []
    
    for i in range(0,len(df)):
        #print(df[col_cat].iloc[i],i,"/",len(df))
        target.append(dict_categories[df[col_cat].iloc[i]]) 
    df["target"] = target
    
    return (df)

def transformsIntoX(df, col_fact):

    df = df.dropna()
    stop_words = set(stopwords.words('portuguese'))
    stop_words.add("art")
    cvect = CountVectorizer(stop_words=stop_words, strip_accents='unicode')
    X = cvect.fit_transform(df[col_fact])
    
    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X)
    X_train_tf = tf_transformer.transform(X)
    
    return (X_train_tf, cvect, tf_transformer)
    
def crossValidate(df, X, smote = False):
    #clf = MultinomialNB()
    #clf = LogisticRegression()
    #clf = KNeighborsClassifier(1)
    #clf =  SVC(kernel="linear", C=0.025)
    #clf =  SVC(gamma = 2, C=1)
    #clf = DecisionTreeClassifier()
    clf = MLPClassifier()
    
    
    if (smote == False):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                        list(df["target"]),
                                                        test_size=0.3, random_state=0)
    else:
        sm = SMOTE(random_state=42,k_neighbors=4)
        #print (X.shape,list(df["target"]))
        X_res, y_res = sm.fit_resample(X,list(df["target"]))
        print('Resampled dataset shape %s' % Counter(y_res))
        
        X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,
                                                            test_size=0.3, random_state=0) 
    
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print ("OVERALL SCORE: ",clf.score(X_test, y_test),"\n")     
    print("REPORT: \n",metrics.classification_report(y_test, predicted), "\n")
    print("CONFUSION: \n", metrics.confusion_matrix(y_test, predicted), "\n")
    
    
    return ( X_train, X_test, y_train, y_test, predicted, clf)

def classify(df, clf, X_test, y_test):
    predicted = clf.predict(X_test)
    print ("OVERALL SCORE: ",clf.score(X_test, y_test),"\n")     
    print("REPORT: \n",metrics.classification_report(y_test, predicted), "\n")
    print("CONFUSION: \n", metrics.confusion_matrix(y_test, predicted), "\n")
    return ("hope this works")

def predictNew(new, cvect, tf_transf, clf):
    X = cvect.transform(new)
    X_test = tf_transf.transform(X)
    
    predicted = clf.predict(X_test)
    predicted_proba = clf.predict_proba(X_test)
    
    df_pred = pd.DataFrame()
    df_pred["bruto"] = new
    df_pred["predicted_index"] = predicted
    df_pred["class"] = [inv_dict_categories[i] for i in range(0,len(predicted))]
    
    return (df_pred, predicted_proba)
    


if __name__=='__main__':

    train = readFile(path_train)
    test = readFile(path_test)
    len(test.dropna())
    len(train.dropna())
    
    dict_categories, inv_dict_categories = getCategoriesDict(train, "Categoria")
    #dict_categories1 = getCategoriesDict(test, "Categoria 1")
    
    
    train = setupTarget(train, dict_categories, "Categoria")
    test = setupTarget(test, dict_categories, "Categoria 1")
    
    X_train, cvect, tf_transf = transformsIntoX(train, "Fato")
    feats = cvect.get_feature_names()
    
    

    
    #X1 = cvect.transform(test[["Ato Infracional (bruto)","target"]].dropna()["Ato Infracional (bruto)"])
    #X1_test = tf_transf.transform(X1)
    
    
    
    X2 = cvect.transform(test[["Fatos Ocorridos Bruto_1","target"]].dropna()["Fatos Ocorridos Bruto_1"])
    X2_test = tf_transf.transform(X2)
    
    
    #X3l = ["AMEAÇA","ECA ART-241","DIRIGIR SEM HABILITAÇÃO","ESTUPRO","CAlúnia","ESTUPRO"]
    #X3 = cvect.transform(X3l)
   # X3_test = tf_transf.transform(X3)
    
    #X_test1 = transformsIntoX(test, "Ato Infracional (bruto)", oldvect = cvect)
    

    #X_train, X_test, y_train, y_test, predicted, clf = crossValidate(train, X_train)
    
    X_train, X_test, y_train, y_test, predicted, clf = crossValidate(train.dropna(), X_train, smote = True)
    
    new = ["AMEAÇA","ECA ART-241","DIRIGIR SEM HABILITAÇÃO","ESTUPRO","CAlúnia","ESTUPRO"]
    df_pred, predprob_new = predictNew(new, cvect, tf_transf, clf)
    print (df_pred["class"])
    
    predicted = clf.predict(X2_test)
    predicted_proba = clf.predict_proba(X2_test)
    #decision = clf.decision_function(X2_test)
    
    
    target1 = test[["Fatos Ocorridos Bruto_1","target"]].dropna()["target"]
    
    print ("OVERALL SCORE: ",clf.score(X2_test, target1),"\n")     
    print("REPORT: \n",metrics.classification_report(target1, predicted), "\n")
    print("CONFUSION: \n", metrics.confusion_matrix(target1, predicted), "\n")
    conf_matrix =  metrics.confusion_matrix(target1, predicted)
    
    i=0
    cont=0
    for j in target1:
        if (inv_dict_categories[predicted[i]]!=inv_dict_categories[j]):
            print("ROW:", i)
            print("PREDICTED:", predicted[i], ": ", inv_dict_categories[predicted[i]])
            print("ACTUAL:", j, ": ",inv_dict_categories[j])
            print("CASE:", test[["Fatos Ocorridos Bruto_1","target"]].dropna().iloc[i]["Fatos Ocorridos Bruto_1"])
            cont += 1
        i += 1
    print("TOTAL ERRORS", cont)
    
    #for i in range(0,len(predicted)):
    #    print(inv_dict_categories[predicted[i]])
    cont = 0
    for i in range(0,len(predicted_proba)):
        for j in range(0,len(predicted_proba[0])):
            if predicted_proba[i,j] >= 0.10:
                cont += 1
        if cont >= 2:
            print ("cont:",cont,"__line:",i)
            print ("probas:")
        cont = 0
        

  
  