'''
Created on 23 mar. 2020

@author: enriq
'''
import sys
sys.path.insert(1, '../subgpylib')
import subgpylib as sg
import pandas as pd 
import numpy as np 
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
import graphviz
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from graphviz import Source
from IPython.display import SVG
from sklearn import preprocessing

if __name__ == '__main__':
    print("Inicio")

    # First, we load the dataframe
    df = pd.read_csv("../datasets/EnriqueThesis/lenses.csv")
    
    # We launch a subgroup discovery algorithm and we store the result
    algCN2SD = sg.AlgorithmCN2SD(beam_width = 3, weighting_scheme = 'multiplicative', gamma = 0)
    result = algCN2SD.fit(df, "class")

    for i, quality in result :
        print("   "+str(i))
        print("       "+str(quality))
    
    
    # Now, the explainability begins
    # We calculate the labels of the dataset, which will be our target attribute
    target_column = []
    index_dict = algCN2SD.__getIndexDictionary__(df)
    for i in df.itertuples(index = False) :
        target_value = 0
        for j in range(len(result)) :
            elem = result[j][0]
            if elem.matchElement(i, index_dict) :
                target_value = target_value + 2**j
        target_column.append(str(target_value))

    df["subgroups"] = target_column
    print(df)


    for column in df.columns:
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column])



    y = df.subgroups
    X = df.drop("subgroups",1)
    print(df)
    #X = X.astype(bool)
    dec_tree = tree.DecisionTreeClassifier().fit(X, y)


    dot_data = StringIO()

    graph = Source(export_graphviz(dec_tree))
    SVG(graph.pipe(format='svg'))

    
    pass