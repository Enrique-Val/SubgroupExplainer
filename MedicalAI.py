'''
Created on 19 ene. 2020

@author: enriq
'''

import subgpylib
from pandas import DataFrame, read_csv
from subgpylib import *
import explainability.SubgroupExplainer as xai


if __name__ == '__main__':
    # Creamos un DataFrame a partir de un fichero .csv
    original_data = read_csv("datasets/EnriqueThesis/CMI.csv", )
    data = original_data.copy()
    del data['episodeId']
    del data['culture_service']
    del data['month']
    del data["year"]
    del data["episodeDuration_inDays"]
    
    print("Original DataFrame")
    print(data)
    print()  
    
    for i in data.columns :
        print(type(data[i].iloc[0]))
    
    '''    
    # Prueba de algoritmo de cubrimiento
    subgCN2SD = SubgroupForCN2SD(Pattern([Selector("bread", Selector.OPERATOR_EQUAL, 'yes')]), Selector("diaper", Selector.OPERATOR_EQUAL, 'yes'))
    print(subgCN2SD)
    i = 0
    index = {}
    for column in data.columns :
        index.update({column : i})
        print(str(i) + " " + column)
        i = i + 1
    '''
    
    '''
    for row in data.itertuples(False) :
        print(row)
        
    columns_without_target = data.columns[data.columns != 'type']
    for column in columns_without_target :
        print(data[column])
    
        

    '''
    # Creamos el objeto de tipo 'AlgorithmCN2SD'.
    # - El valor de soporte es en forma de proporcion.
    alg = subgpylib.AlgorithmCN2SD(beam_width = 3, weighting_scheme = "multiplicative", gamma = 0.3, max_rule_length = 3)
    
    #L = alg.generateSetL(data, ("micIncreases","Yes"), binary_attributes = ["sex"])
    

    # Ejecutamos el algoritmo CN2SD.
    #result = alg.fit(data, "micIncreases", binary_attributes = ["sex"])
    result = alg.fitOneClass(data, ("micIncreases", "Yes"), binary_attributes = ["sex"])
    #result.pop(0)
    #result[0][0].getCondition().removeSelector(result[0][0].getCondition().getListOfSelectors()[2])
    # Visualizamos los resultados.
    print("Subgroups founded")
    for i, (subgroup, qm) in enumerate(result):
        print("Subgroup "+ str(i) +": " + str(subgroup))
        print("\t Quality: " + str(qm))

    print("Number of subgroups: " + str(len(result)))
    list_of_selectors = [e for i in result for e in i[0].getCondition()]
    print("Number of selectors: " + str(len(list_of_selectors)))
    unique_selectors = []
    for i in list_of_selectors :
        if i not in unique_selectors :
            unique_selectors.append(i)
    print("Number of unique selectors: " + str(len(unique_selectors)))
    print()
    print("Start!")
    
    for i in [0, 0.05 ,0.1] :
        tree = xai.SubgroupTreeExplainer(min_samples_split = i)
        target = tree.fit(data, result)
        #print(tree)
        print("min samples split "+str(i))
        print("Number of nodes: " + str(len(tree)))    
        print("Depth of the tree: " + str(tree.depth()))
        print("Branch with less depth: " + str(tree.min_depth()))
        acc = str(round(tree.subset_accuracy(data, target),2))
        print("Accuracy: " + acc)   
        print()
    tree = xai.SubgroupTreeExplainer(min_samples_split = 0.05)
    tree.fit(data, result)
    print(tree)
        