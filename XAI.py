'''
Created on 19 ene. 2020

@author: enriq
'''

import subgpylib as sg
import explainability.SubgroupExplainer as xai
import pandas as pd
from subgpylib import *
import csv
import time
import numpy as np


if __name__ == '__main__':
    dataset_files = ["house"]
    #dataset_files = ["baseball", "autoMPG8", "dee", "ele-1", "forestFires"]
    #["baseball", "autoMPG8", "dee", "ele-1", "forestFires", "concrete", "treasury", "wizmir", "abalone", "puma32h", "ailerons", "elevators", "bikesharing", "california", "house"
    algorithms = ["SD4TS"]
    min_samples_range = [0,0.05]
    filename = './results/article_correctedvm.tex' 
    delimiter='\t'
    lineterminator = "\n"
    
    timem = []
    
    binary_attributes = ["free_agency_eligibility", "free_agent", "arbitration_eligibility", "arbitration" ]

    with open(filename, 'a+') as file:
        writer = csv.writer(file, delimiter='&', lineterminator = "\n", quoting=csv.QUOTE_NONE)
    
        writer.writerow(["\\centering"])  
        writer.writerow(["\\begin{adjustbox}{max width = \\columnwidth}"])  
        writer.writerow(["\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}"])
        writer.writerow(["\\hline"])
        
        writer.writerow(["\multirow{2}{*}{D}", "\multirow{2}{*}{SG alg.}", "\multicolumn{4}{c|}{SG metrics}","\multirow{2}{*}{min split}","\multicolumn{4}{c|}{SGExplainer metrics} \\\\"])
        writer.writerow(["", "", "$|SG|$","$|S|$","$|S_u|$","card","","$|T|$","Depth","Min_depth","Accuracy \\\\"])
    
        for dataset_file in dataset_files :
            data = pd.read_csv("support/dis/"+ dataset_file +"_dis.csv")
            data.columns = data.columns.str.strip().str.lower().str.replace('-', '_').str.replace('(', '').str.replace(')', '')
            target_name = data.columns[-1]
            binary_attributes = []
            if dataset_file == "baseball" :
                binary_attributes = ["free_agency_eligibility", "free_agent", "arbitration_eligibility", "arbitration" ]
            
            print()
            print()
            # We launch a subgroup discovery algorithm and we store the result
            for algorithm in algorithms :
                alg = None
                start_sg = 0
                end_sg = 0
                result = []
                if algorithm == "SD" :
                    alg = sg.AlgorithmSD(g_parameter = 10, min_support = 0.15, beam_width = 20 )
                    target_value = data[target_name].iloc[0]
                    start_sg = time.time()
                    setL = alg.generateSetL(data, (target_name, target_value), binary_attributes = binary_attributes, discretizer = "ncuts", ncuts = 5)
                    result = alg.fit(data, (target_name, target_value), setL)
                    end_sg = time.time()
                elif algorithm == "CN2-SD":
                    alg = sg.AlgorithmCN2SD(beam_width = 3, weighting_scheme = 'multiplicative', gamma = 0.3, max_rule_length = 3, discretizer = "ncuts", ncuts = 5)
                    start_sg = time.time()
                    result = alg.fit(data, target_name, binary_attributes = binary_attributes)
                    end_sg = time.time()
                elif algorithm == "SD4TS" : 
                    alg = sg.AlgorithmSD4TS(max_subgroups=20, min_support=0.15)
                    start_sg = time.time()
                    result = alg.fit(data, target_name, binary_attributes = binary_attributes)
                    end_sg = time.time()
                for i in result :
                    print(i[0])
                
                print(algorithm)
                print("------------------------------")
                print("Number of subgroups: " + str(len(result)))
                list_of_selectors = [e for i in result for e in i[0].getCondition()]
                print("Number of selectors: " + str(len(list_of_selectors)))
                unique_selectors = []
                for i in list_of_selectors :
                    if i not in unique_selectors :
                        unique_selectors.append(i)
                print("Number of unique selectors: " + str(len(unique_selectors)))
                m_cardinality = round(len(list_of_selectors)/len(result),2)
                print("Average cardinality of the subgroups: " + str(m_cardinality))
                
                for msm in min_samples_range :
                    tree= xai.SubgroupTreeExplainer(min_samples_split = msm)
                    start_tree = time.time()
                    target = tree.fit(data, result)
                    end_tree = time.time()
                    #print(tree)

                    print("------------------------------")
                    print("Tree")
                    print("Number of nodes: " + str(len(tree)))
                    print("Depth of the tree: " + str(tree.depth()))
                    print("Branch with less depth: " + str(tree.min_depth()))
                    print("BF of the tree: " + str(tree.branching_factor()))
                    print("Number of leaf nodes: " + str(tree.leaf_nodes()))
                    acc = str(round(tree.subset_accuracy(data, target),2))
                    print("Accuracy: " + acc)
                    print("------------------------------")
                    print("Times")
                    print("SG discovery time: " + str(end_sg - start_sg))
                    print("Tree time: " + str(end_tree-start_tree))
                    timem.append([dataset_file, algorithm, str(msm), end_sg - start_sg, end_tree-start_tree])
                    
                    if algorithm == "SD" and msm==0:
                        writer.writerow(["\\hline"])
                        writer.writerow(["\multirow{6}{*}{"+dataset_file+ "}", "\multirow{2}{*}{"+algorithm + "}", "\multirow{2}{*}{"+str(len(result))+ "}",
                                         "\multirow{2}{*}{"+str(len(list_of_selectors))+ "}","\multirow{2}{*}{"+str(len(unique_selectors))+ "}","\multirow{2}{*}{"+str(m_cardinality)+ "}",
                                         msm,str(len(tree)),str(tree.depth()),str(tree.min_depth()), acc +"\\\\"])
                    elif msm == 0 :
                        writer.writerow(["\cline{2-11}"])
                        writer.writerow(["", "\multirow{2}{*}{"+algorithm + "}", "\multirow{2}{*}{"+str(len(result))+ "}",
                                         "\multirow{2}{*}{"+str(len(list_of_selectors))+ "}","\multirow{2}{*}{"+str(len(unique_selectors))+ "}","\multirow{2}{*}{"+str(m_cardinality)+ "}",
                                         msm,str(len(tree)),str(tree.depth()),str(tree.min_depth()), acc +"\\\\"])
                    else :
                        writer.writerow(["\cline{7-11}"])
                        writer.writerow(["", "","",
                                         "","","",
                                         msm,str(len(tree)),str(tree.depth()),str(tree.min_depth()), acc +"\\\\"])                      
                print()
        
        writer.writerow(["\\hline"])
        writer.writerow(["\\end{tabular}"])  
        writer.writerow(["\\end{adjustbox}"])  
        
    with open("./results/article_time.csv", 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator = "\n", quoting=csv.QUOTE_NONE)
        for i in timem :
            writer.writerow(i)     
    
    pass