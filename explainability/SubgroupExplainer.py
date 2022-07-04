'''
Created on 23 mar. 2020

@author: enriq
'''

from subgpylib import *
from pandas import read_csv, DataFrame
import numpy as np 
from collections import Counter
import math

def getKey(item):
    return item[1]      

class Node():
    """
    This class represents a decision tree surrogate used for explaining the subgroup discovery task.

    IMPORTANT NOTE: This is an internal class used for training a Subgroup Tree Explainer. Please, do not modify manually objects of this class.


    :type value: list
    :param value: List of the samples of a given dataset contained in that node
    :type split: subgpylib.Selector
    :param split: (Takes value None if it is a leaf node) The split used to divide the dataset of the node
    :type result: list or dict
    :param result: (Takes value None if it is not a leaf node) The subgroups that cover the split OR the probabilities of the subgroups covering examples with such attributes.
    :type left: Node
    :param left: Left split
    :type right: Node
    :param right: Right split
    """
    
    def __init__(self, value, split = None, result = None, left = None, right = None) :
        self.value = value
        self.split = split
        self.result = result
        self.left = left
        self.right = right
        
    def fit(self, dataset, subgroup_set, min_samples_split = 0, already_used_splits = []):
        # Create a temporary instance of the dataset
        node_dataset = dataset.loc[self.value, :].reset_index()
        self.result = self.getResults(node_dataset)
        
        # Check if we should split
        if self.stop(node_dataset, min_samples_split) :
            return

        # Get the best split
        self.split = self.selectSplit(node_dataset, subgroup_set, already_used_splits)
            
        # If no possible split could be found, we do no split
        if self.split is None :
            return
            
        # Else, Split data (check first if the data str or numpy. In the latter, we use .item() to extract native types)
        # We iterate over every element of the list (item of the dataset), checking if it matches or not with the current split.
        left_value = []
        right_value = []
        for i in self.value :
            if type(dataset.loc[i, self.split.getAttribute()]) is str :
                if self.split.match(self.split.getAttribute(), dataset.loc[i, self.split.getAttribute()]) :
                    left_value.append(i)
                else : 
                    right_value.append(i)
            else :
                if self.split.match(self.split.getAttribute(), dataset.loc[i, self.split.getAttribute()].item()) :
                    left_value.append(i)
                else : 
                    right_value.append(i)
        
        #left_value = [i for i in self.value if self.split.match(self.split.getAttribute(), dataset.loc[i, self.split.getAttribute()])]
        #right_value = list(np.setdiff1d(self.value,left_value))
                
        # Repeat for each branch
        self.left = Node(left_value)
        self.right = Node(right_value)
        alt_list = already_used_splits.copy()
        alt_list.append(self.split)
        self.left.fit(dataset, subgroup_set, min_samples_split, alt_list)
        alt_list = already_used_splits.copy()
        alt_list.append(self.split)
        self.right.fit(dataset, subgroup_set, min_samples_split, alt_list)
        
    
    
    def stop(self, dataset, min_samples_split) :
        return (not len(dataset['tree_target'].unique()) > 1) or dataset.shape[0] < min_samples_split
    
    
    def selectSplit(self, dataset, subgroup_set, already_used_splits):
        S = []
        SG = []
        
        # Set of all feasible classes
        target_set = []
        for i in subgroup_set :
            if i.getTarget() not in target_set :
                target_set.append(i.getTarget())
        
        # Obtain the maximum length of the subgroups
        max = -100
        for i in subgroup_set :
            if len(i.getCondition()) > max :
                max = len(i.getCondition())
        
            
        i = 1
        while (i <= max) :
            # Selectors that appear in subgroups of subgroup_set whose length == i are stored in S
            for j in subgroup_set :
                if len(j.getCondition().getListOfSelectors()) == i:
                    S = S + [k for k in j.getCondition().getListOfSelectors() if k not in already_used_splits and k not in S]
            # if there are subgroups whose length <= i
            if len(S) > 0 :
                # Form all the feasible 1-selector subgroups and store them in SG
                for selector in S :
                    for target in target_set :
                        new_sg = Subgroup(Pattern([selector]), target)
                        new_sg_quality = self.getQualityMeasure(dataset, new_sg)
                        SG.append((selector, new_sg_quality))
                SG = sorted(SG, key=getKey, reverse=True)
                # Return the one with the highest quality (only if it is not useless)
                while (len(SG) > 0) :
                    # Check if it is useless splitting the data
                    actual = SG[0][0]
                    left_value = []
                    right_value = []
                    for j in range(0,len(dataset)) :
                        if type(dataset.loc[j, actual.getAttribute()]) is str :
                            if actual.match(actual.getAttribute(), dataset.loc[j, actual.getAttribute()]) :
                                left_value.append(j)
                            else : 
                                right_value.append(j)
                        else :
                            if actual.match(actual.getAttribute(), dataset.loc[j, actual.getAttribute()].item()) :
                                left_value.append(j)
                            else : 
                                right_value.append(j)
                    #left_value = [i for i in range(0,len(dataset)) if actual.match(actual.getAttribute(), dataset.loc[i, actual.getAttribute()])]
                    #right_value = list(np.setdiff1d(range(0,len(dataset)),left_value))
                    
                    # If the split is useless (does not split the data), discard it and try with another one
                    if (right_value == [] or left_value == []) :
                        SG.pop(0)
                    else :
                        return actual
            i = i+1
        
       
        return None
          
         
    def obtainBasicMetrics(self, dataset, subgroup):
        """Internal method to obtain the basic metrics (tp, fp, TP and FP) of a subgroup in a dataset.

        It is VERY IMPORTANT to respect the types of the attributes: the value of a selector of the subgroup MUST BE comparable with the value of the corresponding attribute in the dataset.
        
        :type dataset: pandas.DataFrame
        :param dataset: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type subgroup: Subgroup
        :param subgroup: Input subgroup.
        :rtype: tuple
        :return: a tuple with the basic metrics in this order: (tp, fp, TP, FP).
        """
        if type(dataset) is not DataFrame:
            raise TypeError("Parameter 'dataset' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0):
            raise EmptyDatasetError("The input dataset is empty.")
        if not isinstance(subgroup, Subgroup):
            raise TypeError("Parameter 'subgroup' must be of type 'Subgroup' (or subclasses).")
        # We need to use the condition of the subgroup (Pattern) and the target variable (Selector) separatly.
        subgroup_condition = subgroup.getCondition()
        subgroup_target = subgroup.getTarget()
        # We initialize the basic metrics that we want to obtain.
        tp = 0
        fp = 0
        TP = 0
        FP = 0
        index_dict = self.__getIndexDictionary__(dataset)
        row_index = 0
        for row in dataset.itertuples(False):
            # FIRST: we check the condition of the subgroup.
            subgroup_condition_and_row_match = True # Variable to control if the condition of the subgroup and the row match. Initially, yes.
            index_in_subgroup_condition = 0 # Index over the selectors of the condition of the subgroup.
            while (index_in_subgroup_condition < len(subgroup_condition)) and (subgroup_condition_and_row_match): # Iterate over the selectors of the condition of the subgroup.
                current_selector = subgroup_condition.getListOfSelectors()[index_in_subgroup_condition]
                try: # IMPORTANT: If the attribute of the selector is not in the dataset, an exception of pandas (KeyError) will be raised.
                    # If one of the selectors of the condition of the subgroup does not match, the condition of the subgroup does not match (and we can go to the next row).
                    subgroup_condition_and_row_match = current_selector.match(current_selector.getAttribute(), row[index_dict[current_selector.getAttribute()]])
                except KeyError as e:
                    subgroup_condition_and_row_match = False
                index_in_subgroup_condition = index_in_subgroup_condition + 1
            # SECOND: we check the target variable of the subgroup.
            try:
                subgroup_target_and_row_match = subgroup_target.match(subgroup_target.getAttribute(), row[index_dict[subgroup_target.getAttribute()]])
            except KeyError as e:
                subgroup_target_and_row_match = False
            # FINALLY, we check the results.
            if (subgroup_condition_and_row_match) and (subgroup_target_and_row_match):
                tp = tp + 1
            if (subgroup_condition_and_row_match) and (not subgroup_target_and_row_match):
                fp = fp + 1
            if subgroup_target_and_row_match:
                TP = TP + 1
            if not subgroup_target_and_row_match:
                FP = FP + 1
        return (tp, fp, TP, FP)
    
    def obtainDictOfMetrics(self, dataset, subgroup):
        tuple_metrics = self.obtainBasicMetrics(dataset, subgroup)
        return {QualityMeasure.BasicMetric_tp : tuple_metrics[0], QualityMeasure.BasicMetric_fp : tuple_metrics[1], QualityMeasure.BasicMetric_TP : tuple_metrics[2], QualityMeasure.BasicMetric_FP : tuple_metrics[3]}
    
    def getQualityMeasure(self,dataset,subgroup) :
        return QualityMeasureRAcc().compute(self.obtainDictOfMetrics(dataset, subgroup))    
    
    
    def __getIndexDictionary__(self, dataset):
        """Auxiliary method to calculate the index dictionary, a data structure that maps each column name of a pandas dataframe into its integer value.

        :type pandas_dataframe: pandas.DataFrame
        :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :rtype: dict
        :return: the index dictionary for the given pandas dataframe.
        """
        if type(dataset) is not DataFrame :
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0) :
            raise EmptyDatasetError("The input dataset is empty.")
        
        
        i = 0
        index_dict = {}
        for column in dataset.columns :
            index_dict.update({column : i})
            i = i + 1     
        return index_dict 
        
            
    def __str__(self):
        return self.__str_aux__(0)
        
    def __str_aux__(self,n):
        to_return = ""
        tabs = ""
        for i in range(0,n) :
            tabs += "\t"
        to_return += tabs + "---------------------- \n"
        to_return += tabs + "Number of examples: " +str(len(self.value)) + "\n"
        #to_return += tabs + "Examples: " + str(self.value) + "\n"
        
        if self.result is not None :
            if type(self.result) is list :
                to_return += tabs + "Subgroups: " + str(self.result) + "\n"  
            elif type(self.result) is dict :
                dict_str = ""
                for i in self.result :
                    dict_str += str([i]) + " with " + str(self.result[i]) + "%  |  "
                to_return += tabs + "Subgroups: " + dict_str + "\n"  
        if self.split is not None :
            to_return += tabs + str(self.split) + "\n"
        to_return += tabs + "---------------------- \n"  
        if self.left is not None :
            to_return += self.left.__str_aux__(n+1)
        if self.right is not None :
            to_return += self.right.__str_aux__(n+1)
        return to_return


    def __len__(self) :
        total = 1
        if self.left is not None :
            total += self.left.__len__()
        if self.right is not None :
            total += self.right.__len__()
        return total
    
    def depth(self):
        if self.right is None and self.left is None:
            return 1
        if self.right is None :
            return 1+self.left.depth()
        if self.left is None :
            return 1+self.right.depth()
        return 1 + max(self.right.depth(), self.left.depth())
    
    def min_depth(self):
        if self.right is None and self.left is None:
            return 1
        if self.right is None :
            return 1+self.left.min_depth()
        if self.left is None :
            return 1+self.right.min_depth()
        return 1 + min(self.right.min_depth(), self.left.min_depth())

    def count_leaf_nodes(self):
        if self.right is None and self.left is None:
            return 1
        if self.left is not None :
            total1 = self.left.count_leaf_nodes()
        if self.right is not None :
            total2 = self.right.count_leaf_nodes()
        return total1+total2

    def branching_factor(self) :
        tree_size = self.__len__()
        tree_leaves = self.count_leaf_nodes()
        if (tree_size - tree_leaves) == 0:
            return 0
        return (tree_size-1)/(tree_size - tree_leaves)
    
    def classify(self,example):
        # If it is a leaf node, see if the example is well classified
        if self.right is None and self.left is None:
            # If all the examples with those characteristics belong to a specific subgroups
            return self.result
        # Else, examine in with branch is supposed to be the example, according to the split
        else :
            if self.split.match(self.split.getAttribute(), getattr(example,self.split.getAttribute())) :
                return self.left.classify(example)
            else :
                return self.right.classify(example)        
        



    def getResults(self, dataset) :
        if len(dataset['tree_target'].unique()) == 1 :
            return SubgroupTreeExplainer.decode(dataset.iloc[0]['tree_target'])
        elif False :
            # Here, things get complicated. We will return the set of subgroups with a bigger chance of appearing
            d = dict(Counter(dataset['tree_target']))
            m = max(d, key=d.get)
            return SubgroupTreeExplainer.decode(m)
        else :
            # Make a dictionary of all possible tree_target values (all possible sets of subgroups that can cover the tree). The value of each key will be how many times a specific set of subgroups appear
            d = dict(Counter(dataset['tree_target']))
            new = dict()
            
            # Convert the "set of subgroups" dictionary to a "subgroup" dictionary
            for key_i in d :
                l = SubgroupTreeExplainer.decode(key_i)
                for subg in l :
                    if subg in new :
                        new[subg] = new[subg] + d[key_i]
                    else :
                        new[subg] = d[key_i]
            
            # Order the dictionary
            new = {k: v for k, v in sorted(new.items(), key=lambda item: item[1], reverse = True)}
            
            # Convert to percentage
            new.update((x, round((y/dataset.shape[0])*100,2)) for x, y in new.items())
            
            return new


class SubgroupTreeExplainer() :
    """
    This class represents a decision tree surrogate used for explaining the subgroup discovery task.

    IMPORTANT NOTE: For now, the objective of this tree is just to visualize how to data is split and subgroups are formed by theses splits. Please, do not modify the tree. Just create it, train and print it

    :type root: Node
    :param root: Width of the beam.
    :type min_samples_split: int or float in range [0,1]
    :param min_samples_split: The minimum number of samples required to split an internal node, expressed in a proportion in range [0,1]
    """
    def __init__(self, min_samples_split = 0) :
        """Constructor to create an 'SubgroupTreeExplainer' object.
        """
        self.root = None
        self.min_samples_split = min_samples_split
        
    def fit(self, dataset, results, mode = "standard") :
        """Method to train the tree given the input dataset and the results from the algorithm .

        :type dataset: pandas.DataFrame
        :param dataset: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type results: str
        :param results: The list of (subgroup, quality) outputted by a subgroup discovery algorithm . It is VERY IMPORTANT to respect the following conditions:
          (1) it MUST be a lists composed by tuples or lists whose length is two (the first element it's a subgroup, the second, the quality),
        :type mode: str ("standard" or "true_positives")
        :param (OPTIONAL) If "standard" is selected, a tree will be built considering the pertenence of each example to the input subgroups. If "true_positives" is selected, the tree will be built considering if each example is a true positive of the input subgroups
        """
        
        # Check if the parameter 'dataset' is correct
        if type(dataset) is not DataFrame :
            raise TypeError("Parameter 'dataset' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0) :
            raise EmptyDatasetError("The input dataset is empty.")
        
        # Check if the parameter 'results' is correct
        if  type(results) is not list :
            raise TypeError("Parameter 'results' must be a list.")
        for i in results :
            if (type(i) is not list and type(i) is not tuple) or len(i) != 2 :
                raise TypeError("Parameter 'results' must be a list a tuples or lists whose length = 2 (subgroups and quality).")
        for i in results :
            if not isinstance(i[0], Subgroup) :
                raise TypeError("Parameter 'results': The first element of the tuples/lists must be a subgroups.")
            if type(i[1]) is not int and type(i[1]) is not float :
                raise TypeError("Parameter 'results': The second element of the tuples/lists must be int or double.")
        
        # Check if the parameter mode is correct 
        
        # We have to prepare the data
        target_column = []
        for i in range(0, len(dataset.index)) :
            example_i = dataset.iloc[[i],:]
            target_value = 0
            for j in range(len(results)) :
                subg = results[j][0]
                if mode == "standard" :
                    if subg.covers(example_i) :
                        target_value = target_value + 2**j
                elif mode == "true_positives" :
                    if subg.supports(example_i) :
                        target_value = target_value + 2**j
            target_column.append(str(target_value))
        dataset_labeled = dataset.copy()
        dataset_labeled["tree_target"] = target_column
        
        
        # First, we create the root of the tree
        self.root = Node(list(dataset_labeled.index))
        
        # We delegate the fitting to the root
        self.root.fit(dataset_labeled, [i[0] for i in results], min_samples_split = round(self.min_samples_split*dataset_labeled.shape[0], 0))

        target = []
        for i in dataset_labeled.index :
            label_encoded = dataset_labeled.loc[i]['tree_target']
            label = SubgroupTreeExplainer.decode(label_encoded)
            target.append(label)
        return target
    
    
    def __str__(self):
        return self.root.__str__()
        
    def __len__(self) :
        return self.root.__len__()
    
    def depth(self):
        return self.root.depth()
    
    def min_depth(self):
        return self.root.min_depth()
    
    def leaf_nodes(self):
        return self.root.count_leaf_nodes()
        
    def branching_factor(self) :
        return self.root.branching_factor()
    
    def subset_accuracy(self,dataset,target):
        predicted = []
        for i in dataset.itertuples() :
            predicted.append(self.classify(i))
        
        assert(len(target) == len(predicted))
        accuracy = 0
        for i in range(0,len(predicted)) :
            if type(predicted[i]) is list :
                if predicted[i] == target[i] :
                    accuracy += 1
            # If there is uncertainty about the subgroups which the example belongs.
            elif type(predicted[i]) is dict :
                accuracy += 0
        return float(accuracy)/len(predicted)
    
    def classify(self, example) :
        return self.root.classify(example)
    

    @staticmethod
    def decode(codified):
        codified_int = int(codified)
        #print(codified_int)
        if codified_int == 0 :
            return []
        else:
            n = math.trunc(math.log(codified_int,2))
            decodified = []
            for k in range(n,-1,-1) :
                if codified_int - 2**k  >= 0 :
                    decodified.insert(0,str(k))
                    codified_int = codified_int - 2**k
            
            return decodified
        