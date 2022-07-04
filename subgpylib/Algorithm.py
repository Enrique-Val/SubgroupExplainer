# -*- coding: utf-8 -*-

# Author:
#    Antonio López Martínez-Carrasco <anlopezmc@gmail.com>
#    Enrique Valero Leal <enrique.valero@hotmail.com>

"""This file contains the implementation of some subgroup algorithms (and another needed algorithms).

IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
"""

import string
from .Exception import *
from .Selector import *
from .Subgroup import *
from .QualityMeasure import *
from .DataStructure import *
from pandas import DataFrame
from math import inf, isinf
from sklearn import preprocessing 
import numpy as np

class Algorithm(object):
    """This abstract class represents the parent of all subgroup algorithms.

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
    """

    def __init__(self):
        raise AbstractClassError("This is an abstract class.")

    def fit(self):
        raise AbstractMethodError("This is an abstract mehtod.")


class AlgorithmCN2SD(Algorithm) :
    """This class represents the algorithm CN2SD (Adaption for subgroup discovery of the CN2 algorithm, originally developed by Clark and Niblett).

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.


    :type beam_width: int
    :param beam_width: Width of the beam.
	:type weighting_scheme: 'multiplicative' or 'aditive'
	:param wwighting_scheme: Weighting scheme that the CN2SD will use
    :type gamma: int or float
    :param gamma: Multiplicative value (just for the multiplicative scheme, for the additive scheme mustn't be used)
    :type max_rule_length: int
    :param max_rule_length: The maximum length of the induced subgroups condition
    """
    
    def __init__(self, beam_width, weighting_scheme, gamma = -1, max_rule_length = inf, discretizer = "standard", ncuts = 5) :
        """Method to initialize an object of type 'AlgorithmSD'.
        """
        # Check beam_width parameter
        if type(beam_width) is not int:
            raise TypeError("Parameter 'beam_width' must be an integer (type 'int').")
        if not (beam_width > 0) :
            raise ValueError("Width of the beam must be greater than 0.")
       
        # Check weighting_scheme parameter
        if type(weighting_scheme) is not str :
            raise TypeError("Parameter 'weighting_scheme' must be a string")
        if not (weighting_scheme == 'aditive' or weighting_scheme == 'multiplicative'):
            raise TypeError("Parameter 'weighting_scheme' must be 'aditive' or 'multiplicative'.")

        # Check gamma parameter
        if (type(gamma) is not int) and (type(gamma) is not float) :
            raise TypeError("Parameter 'gamma' must be an integer (type 'int') or a float.")
        if (not (weighting_scheme == 'multiplicative')) and (gamma !=-1) :
            raise TypeError("Parameter 'gamma' is unnecesary for the "+weighting_scheme+" weighting scheme.")
        if (weighting_scheme == 'multiplicative') and (gamma < 0) and (gamma > 1) :
            raise ValueError("Parameter 'gamma' must be in range [0,1].")
        
        # Check max_rule_length parameter
        if (isinf(max_rule_length) and max_rule_length > 0) :
            None # Do nothing, since it is the default value (positive infinite)          
        else :
            if type(max_rule_length) is not int :
                raise TypeError("Parameter 'max_rule_length' must be an integer (type 'int').")
            if max_rule_length < 1 :
                raise TypeError("Parameter 'max_rule_length' must be greater than 0.")
        
        # Check if the discretizer method is will inputed
        if type(discretizer) is not str :
            raise TypeError("Parameter 'discretizer' must be a string")
        if not (discretizer == 'standard' or discretizer == 'ncuts'):
            raise TypeError("Parameter 'discretizer' must be 'standard' or 'ncuts'.")
        
        self.beam_width = beam_width
        self.weighting_scheme = weighting_scheme
        self.gamma = gamma
        self.max_rule_length = max_rule_length
        self.discretizer = discretizer
        self.ncuts = ncuts




    def fit(self, dataset, target_attribute, binary_attributes = []) :
        """Method to run the algorithm CN2 and generate subgroups considering all the values of the target attribute.

        :type dataset: pandas.DataFrame
        :param dataset: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type target_attribute: str
        :param target_attribute: The name of the target attribute. It is VERY IMPORTANT to respect the following conditions:
          (1) the name of the target attribute MUST be a string,
          (2) the name of the target attribute MUST exist in the dataset.
        :type binary_attributes: list
        :param binary_attributes: (OPTIONAL) List of categorical values to be considered as binary. It is VERY IMPORTANT to respect the following conditions:
          (1) binary_attributes must be a list,
          (2) binary_attributes must contain only attributes of pandas_dataframe,
          (3) each attribute of the list must have a maximum of two values.
        :rtype: list
        :return: a list of lists with the k best subgroups (k = beam_width) and its quality measures.
        """
        
        # Check if the parameter 'dataset' is correct
        if type(dataset) is not DataFrame :
            raise TypeError("Parameter 'dataset' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0) :
            raise EmptyDatasetError("The input dataset is empty.")
        
        # Check if the parameter 'target_attribute' is correct
        if  type(target_attribute) is not str :
            raise TypeError("Parameter 'target_attribute' must be string.")
        if target_attribute not in dataset.columns:
            raise ValueError("The name of the target attribute (named "+ target_attribute+") is not an attribute of the input dataset.")
        
        # Check binary_attributes
        if type(binary_attributes) is not list :
            raise TypeError("Parameter 'binary_attributes' must be a list")
        for i in binary_attributes :
            if i not in list(dataset) :
                raise ValueError("Parameter 'binary_attributes' must contain only attributes of 'dataset'")
            elif len(dataset[i].unique()) > 2 :
                raise ValueError("Parameter 'binary_attributes' must contain only the name of attributes with no more than 2 possible values")
   

        # Initialization.     
        # List with the weight of each row of each item (row) of the dataset. Initially it is 1.
        rule_list = []
        
        # Get all the feasible values of a class
        target_values = dataset[target_attribute].unique()
        
        # Get the list of all the selector (if the set of selectors IS dependent on the target_value, you will have to generate it for each target value. This is usually not the case, so we do it here with any value for the target)
        selector_list = self.generateSetOfSelectors(dataset, (target_attribute, target_values[0]), binary_attributes= binary_attributes) 
        
        for c in target_values :
            subgroup_class_c = self.fitOneClass(dataset, (target_attribute, c), binary_attributes = binary_attributes, selectors = selector_list)
            for i, quality in subgroup_class_c:
                rule_list.append((i,quality))
               
        return rule_list

    
    def fitOneClass(self, dataset, target_value, weights = None, binary_attributes = [], selectors = None) :
        """Method to run the algorithm CN2 and generate subgroups considering a value for the target attribute.

        :type dataset: pandas.DataFrame
        :param dataset: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type target_value: tuple
        :param target_value: Tuple with the name of the target attribute (first element) and with the value of this attribute (second element). EXAMPLE1: ("age", 25). EXAMPLE2: ("class", "Setosa"). It is VERY IMPORTANT to respect the following conditions:
          (1) the name of the target attribute MUST be a string,
          (2) the name of the target attribute MUST exist in the dataset,
          (3) it is VERY IMPORTANT to respect the types of the attributes: the value in the tuple (second element) MUST BE comparable with the values of the corresponding attribute in the dataset,
          (4) the value of the target attribute MUST exist in the dataset.
        :type weights: list
        :param weights: List containing the weights of each transaction of the database. If not set, the weight of each item will be 1. The following condition must be respected:
          (1) the name of the target attribute MUST be a list,
          (2) the length of the list attribute MUST be the same that the number of rows of the dataset,
          (3) the elements of the list should be numbers (int or float) in the range [0,1].
        :type binary_attributes: list
        :param binary_attributes: (OPTIONAL) List of categorical values to be considered as binary. It is VERY IMPORTANT to respect the following conditions:
          (1) binary_attributes must be a list,
          (2) binary_attributes must contain only attributes of pandas_dataframe,
          (3) each attribute of the list must have a maximum of two values.
        :rtype: list
        :return: a list of lists with the k best subgroups (k = beam_width) and its quality measures.
        """
        
        # Check if the parameter 'dataset' is correct
        if type(dataset) is not DataFrame:
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0):
            raise EmptyDatasetError("The input dataset is empty.")
        
        # Check if the parameter 'target_value' is correct
        if ( type(target_value) is not tuple ) :
            raise TypeError("Parameter 'target_value' must be a tuple.")
        if (len(target_value) != 2):
            raise ValueError("Parameter 'target_value' must be of length 2.")
        if type(target_value[0]) is not str:
            raise ValueError("The name of the target attribute (first element of the parameter 'target_value') must be a string.")
        if target_value[0] not in dataset.columns:
            raise ValueError("The name of the target attribute (first element of the parameter 'target_value') is not an attribute of the input dataset.")

        #Check if the parameter 'weights' is correct
        if (weights is None) :
            weights = [1] * len(dataset.index)
        else :
            if type(weights) is not list :
                raise TypeError("Parameter 'weights' must be a list")
            if len(weights) != len(dataset.index) : 
                raise TypeError("Parameter 'weights' must have the same number of elements that rows in the dataset")
            for i in weights :
                if (type(i) is int or type(i) is float) and (i > 0 and i < 1) :
                    raise TypeError("Parameter 'weights': The elements of the list must be int or float in range [0,1]")
        
        # Check binary_attributes
        if type(binary_attributes) is not list :
            raise ValueError("Parameter 'binary_attributes' must be a list")
        for i in binary_attributes :
            if i not in list(dataset) :
                raise ValueError("Parameter 'binary_attributes' must contain only attributes of 'dataset'")
            elif len(dataset[i].unique()) > 2 :
                raise ValueError("Parameter 'binary_attributes' must contain only the name of attributes with no more than 2 possible values")
        
        # Check selectors
        # TODO

        selector_list = []
        if selectors is None :
            selector_list = self.generateSetOfSelectors(dataset, target_value, binary_attributes= binary_attributes) 
        else :
            selector_list = selectors
        subgroup_list = []
        
        # Initialization
        while True :
            best_condition, quality = self.__findBestCondition__(dataset, weights, target_value, selector_list)
            if best_condition != Pattern([]) : 
                # We build a subgroup  with the best condition and add it to the list
                subgroup = SubgroupForCN2SD(best_condition, Selector(target_value[0], Selector.OPERATOR_EQUAL, target_value[1]))
                if subgroup not in [[ i for i, j in subgroup_list ], [ j for i, j in subgroup_list ]][0]  :
                    subgroup_list.append((subgroup, quality))
                
                # Apply the covering algorithm
                # First, we need to get the association between pandas indexing and array indexing
                index = self.__getIndexDictionary__(dataset)  
                if self.weighting_scheme == 'aditive' :
                    # Aditive covering algorithm
                    i = 0
                    for row in dataset.itertuples(False) :
                        if subgroup.matchElement(row, index) :
                            if weights[i] > 0:
                                weights[i] = 1/(1/weights[i] + 1)
                                if weights[i] < 0.1 :
                                    weights[i] = 0
                        i = i + 1
                
                elif self.weighting_scheme == 'multiplicative' :
                    # Multiplicative covering algorithm (using gamma)
                    i = 0
                    for row in dataset.itertuples(False) :
                        if subgroup.matchElement(row, index) :
                            weights[i] = weights[i] * self.gamma
                            if weights[i] < 0.1 :
                                weights[i] = 0
                        i = i + 1
            else :
                break
            
        
        return subgroup_list
            
    def __findBestCondition__(self, dataset, weights, target_value, selector_list):
        target_selector = Selector(target_value[0], Selector.OPERATOR_EQUAL, target_value[1])
        # List of potential conditions for the induced subgroup (type = list of Pattern).
        # It should be initialized as empty, but if we do so, we won't be able to iterate over it the first time
        # Being so, we decided to put and empty condition (empty Pattern)
        beam = [Pattern([])]
        # Best condition found (type = Pattern). Initialized as empty             
        best_condition = Pattern([])
        # WRAcc associated to the best condition. Initially 0, since WRAcc([] -> target) = 0
        best_WRAcc = 0
        size = 0
        while True :
            
            '''
            print()
            print("ITERATION "+str(size))
            print("Beam:")
            for b in beam :
                print(b)
            '''
            new_beam = []
            # Create new_beam = x^y, where x belongs to beam and y belongs to the set of all possible selectors
            for b in beam :
                for selector in selector_list :
                    new_b = b.copy()
                    new_b.addSelector(selector)
                    new_b_WRAcc = self.__getModifiedWRAcc__(dataset, weights, SubgroupForCN2SD(new_b, target_selector))
                    if new_b not in beam and new_b not in new_beam and new_b_WRAcc != 0:
                        # Do an ordered insertion with some characteristics. Not specified in the pseudocode, but it will improve the efficiency:
                        #    Just add the subgroup new_b to new_beam if it will improve new_beam or the length of new_beam is lower than the user specified maximum beam width
                        i = 0
                        while i < len(new_beam) :
                            i_WRAcc = self.__getModifiedWRAcc__(dataset, weights, SubgroupForCN2SD(new_beam[i], target_selector))
                            if new_b_WRAcc > i_WRAcc :
                                break
                            i = i + 1
                        new_beam.insert(i,new_b)
                        if len(new_beam) > self.beam_width :
                            new_beam.pop(self.beam_width)

            # Remove from new_beam the elements in beam
            # Done while iterating
            
            # Remove from new_beam the null elements (ex. age = 5 and age = 7)
            # Not yet implemented. Probably another possibility will be studied, since these calculations are quite complex
            # The solution taken right now consists on ignoring the subgroups with WRAcc <= 0 (done while iterating)

            '''
            print("New beam (best "+str(self.beam_width)+ " conditions)")
            for b in new_beam :
                print(b)
                print("     WRAcc: " +  str(self.__getModifiedWRAcc__(dataset, weights, SubgroupForCN2SD(b,target_selector))))
            print()
            '''   

            
            # If the best element of the new beam (if there is one) is better that our best condition so forth,
            # We will replace the best condition by this element. This element is the first, since new_beam is ordered
            # according to its WRAcc    
            if new_beam != [] :
                new_beam_best_WRAcc = self.__getModifiedWRAcc__(dataset, weights, SubgroupForCN2SD(new_beam[0], target_selector))
                if new_beam_best_WRAcc > best_WRAcc : # and SubgroupForCN2SD(new_beam[0], target_selector) not in founded :
                    best_condition = new_beam[0]
                    best_WRAcc = new_beam_best_WRAcc
            
            '''
            print("Best condition")
            print(best_condition)
            print("     WRAcc: " +  str(self.__getModifiedWRAcc__(dataset, weights, SubgroupForCN2SD(best_condition, target_selector))))
            '''
            
            
            # Remove the worst elements in new_beam until its size == beam_width (user-defined size of the beam)
            new_beam = new_beam[:self.beam_width]      
              
            # Let beam be the new_beam  
            beam = new_beam
            size = size + 1
            
            # Repeat until no elements in beam
            if beam == [] or size >= self.max_rule_length :
                break
        
        '''
        print ("End find best condition")
        print("-------------------------")
        '''
        
        return (best_condition, self.__getModifiedWRAcc__(dataset, weights, SubgroupForCN2SD(best_condition, target_selector)))
    
    
    def generateSetOfSelectors(self, pandas_dataframe, tuple_target_attribute_value, binary_attributes = []):
        """Method to generate the set of all feasible attribute values (set of features L) used in SD algorithm.

        :type pandas_dataframe: pandas.DataFrame
        :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type tuple_target_attribute_value: tuple
        :param tuple_target_attribute_value: Tuple with the name of the target attribute (first element) and with the value of this attribute (second element). EXAMPLE1: ("age", 25). EXAMPLE2: ("class", "Setosa"). It is VERY IMPORTANT to respect the following conditions:
          (1) the name of the target attribute MUST be a string,
          (2) the name of the target attribute MUST exist in the dataset,
          (3) it is VERY IMPORTANT to respect the types of the attributes: the value in the tuple (second element) MUST BE comparable with the values of the corresponding attribute in the dataset,
          (4) the value of the target attribute MUST exist in the dataset.
        :rtype: list
        :return: all feasible attribute values (set of features L) used in SD algorithm (stored in a list).
        """
        if type(pandas_dataframe) is not DataFrame:
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        if (pandas_dataframe.shape[0] == 0) or (pandas_dataframe.shape[1] == 0):
            raise EmptyDatasetError("The input dataset is empty.")
        if ( type(tuple_target_attribute_value) is not tuple ):
            raise TypeError("Parameter 'tuple_target_attribute_value' must be a tuple.")
        if (len(tuple_target_attribute_value) != 2):
            raise ValueError("Parameter 'tuple_target_attribute_value' must be of length 2.")
        if type(tuple_target_attribute_value[0]) is not str:
            raise ValueError("The name of the target attribute (first element in parameter 'tuple_target_attribute_value') must be a string.")
        if tuple_target_attribute_value[0] not in pandas_dataframe.columns:
            raise ValueError("The name of the target attribute (first element in parameter 'tuple_target_attribute_value') is not an attribute of the input dataset.")
        # Variable to store all selectors of set of features L.
        #   - It is very important to AVOID DUPLICATED Selectors. So, we will use a PYTHON DICTIONARY where the key is the selector and the value is None (the value does not matter).
        final_set_l = dict()
        # We generate the set L.
        columns_without_target = pandas_dataframe.columns[pandas_dataframe.columns != tuple_target_attribute_value[0]]
        first_element_index = pandas_dataframe.index[0]
        for column in columns_without_target: # Iterate over dataframe column names, except target column name.
            # We check the possible types of the values of the column.
            #   - The type of the strings in a pandas DataFrame is directly 'str'. 
            #   - If the element gotten with 'loc' is not 'str', we have to use 'item' method to get the "primitive element" (element of the primitive type).
            if (type(pandas_dataframe[column].iloc[0]) is str): # Only check the first element, because all elements of the column are of the same type.
                if column in binary_attributes :
                    bin_values = pandas_dataframe[column]
                    for value in bin_values :
                        final_set_l[ Selector(column, Selector.OPERATOR_EQUAL, value) ] = None
                else :
                    index_dict = self.__getIndexDictionary__(pandas_dataframe)
                    for row in pandas_dataframe.itertuples(False):
                        if (row[index_dict[tuple_target_attribute_value[0]]] == tuple_target_attribute_value[1]): # If the example/row is positive.
                            final_set_l[ Selector(column, Selector.OPERATOR_EQUAL, row[index_dict[column]]) ] = None
                        elif (row[index_dict[tuple_target_attribute_value[0]]] != tuple_target_attribute_value[1]): # If the example/row is negative.
                            final_set_l[ Selector(column, Selector.OPERATOR_NOT_EQUAL, row[index_dict[column]]) ] = None
            elif self.discretizer == "ncuts" and ((type(pandas_dataframe[column].iloc[0].item()) is float) or (type(pandas_dataframe[column].iloc[0].item()) is int)) : 
                # Reshape the column
                mod = pandas_dataframe[column]
                mod = np.array(mod)
                mod = mod.reshape(-1,1)
                
                # Find 5 (actually 7) cuts. n, for parametrization
                est = preprocessing.KBinsDiscretizer(n_bins=self.ncuts, encode='ordinal')
                est.fit(mod)
                X = est.transform(mod)
                print(est.bin_edges_[0])
                
                for i in range(1,len(est.bin_edges_[0])-1) :
                    #final_set_l[ Selector(column, Selector.OPERATOR_EQUAL, est.bin_edges_[0][i]) ] = None
                    #final_set_l[ Selector(column, Selector.OPERATOR_NOT_EQUAL, est.bin_edges_[0][i]) ] = None
                    final_set_l[ Selector(column, Selector.OPERATOR_LESS_OR_EQUAL, float(est.bin_edges_[0][i])) ] = None
                    final_set_l[ Selector(column, Selector.OPERATOR_GREATER, float(est.bin_edges_[0][i])) ] = None
                    #print(intervals[i])
            elif (type(pandas_dataframe[column].iloc[0].item()) is float) and self.discretizer == "standard":
                # If the attribute is continuous, we have to get the positive examples and the negative examples.
                pandas_dataframe_positive_examples = pandas_dataframe[ pandas_dataframe[tuple_target_attribute_value[0]] == tuple_target_attribute_value[1] ]
                pandas_dataframe_negative_examples = pandas_dataframe[ pandas_dataframe[tuple_target_attribute_value[0]] != tuple_target_attribute_value[1] ]
                # We generate all possible pairs with the positive and negative examples.
                index_dict_positive_examples = self.__getIndexDictionary__(pandas_dataframe_positive_examples)
                index_dict_negative_examples = self.__getIndexDictionary__(pandas_dataframe_negative_examples)
                for positive_example_row in pandas_dataframe_positive_examples.itertuples(False):
                    for negative_example_row in pandas_dataframe_negative_examples.itertuples(False):
                        final_set_l[ Selector(column, Selector.OPERATOR_LESS_OR_EQUAL, (positive_example_row[index_dict_positive_examples[column]]+negative_example_row[index_dict_negative_examples[column]])/2) ] = None
                        final_set_l[ Selector(column, Selector.OPERATOR_GREATER, (positive_example_row[index_dict_positive_examples[column]]+negative_example_row[index_dict_negative_examples[column]])/2) ] = None
            elif (type(pandas_dataframe[column].iloc[0].item()) is int) and self.discretizer == "standard":
                # If the attribute is an integer, we have to get the positive examples and the negative examples.
                pandas_dataframe_positive_examples = pandas_dataframe[ pandas_dataframe[tuple_target_attribute_value[0]] == tuple_target_attribute_value[1] ]
                pandas_dataframe_negative_examples = pandas_dataframe[ pandas_dataframe[tuple_target_attribute_value[0]] != tuple_target_attribute_value[1] ]
                # We generate all possible pairs with the positive and negative examples.
                index_dict_positive_examples = self.__getIndexDictionary__(pandas_dataframe_positive_examples)
                index_dict_negative_examples = self.__getIndexDictionary__(pandas_dataframe_negative_examples)
                for positive_example_row in pandas_dataframe_positive_examples.itertuples(False):
                    for negative_example_row in pandas_dataframe_negative_examples.itertuples(False):
                        final_set_l[ Selector(column, Selector.OPERATOR_LESS_OR_EQUAL, (positive_example_row[index_dict_positive_examples[column]]+negative_example_row[index_dict_negative_examples[column]])/2) ] = None
                        final_set_l[ Selector(column, Selector.OPERATOR_GREATER, (positive_example_row[index_dict_positive_examples[column]]+negative_example_row[index_dict_negative_examples[column]])/2) ] = None
                        final_set_l[ Selector(column, Selector.OPERATOR_EQUAL, positive_example_row[index_dict_positive_examples[column]]) ] = None
                        final_set_l[ Selector(column, Selector.OPERATOR_NOT_EQUAL, negative_example_row[index_dict_negative_examples[column]]) ] = None
        # In variable 'final_set_l', we do not have duplicates. Now, we have to return it as list.
        return list(final_set_l)        
            
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
    
    def __getModifiedWRAcc__(self, dataset, weights, subgroup): 
        """Internal method to get the modified WRAcc of a subgroup given a dataset with its rows weighted (as described by Lavrac, 2004)

        It is VERY IMPORTANT to respect the types of the attributes: the value of a selector of the subgroup MUST BE comparable with the value of the corresponding attribute in the dataset.
        
        :type pandas_dataframe: pandas.DataFrame
        :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type weights: list
        :param weights: List containing the weights of each transaction of the database. If not set, the weight of each item will be 1. The following condition must be respected:
          (1) the name of the target attribute MUST be a list,
          (2) the length of the list attribute MUST be the same that the number of rows of the dataset,
          (3) the elements of the list should be numbers (int or float) in the range [0,1].
        :type subgroup: Subgroup
        :param subgroup: Input subgroup.
        :rtype: float
        :return: The modified WRAcc of the subgroup for the weighted database.
        """
        # Check if the parameter 'dataset' is correct
        if type(dataset) is not DataFrame:
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0):
            raise EmptyDatasetError("The input dataset is empty.")
        
        #Check if the parameter 'weights' is correct
        if (weights is None) :
            weights = [1] * len(dataset.index)
        else :
            if type(weights) is not list :
                raise TypeError("Parameter 'weights' must be a list")
            if len(weights) != len(dataset.index) : 
                raise TypeError("Parameter 'weights' must have the same number of elements that rows in the dataset")
            for i in weights :
                if not ((type(i) is int or type(i) is float) and (i >= 0 and i <= 1)) :
                    print(weights)
                    raise TypeError("Parameter 'weights': The elements of the list must be int or float in range [0,1]")

        # Check if the 'parameter' subgroup is correct
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
                tp = tp + weights[row_index]         
            if (subgroup_condition_and_row_match) and (not subgroup_target_and_row_match):
                fp = fp + weights[row_index]
            if subgroup_target_and_row_match:
                TP = TP + weights[row_index]
            if not subgroup_target_and_row_match:
                FP = FP + weights[row_index]
                  
            row_index = row_index + 1
         
        '''
        print("tp "+str(tp))
        print("fp "+str(fp))
        print("TP "+str(TP))
        print("FP "+str(FP))
        q = QualityMeasure
        '''
        if (tp+fp == 0) :
            return 0
        return ( (tp+fp) / (TP+FP))  * (  tp / (tp+fp)  -  TP / (TP+FP ) )        
        
        
        
class AlgorithmSD4TS(Algorithm) :  
    """This class represents the algorithm SD (Subgroup Discovery).

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.

    :type min_support: int or float
    :param min_support: Minimum support that need to have a SUBGROUP to be considered. Value in form of PROPORTION (between 0 and 1).
    :type max_subgroups: int
    :param max_subgroups: Maximum number of subgroups to return.
    """

    def __init__(self, min_support, max_subgroups, discretizer = "ncuts", ncuts = 5):
        """Method to initialize an object of type 'AlgorithmSD'.
        """
        if (type(min_support) is not int) and (type(min_support) is not float):
            raise TypeError("Parameter 'min_support' must be an integer (type 'int') or a float.")
        if min_support < 0 or min_support > 1:
            raise ValueError("The minimum support must be a number in range [0,1].")
        if type(max_subgroups) is not int:
            raise TypeError("Parameter 'max_subgroups' must be an integer (type 'int').")
        if not (max_subgroups > 0):
            raise ValueError("The maximum number of produced subgroups must be greater than 0.")
        self.minSupport = min_support
        self.max_subgroups = max_subgroups
        self.discretizer = discretizer
        self.ncuts = ncuts


    def fit(self, dataset, target_attribute, binary_attributes = []):
        """Method to run the algorithm SD4TS and generate subgroups.

        :type pandas_dataframe: pandas.DataFrame
        :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type tuple_target_attribute_value: tuple
        :param tuple_target_attribute_value: Tuple with the name of the target attribute (first element) and with the value of this attribute (second element). EXAMPLE1: ("age", 25). EXAMPLE2: ("class", "Setosa"). It is VERY IMPORTANT to respect the following conditions:
          (1) the name of the target attribute MUST be a string,
          (2) the name of the target attribute MUST exist in the dataset,
          (3) it is VERY IMPORTANT to respect the types of the attributes: the value in the tuple (second element) MUST BE comparable with the values of the corresponding attribute in the dataset,
          (4) the value of the target attribute MUST exist in the dataset.
        :param binary_attributes: (OPTIONAL) List of categorical values to be considered as binary. It is VERY IMPORTANT to respect the following conditions:
          (1) binary_attributes must be a list,
          (2) binary_attributes must contain only attributes of pandas_dataframe,
          (3) each attribute of the list must have a maximum of two values.
        :rtype: list
        :return: a list of lists with the 'max_subgroups' best subgroups
        """
        if type(dataset) is not DataFrame:
            raise TypeError("Parameter 'dataset' must be a pandas DataFrame.")
        if (dataset.shape[0] == 0) or (dataset.shape[1] == 0):
            raise EmptyDatasetError("The input dataset is empty.")
        if type(target_attribute) is not str:
            raise TypeError("The name of the target_attribute must be a string.")
        if target_attribute not in dataset.columns:
            raise ValueError("The name target_attribute is not an attribute of the input dataset.")
        
        # Check binary_attributes
        if type(binary_attributes) is not list :
            raise TypeError("Parameter 'binary_attributes' must be a list")
        for i in binary_attributes :
            if i not in list(dataset) :
                raise ValueError("Parameter 'binary_attributes' must contain only attributes of 'dataset'")
            elif len(dataset[i].unique()) > 2 :
                raise ValueError("Parameter 'binary_attributes' must contain only the name of attributes with no more than 2 possible values")
   

        min_quality = 0
        classes = dataset[target_attribute].unique()
        # Algorithm
        # optimizable_candidates <- any 1 attribute-value-pair that satisfy the min_support threshold
        optimizable_candidates = []
        characteristics = self.generateSetOfSelectors(dataset, target_attribute, binary_attributes = binary_attributes)
        still2test = []
        for c in classes :
            target_selector = Selector(target_attribute,Selector.OPERATOR_EQUAL, c)
            for i in characteristics :
                subg_i = SubgroupForSD4TS(Pattern([i]),target_selector)
                dict_metrics_i = self.obtainDictOfMetrics(dataset, subg_i)
                if QualityMeasureSupport().compute(dict_metrics_i) >= self.minSupport :
                    still2test.append(i)
                    i_WRAcc = QualityMeasureWRAcc().compute(dict_metrics_i)
                    j = 0
                    while j < len(optimizable_candidates) :
                        j_WRAcc = self.__getQualityMeasure__(dataset, optimizable_candidates[j])
                        if i_WRAcc > j_WRAcc :
                            break
                        j = j + 1
                    optimizable_candidates.insert(j,subg_i)
                
        # top_candidates <- best 'max_subgroups' subgroups (taking a quality measure) from optimizable_candidates
        top_candidates = optimizable_candidates[:self.max_subgroups]
        
        # min_quality <- worst quality measure value from the top_candidates set
        if len(top_candidates) != 0 :
            min_quality = self.__getQualityMeasure__(dataset, top_candidates[-1])
        
        # Remove subgroups from optimizable_candidates that do not satisfy min_quality
        i = len(top_candidates)
        while i < len(optimizable_candidates) :
            if self.__getQualityMeasure__(dataset, optimizable_candidates[i]) < min_quality :
                optimizable_candidates = optimizable_candidates[:i]
                break
            i = i + 1
        '''
        # Printing
        print()
        print("Top candidates")
        for c in top_candidates :
            print("   "+str(c))
            print("   Quality: "+str(self.__getQualityMeasure__(dataset, c)))
        print()
        
        print("Optimizable candidates")
        for c in optimizable_candidates :
            print("   "+str(c))
            print("   Quality: "+str(self.__getQualityMeasure__(dataset, c)))
        print()
        print("min_quality: "+str(min_quality))
        print()  
        '''
        
        # while optimizable_candidates is not empty do
        while optimizable_candidates :
            # c1 ←take best candidate from optimizable_candidates
            c1 = optimizable_candidates.pop(0)
                    
            # for ∀c2 in  still2test do
            for c2 in still2test :
                
                # c_new <- generate new candidates from c1 and c2
                if (c2 in c1.getCondition().getListOfSelectors()) :
                    continue
                c_new = SubgroupForSD4TS(c1.getCondition() + Pattern([c2]), target_selector)       
                
                # if support (c_new, dataset) > min_support and qualityMeasure(c_new,dataset) > min_quality then
                dict_metrics_c_new = self.obtainDictOfMetrics(dataset, c_new)
                support_c_new = QualityMeasureSupport().compute(dict_metrics_c_new)
                quality_c_new = QualityMeasureWRAcc().compute(dict_metrics_c_new)

                if support_c_new > self.minSupport and quality_c_new > min_quality and c_new not in optimizable_candidates :
                    # optimizable_candidates ← c_new
                    i = 0
                    while i < len(optimizable_candidates) :
                        quality_i = self.__getQualityMeasure__(dataset, optimizable_candidates[i])
                        if quality_c_new > quality_i :
                            break
                        i = i + 1
                    optimizable_candidates.insert(i,c_new)
                    
                    # if c_new is better than the worst from TopCandidates then
                    if quality_c_new > self.__getQualityMeasure__(dataset, top_candidates[-1]) and c_new not in top_candidates :
                    
                        # top_candidates <- cnew
                        i = 0
                        while i < len(top_candidates) :
                            quality_i = self.__getQualityMeasure__(dataset, top_candidates[i])
                            if quality_c_new > quality_i :
                                break
                            i = i + 1
                        top_candidates.insert(i,c_new)
                        
                        # if size of top_candidates is greater than max_subgroups then
                        if len(top_candidates) > self.max_subgroups :
                        
                            # Remove the worst solution from top_candidates
                            top_candidates.pop()
                            
                        # end if
                        # min_quality <- worst quality measure value from the top_candidates set
                        min_quality = self.__getQualityMeasure__(dataset, top_candidates[-1])
                        
                        # Remove subgroups from optimizable_candidates that do not satisfy min_quality
                        # We will do it outside the loop "for c2 in still2test", for better efficiency
                        

                        
                    # end if
                # end if
            # end for
            # Remove subgroups from optimizable_candidates that do not satisfy min_quality
            i = 0
            while i < len(optimizable_candidates) :
                if self.__getQualityMeasure__(dataset, optimizable_candidates[i]) < min_quality :
                    optimizable_candidates =optimizable_candidates[:i]
                    break
                i = i + 1
        # end while
        # return top_candidates
        return [(i,self.__getQualityMeasure__(dataset,i)) for i in top_candidates]
    
    
    def generateSetOfSelectors(self, pandas_dataframe, tuple_target_attribute_value, binary_attributes = []):
        """Method to generate the set of all feasible attribute values (set of features L) used in SD algorithm.

        :type pandas_dataframe: pandas.DataFrame
        :param pandas_dataframe: Input dataset. It is VERY IMPORTANT to respect the following conditions:
          (1) the dataset must be a pandas dataframe,
          (2) the dataset must not contain missing values,
          (3) for each attribute, all its values must be of the same type.
        :type tuple_target_attribute_value: tuple
        :param tuple_target_attribute_value: Tuple with the name of the target attribute (first element) and with the value of this attribute (second element). EXAMPLE1: ("age", 25). EXAMPLE2: ("class", "Setosa"). It is VERY IMPORTANT to respect the following conditions:
          (1) the name of the target attribute MUST be a string,
          (2) the name of the target attribute MUST exist in the dataset,
          (3) it is VERY IMPORTANT to respect the types of the attributes: the value in the tuple (second element) MUST BE comparable with the values of the corresponding attribute in the dataset,
          (4) the value of the target attribute MUST exist in the dataset.
        :param binary_attributes: (OPTIONAL) List of categorical values to be considered as binary. It is VERY IMPORTANT to respect the following conditions:
          (1) binary_attributes must be a list,
          (2) binary_attributes must contain only attributes of pandas_dataframe,
          (3) each attribute of the list must have a maximum of two values.
        :rtype: list
        :return: all feasible attribute values (set of features L) used in SD algorithm (stored in a list).
        """
        if type(pandas_dataframe) is not DataFrame:
            raise TypeError("Parameter 'pandas_dataframe' must be a pandas DataFrame.")
        if (pandas_dataframe.shape[0] == 0) or (pandas_dataframe.shape[1] == 0):
            raise EmptyDatasetError("The input dataset is empty.")
        if ( type(tuple_target_attribute_value) is not str ):
            raise TypeError("Parameter 'tuple_target_attribute_value' must be a string.")
        if tuple_target_attribute_value not in pandas_dataframe.columns:
            raise ValueError("The name of the target attribute is not an attribute of the input dataset.")
        # Variable to store all selectors of set of features L.
        #   - It is very important to AVOID DUPLICATED Selectors. So, we will use a PYTHON DICTIONARY where the key is the selector and the value is None (the value does not matter).
        final_set_l = []
        # We generate the set L.
        columns_without_target = pandas_dataframe.columns[pandas_dataframe.columns != tuple_target_attribute_value]
        for column in columns_without_target: # Iterate over dataframe column names, except target column name.
            # We check the possible types of the values of the column.
            #   - The type of the strings in a pandas DataFrame is directly 'str'. 
            #   - If the element gotten with 'loc' is not 'str', we have to use 'item' method to get the "primitive element" (element of the primitive type).
            if (type(pandas_dataframe[column].iloc[0]) is str): # Only check the first element, because all elements of the column are of the same type.
                diff_values = pandas_dataframe[column].unique()
                for value in diff_values : 
                    final_set_l.append(Selector(column, Selector.OPERATOR_EQUAL, value))
                if column not in binary_attributes :
                    for value in diff_values :
                        final_set_l.append(Selector(column, Selector.OPERATOR_NOT_EQUAL, value))
            elif self.discretizer == "ncuts" and ((type(pandas_dataframe[column].iloc[0].item()) is float) or (type(pandas_dataframe[column].iloc[0].item()) is int)) : 
                # Reshape the column
                mod = pandas_dataframe[column]
                mod = np.array(mod)
                mod = mod.reshape(-1,1)
                
                # Find 5 (actually 7) cuts. n, for parametrization
                est = preprocessing.KBinsDiscretizer(n_bins=self.ncuts, encode='ordinal')
                est.fit(mod)
                X = est.transform(mod)
                print(est.bin_edges_[0])
                
                for i in range(1,len(est.bin_edges_[0])-1) :
                    #final_set_l[ Selector(column, Selector.OPERATOR_EQUAL, est.bin_edges_[0][i]) ] = None
                    #final_set_l[ Selector(column, Selector.OPERATOR_NOT_EQUAL, est.bin_edges_[0][i]) ] = None
                    final_set_l.append(Selector(column, Selector.OPERATOR_LESS_OR_EQUAL, float(est.bin_edges_[0][i])))
                    final_set_l.append(Selector(column, Selector.OPERATOR_GREATER, float(est.bin_edges_[0][i])))
                    #print(intervals[i])
        
        # In variable 'final_set_l', we do not have duplicates. Now, we have to return it as list.
        return final_set_l     
            
    def __getIndexDictionary__(self, dataset):
        """Auxiliary method to calculate de index dictionary, a data structure that maps each column name of a pandas dataframe into its integer value.

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
    
    def __getQualityMeasure__(self,dataset,subgroup) :
        return QualityMeasureWRAcc().compute(self.obtainDictOfMetrics(dataset, subgroup))
    
    
    
    
           
