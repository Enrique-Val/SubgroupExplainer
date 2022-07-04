# -*- coding: utf-8 -*-

# Author:
#    Antonio López Martínez-Carrasco <anlopezmc@gmail.com>

"""This file contains the implementation of quality measures.

IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
"""

from .Exception import *

class QualityMeasure(object):
    """This abstract class represents the parent of all quality measures.

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
    """

    # VERY IMPORTANT: We only consider these basic metrics.
    BasicMetric_tp = "tp"
    BasicMetric_fp = "fp"
    BasicMetric_TP = "TP"
    BasicMetric_FP = "FP"

    def __init__(self):
        raise AbstractClassError("This is an abstract class.")

    def compute(self, dict_of_basic_metrics):
        raise AbstractMethodError("This is an abstract mehtod.")



class QualityMeasureWRAcc(QualityMeasure):
    """This class represents the quality measure 'WRAcc'.

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
    """

    def __init__(self):
        pass

    def compute(self, dict_of_basic_metrics):
        """Method to compute the quality measure 'WRAcc'.
        
        :type dict_of_basic_metrics: dict
        :param dict_of_basic_metrics: Python dictionary that contains all needed basic metrics to compute this quality measure.
        :rtype: float
        :return: the value of the quality measure.
        """
        if type(dict_of_basic_metrics) is not dict:
            raise TypeError("Parameter 'dict_of_basic_metrics' must be a python dictionary.")
        if (QualityMeasure.BasicMetric_tp not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'tp' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_fp not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'fp' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_TP not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'TP' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_FP not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'FP' is not in 'dict_of_basic_metrics'.")
        
        if dict_of_basic_metrics["tp"]+dict_of_basic_metrics["fp"] == 0 :
            return 0
        
        return ( (dict_of_basic_metrics["tp"]+dict_of_basic_metrics["fp"]) / (dict_of_basic_metrics["TP"]+dict_of_basic_metrics["FP"]) ) * ( ( dict_of_basic_metrics["tp"] / (dict_of_basic_metrics["tp"]+dict_of_basic_metrics["fp"]) ) - ( dict_of_basic_metrics["TP"] / (dict_of_basic_metrics["TP"]+dict_of_basic_metrics["FP"]) ) ) # ( (tp+fp) / (TP+FP) ) * ( ( tp / (tp+fp) ) - ( TP / (TP+FP) ) )
    
    
class QualityMeasureRAcc(QualityMeasure):
    """This class represents the quality measure 'Relative Accuracy.

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
    """

    def __init__(self):
        pass

    def compute(self, dict_of_basic_metrics):
        """Method to compute the quality measure 'WRAcc'.
        
        :type dict_of_basic_metrics: dict
        :param dict_of_basic_metrics: Python dictionary that contains all needed basic metrics to compute this quality measure.
        :rtype: float
        :return: the value of the quality measure.
        """
        if type(dict_of_basic_metrics) is not dict:
            raise TypeError("Parameter 'dict_of_basic_metrics' must be a python dictionary.")
        if (QualityMeasure.BasicMetric_tp not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'tp' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_fp not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'fp' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_TP not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'TP' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_FP not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'FP' is not in 'dict_of_basic_metrics'.")
        
        if dict_of_basic_metrics["tp"]+dict_of_basic_metrics["fp"] == 0 :
            return 0
        
        return ( ( dict_of_basic_metrics["tp"] / (dict_of_basic_metrics["tp"]+dict_of_basic_metrics["fp"]) ) - ( dict_of_basic_metrics["TP"] / (dict_of_basic_metrics["TP"]+dict_of_basic_metrics["FP"]) ) ) #( ( tp / (tp+fp) ) - ( TP / (TP+FP) ) )
    
    
class QualityMeasureAccuracy(QualityMeasure):
    """This class represents the quality measure 'Relative Accuracy.

    IMPORTANT NOTE: You must not access directly to the attributes of the objects. You must use the corresponding methods.
    """

    def __init__(self):
        pass

    def compute(self, dict_of_basic_metrics):
        """Method to compute the quality measure 'WRAcc'.
        
        :type dict_of_basic_metrics: dict
        :param dict_of_basic_metrics: Python dictionary that contains all needed basic metrics to compute this quality measure.
        :rtype: float
        :return: the value of the quality measure.
        """
        if type(dict_of_basic_metrics) is not dict:
            raise TypeError("Parameter 'dict_of_basic_metrics' must be a python dictionary.")
        if (QualityMeasure.BasicMetric_tp not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'tp' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_fp not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'fp' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_TP not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'TP' is not in 'dict_of_basic_metrics'.")
        if (QualityMeasure.BasicMetric_FP not in dict_of_basic_metrics):
            raise BasicMetricNotFoundError("The basic metric 'FP' is not in 'dict_of_basic_metrics'.")
        
        if dict_of_basic_metrics["tp"]+dict_of_basic_metrics["fp"] == 0 :
            return 0
        
        return ( dict_of_basic_metrics["tp"] / (dict_of_basic_metrics["tp"] + dict_of_basic_metrics["fp"]) ) # tp / tp + fp
