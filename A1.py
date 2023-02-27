#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, ensemble, metrics, svm, model_selection, linear_model


# In[ ]:


def training_test_split(X, y, test_size=0.3, random_state=None):
    """ Split the features X and labels y into training and test features and labels. 
    
    `split` indicates the fraction (rounded down) that should go to the test set.

    `random_state` allows to set a random seed to make the split reproducible. 
    If `random_state` is None, then no random seed will be set.
    
    """
     # Determine the number of instances in the dataset and determine the number of instances that should be allocated to the test set
    n_instances = len(X)
    n_test_instances = int(np.floor(test_size * n_instances))
    
    # Set the random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Randomly shuffle the indices of the instances
    shuffled_indices = np.random.permutation(n_instances)
    
    # Select the instances that will be in the training and test sets
    test_indices = shuffled_indices[:n_test_instances]
    train_indices = shuffled_indices[n_test_instances:]
    
    # Split the data into training and test sets
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # raise NotImplementedError('Your code here')
    return X_train, X_test, y_train, y_test


# In[ ]:


def true_negatives(true_labels, predicted_labels, positive_class):
    neg_true = true_labels != positive_class  # actually negative class
    neg_predicted = predicted_labels != positive_class # predicted to be negative class
    match = neg_true & neg_predicted # use logical AND (that's the `&`) to find elements that are True in both arrays
    #raise NotImplementedError('Your code here')
    return np.sum(match)  # count them


def false_negatives(true_labels, predicted_labels, positive_class):
    neg_predicted = predicted_labels != positive_class  # predicted to be negative class
    pos_true = true_labels == positive_class  # actually positive class
    match = neg_predicted & pos_true  # The `&` is element-wise logical AND
    #raise NotImplementedError('Your code here')
    return np.sum(match)  # count the number of matches


# In[ ]:


def recall(true_labels, predicted_labels, positive_class):
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return TP / (TP + FN)


# In[ ]:


def accuracy(true_labels, predicted_labels, positive_class):
    FP = false_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    TP = true_positives(true_labels, predicted_labels, positive_class)
    FN = false_negatives(true_labels, predicted_labels, positive_class)
    return (TP + TN)/(FP+TN+TP+FN)


# In[ ]:


def specificity(true_labels, predicted_labels, positive_class):
    FP = false_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    return TN/(FP+TN)


# In[ ]:


def balanced_accuracy(true_labels, predicted_labels, positive_class):
    FP = false_positives(true_labels, predicted_labels, positive_class)
    TN = true_negatives(true_labels, predicted_labels, positive_class)
    return TN/(FP+TN)


# In[ ]:


def F1(true_labels, predicted_labels, positive_class):
    #had to add a 1 because 'precision/recall' was already mentioned in previous statements.
    precision1 = precision(true_labels, predicted_labels, positive_class)
    recall1 = recall(true_labels, predicted_labels, positive_class)
    return 2*(precision1*recall1)/(precision1+recall1)


# In[ ]:


#Could not find exercise 8 anymore?? But I will upload the not skeleton version as well as A1Leo.

