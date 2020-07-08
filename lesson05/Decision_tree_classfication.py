# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:49:37 2020

@author: xiangguchn
"""

import numpy as np
from collections import Counter
import pandas as pd
import operator

def entropy(elements):
    # count how many kind elements and their number
    counter = Counter(elements)
    # calculate probalities for every elements
    probs = [counter[c] / len(elements) for c in elements]
    # calculate entropy of elements
    return - sum(p * np.log2(p) for p in probs)

# find the min spliter of train data
def find_the_min_spilter(training_data: pd.DataFrame, target: str) -> str:
    
    # all fields of training_data except target one
    x_fields = set(training_data.columns.tolist()) - {target}
    # set a inital value for spliter and min_entropy
    spliter = None
    min_entropy = float('inf')
    
    # try every field and value to find one with min entropy
    for f in x_fields:
        # values of one field in training_data
        values = set(training_data[f])
        nf = len(training_data[f])
        # 
        for v in values:
            # Generate child nodes
            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()
            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()
            # length of every nodes
            n1 = len(sub_spliter_1)
            n2 = len(sub_spliter_2)
            # calculate entropy of v in f
            entropy_v = n1/nf*entropy(sub_spliter_1) + n2/nf*entropy(sub_spliter_2)  # change here 
            
            # replace spliter and min_entropy if spliter with smaller entropy is found
            if entropy_v <= min_entropy:
                min_entropy = entropy_v
                spliter = (f, v)
                
    return spliter, min_entropy


# generate sub train data for decision tree nodes
def sub_train_data_gen(train_data,spliter0,value):
    ssub = [i for i in range(len(train_data[spliter0])) if train_data[spliter0][i] == value]
    sub_train_data = {}
    keys = list(train_data.keys())
    keys.remove(spliter0)
    for i in keys:
        sub_train_data[i] = [train_data[i][j] for j in ssub]
    return sub_train_data

# set the majority count target as the finnal result
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# generate decision tree with train data and target
def decision_tree_generate(train_data,features,target):
    
    # all the target list
    classList = train_data[target]
    # if all the class list is same
    if classList.count(classList[0])==len(classList): 
        return classList[0]
    # if there is no more feature except target
    if len(features) == 0:
        return majorityCnt(classList)
    
    # turn dict to pandas.core.frame.DataFrame
    dataset = pd.DataFrame.from_dict(train_data)
    # Find the min spilter and its entropy
    spliter, min_entropy = find_the_min_spilter(dataset, target)
    # set decision tree nodes
    decision_tree = {spliter[0]:{}}
    # detele features of min spilter
    features.remove(spliter[0])
    # all kind of feature value in spliter key
    uniquefv = set(train_data[spliter[0]])
    for value in uniquefv:
        # features of sub_train_data
        subfeatures = features[:]
        # generate sub train_data
        sub_train_data = sub_train_data_gen(train_data,spliter[0],value)
        # generate decision tree with recursion
        decision_tree[spliter[0]][value] = decision_tree_generate(sub_train_data, subfeatures, target)
    # features.remove(spliter[0])
    return decision_tree

# classfy new case with decision_tree
def decision_tree_classfier(new_case,decision_tree):
    la = list(decision_tree.keys())
    while len(la) > 0 and isinstance(decision_tree,dict):
        la = list(decision_tree.keys())
        decision_tree = decision_tree[la[0]][new_case[la[0]][0]]
    return decision_tree

# data for train
train_data = {
    'gender':['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}
target = 'bought'
# features is all keys without target
features = list(train_data.keys())
features.remove(target)

# generate decision tree and save decision tree result to dict
decision_tree = decision_tree_generate(train_data,features,target)
print(decision_tree)

# new case for testing decision tree
new_case = {
    'gender':['F'],
    'income': ['-10'],
    'family_number': [1]
}
boughtD = decision_tree_classfier(new_case,decision_tree)

# print the result
print(boughtD)