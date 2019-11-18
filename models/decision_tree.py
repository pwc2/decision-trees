"""
    File name: decision_tree.py
    Author: Patrick Cummings
    Date created: 11/16/2019
    Date last modified: 11/16/2019
    Python Version: 3.7

"""

import pandas as pd
train = pd.read_csv('data/pa3_train.csv')

# find benefit for each split and store in dict, except labels
split_b = {k:0 for k in train.columns.to_list()[:-1]}

for feature in list(split_b.keys()):
    # left true, right false
    vals = train[feature].to_list()


