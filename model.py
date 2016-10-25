#!/usr/bin/python

import h5py
import csv
import numpy as np
from six.moves import cPickle as pickle
import pandas as pd


# read in data
dataset = '/media/dat1/liao/dataset/human_robot/'
data_filename = dataset + 'data.hdf5'
test_rank_filename = dataset + 'part_data/test_col_0'

f = h5py.File(data_filename)
train_data_label = f['train_data_label'][:]
test_data = f['test_data'][:]
f.close()

with open(test_rank_filename, 'rb') as f:
  test_rank = pickle.load(f)

test_data = test_data[:, [0,1,2,8,9,15,16,17,18,19,21,22,23]]
train_data_label = train_data_label[:, [0,1,2,8,9,15,16,17,18,19,21,22,23, -1]]

print ('train: ', train_data_label.shape)
print ('test: ', test_data.shape)

from sklearn import neighbors, svm, tree, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

clf = LogisticRegression().fit(train_data_label[:,:-1], train_data_label[:,-1])

#clf = GradientBoostingClassifier(n_estimators=100, #learning_rate=1.0,
#        verbose = 1, random_state=1).fit(train_data_label[:,:-1], train_data_label[:,-1])
result = clf.predict(test_data)

print (result)

with open('lr_human_robot.csv', 'w') as f:
  for i in range(len(result)):
    if result[i] == 1: f.write(str(test_rank[i]) + '\n')    

rr = clf.predict_proba(test_data)
with open('lr_hr_proba.csv', 'w') as f:
  for i in range(len(test_rank)):
    f.write(str(test_rank[i]) + ',' + str(rr[i][1]) + '\n') 
