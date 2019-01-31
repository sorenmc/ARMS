from load_data import load_data, convert_labels
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# percent of data to use for training & and validation
train_val_split = 0.8
# percent of training/validation data to use for validation
val_split = 0.1
indices_directory = "Data"

data = load_data()
_, labels = load_data()
labels = convert_labels(labels, list(set(labels)))
sss = StratifiedShuffleSplit(1, train_size=train_val_split)

#split off test set set
for train_and_val_index, test_index in sss.split(X=np.zeros(len(labels)), y=labels):
    f = open('{}/test_indices.csv'.format(indices_directory), 'w')
    f.write(','.join([str(int(t)) for t in test_index]))
    f.close()

#split remaining examples into test & val
sss = StratifiedShuffleSplit(1, train_size=1 - val_split)
for train_index, val_index in sss.split(
        X=np.zeros(len(train_and_val_index)),
        y=[labels[ix] for ix in train_and_val_index]):

    f = open('{}/train_indices.csv'.format(indices_directory), 'w')
    f.write(','.join([str(int(t)) for t in train_index]))
    f.close()

    f = open('{}/val_indices.csv'.format(indices_directory), 'w')
    f.write(','.join([str(int(t)) for t in val_index]))
    f.close()
