import numpy as np
from numpy import array
import pandas as pd

import os
import glob

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# enumerate adds a number like [0, [22, 35, 42], ... ] to each sample
# creates data and then looks for row with max length, then adds 0 triplet until each row is the same length as max
# length
def create_all_data():
    data = []
    temp = []
    max_len = 0
    label_names = [f for f in os.listdir("HMP_Dataset")]
    for label in label_names:
        file_list = glob.glob(os.path.join(os.getcwd(), "HMP_Dataset/" + label, "*.txt"))
        for file in file_list:
            with open(file) as f:
                for line in f:
                    line = line.split()
                    line = [int(i) for i in line]
                    temp.append(line)
                data.append(temp)
                temp = []
    for row in data:
        if len(row) > max_len:
            max_len = len(row)
    for index, row in enumerate(data):
        while len(row) != max_len:
            data[index].append([0, 0, 0])
    return data


def create_labels():
    labels = []
    label_names = [f for f in os.listdir("HMP_Dataset")]
    for label in label_names:
        file_list = glob.glob(os.path.join(os.getcwd(), "HMP_Dataset/" + label, "*.txt"))
        for num in range(len(file_list)):
            labels.append(label)
    return labels


def transform_labels(labels):
    t = []
    label_names = sorted(list(set(labels)))
    for label in labels:
        t.append(label_names.index(label))
    return t


# data is a list of labels, turns data into array called values
# LabelEncoder turns the 'string' labels into labels between 0 and n where n is number of labels
# fit_transform actually takes in array of strings and turns them into numbers
# after this, it reshapes the array so that there is now a row for each label
# OneHotEncoder and fit_transform then turns the number that represents the label in each row into a one hot encoding
def create_onehot_labels(labels):
    data = labels
    values = array(data)
    le = LabelEncoder()
    num_labels = le.fit_transform(values)
    num_labels = num_labels.reshape(len(num_labels), 1)
    enc = OneHotEncoder(sparse=False, categories='auto')
    onehot_labels = enc.fit_transform(num_labels)
    return onehot_labels


def stratify(features, labels):
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, stratify=labels)
    return x_train, x_test, y_train, y_test


one_d = 850
two_d = 9167
three_d = 3
np_labels = create_onehot_labels(create_labels())
pd_data = pd.DataFrame(create_all_data()).values
np_data = np.zeros((one_d, two_d, three_d))
for i in range(one_d):
    for j in range(two_d):
        for k in range(three_d):
            np_data[i, j, k] = pd_data[i, j][k]
np_data = np.reshape(np_data, (one_d, (two_d*three_d)))


x_train, x_test, y_train, y_test = train_test_split(np_data, np_labels, test_size=0.2, shuffle=True, stratify=np_labels)

sess = tf.Session()

model = Sequential()
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=14, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)

test_loss_acc = model.evaluate(x_test, y_test)
print(test_loss_acc)

sess.close()

'''
pd_x_test = pd.DataFrame(x_test)
pd_x_test.to_csv('x_test.csv', index=False)
pd_y_train = pd.DataFrame(y_train)
pd_y_train.to_csv('y_train.csv', index=False)
pd_y_test = pd.DataFrame(y_test)
pd_y_test.to_csv('y_test.csv', index=False)
'''

'''
np.savetxt("x_train.csv", x_train, delimiter=",", fmt='%0.0f')
np.savetxt("x_test.csv", x_test, delimiter=",", fmt='%0.0f')
np.savetxt("y_train.csv", y_train, delimiter=",", fmt='%0.0f')
np.savetxt("y_test.csv", y_test, delimiter=",", fmt='%0.0f')
'''
