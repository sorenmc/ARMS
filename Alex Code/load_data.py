import os
import numpy as np



def load_data():
    directories = [f for f in os.listdir("Data")
              if not os.path.isfile(os.path.join("Data", f))]

    examples = []
    labels = []
    for label in directories:
        path = os.path.join("Data", label)
        filenames = os.listdir(path)

        for filename in filenames:
            file = open(os.path.join(path, filename))
            file_lines = file.readlines()
            samples = [[int(r) for r in line.split(' ')] for line in file_lines]


            examples.append(samples)
            labels.append(label)

    # Group _MODEl examples and not model examples into same class
    return examples, [label.replace('_MODEL', '') for label in labels]

def pad_examples(examples):
    max_len = max([len(example) for example in examples])

    features_set = []
    masks = []

    for example in examples:

        # [0,0,0] if beyond sequence length
        features = np.stack([np.array(example[i]) if i < len(example) else np.zeros((3)) for i in range(max_len)])
        features_set.append(features)

        # Sequence of 0's and 1's, 1 if the example contains data at position i
        masks.append(np.array([1 if i < len(example) else 0 for i in range(max_len)]))

    features_set = np.stack(features_set)
    masks = np.stack(masks)

    return features_set, masks

# Converts the string labels into a one-hot representation
def convert_labels(labels, label_set):
    one_hot_labels = []
    num_classes = len(label_set)
    for label in labels:
        one_hot = np.array([1 if i == label_set.index(label) else 0 for i in range(num_classes)])
        one_hot_labels.append(one_hot)

    return np.stack(one_hot_labels)


"""Directly from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



