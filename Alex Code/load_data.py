import os
import numpy as np


path_to_data = "Data"

# Use for loading data from the trainin/test indices files.
def load_data_subset(indices_path):
    examples, labels = load_data()
    with open(indices_path, 'r') as f:
        indices = [int(ix) for ix in f.readline().split(',')]
    return [examples[ix] for ix in indices], [labels[ix] for ix in indices]

def load_data():
    directories = [f for f in os.listdir(path_to_data)
              if not os.path.isfile(os.path.join(path_to_data, f))]
    directories.sort()

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




