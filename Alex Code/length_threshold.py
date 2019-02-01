from load_data import load_data
from math import ceil

""" 
Returns examples with length (number of readings) less than a constant
Such that threshold% of the examples are returned. Returns the constant length and the examples
"""
def get_examples_below_length_threshold(examples, labels, threshold=0.9):
    examples_labels = zip(examples, labels)
    examples_labels = sorted(examples_labels, key=lambda e_l: len(e_l[0]))

    examples, labels = zip(*examples_labels)
    truncated_length = ceil(len(examples) * threshold)

    examples = examples[:truncated_length]
    labels = labels[:truncated_length]
    length_constant = len(examples[-1])

    # In order to do splits we must now remove classes that only have one example
    trimmed_examples = []
    trimmed_labels = []
    for label in list(set(labels)):
        rel_examples = [example for i, example in enumerate(examples) if labels[i] == label]
        rel_labels = [label] * len(rel_examples)
        if len(rel_examples) > 1:
            trimmed_examples.extend(rel_examples)
            trimmed_labels.extend(rel_labels)

    examples = trimmed_examples
    labels = trimmed_labels

    return length_constant, examples, labels
