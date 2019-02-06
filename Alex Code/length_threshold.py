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

    trunc_examples = []
    for i in range(len(examples)):
        if i < truncated_length:
            trunc_examples.append(examples[i])
            length_constant = len(examples[i])
        else:
            trunc_examples.append(examples[i][:length_constant])

    return length_constant, trunc_examples, labels
