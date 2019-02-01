from load_data import load_data


examples, labels = load_data()

label_set = list(set(labels))

for label in label_set:
    label_examples = [example for i, example in enumerate(examples) if labels[i] == label]
    print('{} : {}, {} - {}'.format(
        label,
        len(label_examples),
        min([len(label_example) for label_example in label_examples]),
        max([len(label_example) for label_example in label_examples])
    ))