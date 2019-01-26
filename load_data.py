import os



def load_data():
    label_set = [f for f in os.listdir("Data")
              if not os.path.isfile(os.path.join("Data", f))]

    examples = []
    labels = []
    for label in label_set:
        path = os.path.join("Data", label)
        filenames = os.listdir(path)

        for filename in filenames:
            file = open(os.path.join(path, filename))
            file_lines = file.readlines()
            samples = [[int(r) for r in line.split(' ')] for line in file_lines]


            examples.append(samples)
            labels.append(label)

    return examples, labels
