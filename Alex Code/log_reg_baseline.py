from load_data import load_data_subset, pad_examples, convert_labels
from length_threshold import get_examples_below_length_threshold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from statistics import mean

#Whether or not to run on the test set.
evaluate = True
num_folds = 10
truncate_threshold = 1
# Inverse regularization penalty
C = 0.5

def train(return_model=False):
    examples, labels = load_data_subset('Data/train_indices.csv')

    print(min([len(ex) for ex in examples]))
    if truncate_threshold < 1:
        _, examples, labels = get_examples_below_length_threshold(examples, labels, threshold=truncate_threshold)

    print('Training set size: {}'.format(len(examples)))
    print('Num classes: {}'.format(len(sorted(list(set(labels))))))

    X, _ = pad_examples(examples)
    # Have to 'flatten' array to 2D to work with logsitic regression
    X = np.reshape(X, (X.shape[0], X.shape[1] * 3))
    train_label_set = sorted(list(set(labels)))
    ls = convert_labels(labels, train_label_set)
    y = np.argmax(ls, axis=1)

    print(X.shape)

    '''Split of fine-tuning set'''
    sss = StratifiedShuffleSplit(1, train_size=0.8)
    for train_index, val_index in sss.split(X=X, y=y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

    crossval_models = []
    crossval_accuracies = []
    crossval_training_accuracies = []
    '''Cross validation'''
    for fold in range(num_folds):

        '''Split off fold'''
        sss = StratifiedShuffleSplit(1, train_size=0.8)
        for trainfold_index, valfold_index in sss.split(X=X_train, y=y_train):
            X_trainfold = X_train[trainfold_index]
            y_trainfold = y_train[trainfold_index]
            X_valfold = X_train[valfold_index]
            y_valfold = y_train[valfold_index]

        lr = LogisticRegression(
            class_weight='balanced',
            multi_class='multinomial',
            solver='newton-cg',
            max_iter=200,
            tol=1e-4,
            C=C
        )
        lr.fit(X_trainfold, y_trainfold)

        predictions = lr.predict(X_valfold)
        accuracy = accuracy_score(y_true = y_valfold, y_pred=predictions)

        crossval_accuracies.append(accuracy)
        crossval_training_accuracies.append(accuracy_score(y_true=y_trainfold, y_pred=lr.predict(X_trainfold)))
        crossval_models.append(lr)

        print('Fold {} accuracy: {}'.format(fold, accuracy))

    best_ID = crossval_accuracies.index(max(crossval_accuracies))
    best_model = crossval_models[best_ID]

    predictions = best_model.predict(X_val)
    accuracy = accuracy_score(y_true=y_val, y_pred=predictions)
    print('Best model training_accuracy: {}'.format(crossval_training_accuracies[best_ID]))
    print("Mean cross-val accuracy: {}".format(mean(crossval_accuracies)))
    print('Validation accuracy: {}'.format(accuracy))

    if return_model:
        return best_model

if not evaluate:
    train()
else:
    model = train(return_model=True)

    #Get max length in training set
    examples, labels = load_data_subset('Data/train_indices.csv')
    if truncate_threshold > 0:
        maxlen, _, _ = get_examples_below_length_threshold(examples, labels, threshold=truncate_threshold)

    #Load test data & truncate
    test_examples, test_labels = load_data_subset('Data/test_indices.csv')
    test_examples, _ = pad_examples(test_examples)
    if test_examples.shape[1] > maxlen:
        test_examples = test_examples[:, :maxlen, :]



    test_examples = np.reshape(test_examples, (test_examples.shape[0], test_examples.shape[1] * 3))
    test_label_set = sorted(list(set(test_labels)))
    print(test_label_set)
    test_labels = convert_labels(test_labels, test_label_set)
    test_labels = np.argmax(test_labels, axis=1)

    test_predictions = model.predict(test_examples)

    test_accuracy = accuracy_score(y_true=test_labels, y_pred=test_predictions)

    print('Test accuracy: {}'.format(test_accuracy))

