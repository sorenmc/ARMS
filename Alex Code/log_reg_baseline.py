from load_data import load_data_subset, pad_examples, convert_labels
from length_threshold import get_examples_below_length_threshold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

num_folds = 10


def train():
    examples, labels = load_data_subset('Data/train_indices.csv')
    _, examples, labels = get_examples_below_length_threshold(examples, labels, threshold=0.90)

    X, _ = pad_examples(examples)
    # Have to 'flatten' array to 2D to work with logsitic regression
    X = np.reshape(X, (X.shape[0], X.shape[1] * 3))
    y = convert_labels(labels, list(set(labels)))

    '''Split of fine-tuning set'''
    sss = StratifiedShuffleSplit(1, train_size=0.8)
    for train_index, val_index in sss.split(X=X, y=y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]

    crossval_models = []
    crossval_accuracies = []
    '''Cross validation'''
    for fold in range(num_folds):

        '''Split off fold'''
        sss = StratifiedShuffleSplit(1, train_size=0.8)
        for trainfold_index, valfold_index in sss.split(X=X_train, y=y_train):
            X_trainfold = X[trainfold_index]
            y_trainfold = y[trainfold_index]
            X_valfold = X[valfold_index]
            y_valfold = y[valfold_index]

        lr = LogisticRegression(
            class_weight='balanced',
            multi_class='multinomial',
            solver='newton-cg',
            max_iter=200,
            tol=1e-4
        )
        lr.fit(X_trainfold, y_trainfold)

        predictions = lr.predict(X_valfold)
        accuracy = accuracy_score(y_true = y_valfold, y_pred=predictions)

        crossval_accuracies.append(accuracy)
        crossval_accuracies.append(lr)

        print('Fold {} accuracy: {}'.format(fold, accuracy))

    best_ID = crossval_accuracies.index(max(crossval_accuracies))
    best_model = crossval_models[best_ID]

    predictions = best_model.predict(X_val)
    accuracy = accuracy(y_true=y_val, y_pred=predictions)

    print('Validation accuracy: {}'.format(accuracy))



train()



