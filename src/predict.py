"""The module provides functions for evaluating the average performance of a model across all folds.

Example:
    Example command line to evaluate a model across all folds:
    $ python predict.py -I "../data/data.h5" -P "../results/1_100_16_100/10070824//params.json" -M "../results/1_100_16_100/22050824/model/"
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import metrics
import numpy as np
import argparse
import save_load


def eval(trained_models, test, pad_length):
    """Evaluates a model across the folds.

    Args:
        trained_models: trained models
        test: dataframe containing the test dataset
        pad_length: padding length used for preprocessing the input data

    Returns:

    """
    sequences = test.text.to_list()
    print(sequences)
    # pad training sequences to length of pad_length
    x_test = pad_sequences(sequences, maxlen=pad_length)
    y_test = test.code.to_numpy()

    accs = []
    cms = []

    for model in trained_models:
        y_hat_test = np.argmax(model.predict(x_test), axis=1)
        accuracy = metrics.accuracy_score(y_test, y_hat_test)
        confusion_matrix = metrics.confusion_matrix(y_test, y_hat_test)
        accs.append(accuracy)
        cms.append(confusion_matrix)

    mean_acc = np.mean(np.array(accs))
    std_acc = np.std(np.array(accs))
    mean_cm = np.mean(np.array(cms), axis=0)
    std_cm = np.std(np.array(cms), axis=0)

    print(mean_acc)
    print(std_acc)
    print(mean_cm)
    print(std_cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', help='path to the preprocessed data hdf file', required=True)
    parser.add_argument('-M', '--model', help='path to the model folder containing the model.json and the weights of'
                                              ' the individual folds', required=True)
    parser.add_argument('-P', '--params', help='path to the json parameter file used for'
                                               ' training the models', required=True)
    args = parser.parse_args()


    params = save_load.load_params(args.params)
    models = save_load.load_models(args.model, params['validation_splits'])
    test_data = save_load.load_test_data(args.input)

    eval(models, test_data, params['pad_length'])
