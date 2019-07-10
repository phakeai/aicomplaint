"""The module provides helper functions to save and load data.

"""

from tensorflow.python.keras.models import model_from_json
import json
import datetime
import pandas as pd
import numpy as np
import h5py
import os
import math


def create_folder(path):
    """Creates a folder.

    Args:
        path: path to the folder.

    Returns:
        path: path to the created folder.
    """
    try:
        if not os.path.isdir(path):
            os.mkdir(path)
    except OSError:
        print("Creation of the directory "+path+" failed")
    else:
        print("Successfully created the directory "+path)
    return path


def get_weight_paths(model_path):
    """Generates the file paths for the models of a single cross-validation.

    Args:
        model_path: path to the folder containing the models.

    Returns:
        a list of model paths.
    """
    weight_paths = []
    for file in os.listdir(model_path):
        if file.endswith('.h5'):
            weight_paths.append(os.path.join(model_path, file))
    return weight_paths


def create_ts_path(params, output_folder):
    """Creates a new folder. The folder's name contains the parameter combination of a model.
    The function also creates a subfolder named with a timestamp that contains.
    the evaluation data of single cross-validation.

    Args:
        params: dictionary containing the model parameters.
        output_folder: path to the result folder that will contain the results of all cross-validations.

    Returns:
        path: returns the created path.
    """
    dt = datetime.datetime.now()
    path = output_folder
    ts = '{:02d}{:02d}{:02d}{:02d}'.format(dt.day, dt.month, dt.hour, dt.minute)

    if params is not None:
        ps = str(params['layers']) + '_'+str(params['epochs']) + '_'+str(params['hidden_units']) +\
             '_' + str(params['pad_length']) + ('_r' if params['regularization'] > 0 else '') + '/'
        path += ps
        create_folder(path)
        path += ts
        create_folder(path)

    return path


def save_dictionary(path, word_to_id):
    """Saves a word to id mapping as a json file.

    Args:
        path: path to the folder that will contain the json file.
        word_to_id: dictionary containing the mapping of words to ids.

    Returns:

    """
    with open(path+'dictionary.json', 'w') as f:
        json.dump(word_to_id, f)


def load_dictionary(path):
    """Loads a dictionary mapping words to ids.

    Args:
        path: path of the json file containing the dictionary.

    Returns:
        word_to_id: dictionary mapping words to ids.
    """
    with open(path, 'r') as f:
        word_to_id = json.load(f)
    return word_to_id


def save_model(model, path, weight_path):
    """Saves a model.

    Args:
        model: a model object.
        path: path to the folder that will contain the saved model.
        weight_path: file name for the model of a specific training fold.

    Returns:

    """
    model_path = create_folder(path+'/model/')

    # path for the model specifications (save path)
    model_json_file = model_path + '/model.json'
    # path for the trained model weights (save path)
    model_weights = model_path + '/'+weight_path
    # save the model
    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_weights)
    print("model saved")


def load_data(dataset, validation_split):
    """

    Args:
        dataset: path to the hdf file
        validation_split: fraction of data that is used for validation

    Returns:
        folds: list of evaluation folds
    """
    folds = []
    fold_number = math.floor(1 / validation_split)
    for i in range(fold_number):
        train = pd.read_hdf(dataset, 'train_{}'.format(i))
        valid = pd.read_hdf(dataset, 'valid_{}'.format(i))
        folds.append((train, valid))
    return folds


def load_test_data(dataset):
    """Loads the test data from the hdf file.

    Args:
        dataset: path to the hdf file

    Returns:
        test: pandas dataframe containing the test data.
    """
    test = pd.read_hdf(dataset, 'test')
    return test


def load_params(path):
    """Loads the parameter configuration or parameter search space as defined in the params.json.

    Args:
        path: path to the params.json file.

    Returns:
        params: dictionary representing a single parameter configuration or a parameter search space.
    """
    with open(path, 'r') as input_file:
        params = json.load(input_file)
    return params


def save_params(params, path):
    """Saves the parameter configuration for a single cross-validation.

    Args:
        params: dictionary containing the parameters of a single cross-validation.
        path: path to the folder that will contain the saved params.json.

    Returns:

    """
    with open(path+'/params.json', 'w') as output_file:
        json.dump(params, output_file)


def write_final_results(path, histories):
    """Computes the mean values for a the histories of a single cross-validation.

    Args:
        path: path to the folder that will contain the saved mean values.
        histories: list of histories.

    Returns:

    """
    histories = np.array(histories)
    history_mean = np.mean(histories, axis=0)
    np.savetxt(path+'/history_means.csv', history_mean, delimiter=';')


def save_history(file, history):
    """Saves the history of a single training fold.

    Args:
        file: path to the file that will contain the training history of a single fold.
        history: history that will be saved.

    Returns:

    """
    loss = history['loss']
    acc = history['acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']

    fold_h = np.array([loss, acc, val_loss, val_acc])
    fold_h = fold_h.transpose()
    np.savetxt(file, fold_h, delimiter=';')
    return fold_h


def load_embedding(embedding_file):
    """Loads the embedding generated during preprocessign

    Args:
        embedding_file: path to the file containing the embedding hdf file.

    Returns:
        embedding_matrix: a numpy Array representing the embedding matrix
    """
    with h5py.File(embedding_file, 'r') as f:
        embedding_matrix = np.array(f.get('embedding'))

    return embedding_matrix


def load_model(json_path, weights_path):
    """Loads the model using a json description of the model and a hdf file containing the weights.

    Args:
        json_path: path to the model.json file.
        weights_path: path to the file containing the weights of an individual training fold.

    Returns:

    """
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json)
    json_file.close()
    model.load_weights(weights_path)
    return model


def load_models(model_path):
    """Loads the all models of a single cross-validation.

    Args:
        model_path: path to the folder containing the models of a cross-validation and the model
        description provided as model.json file.

    Returns:
        models: list of model objects
    """
    models = []
    weight_paths = get_weight_paths(model_path)
    for path in weight_paths:
        model = load_model(model_path+'/model.json', path)
        models.append(model)

    return models