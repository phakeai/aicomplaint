"""The module used trains and evaluates the models generated within a predefined parameter search space.

Example:
    Example command line to preprocess complaint management data:

    $ python train.py -I "../data/data.h5" -P "params.json" -E "../data/embedding.h5"
"""
import save_load
import model
import preprocessing
import argparse


def train_model(x_train, y_train, x_valid, y_valid, m, batch_size, epochs):
    """Trains and validates a single model based on the provided data and parameters.

    Args:
        x_train: training samples that are fed into the network
        y_train: expected output for the training samples
        x_valid: validation samples that are fed into the network
        y_valid: expected output for the validation samples
        m: model used for training and validation
        batch_size: batch size used for training the model
        epochs: number of epochs used for training the model

    Returns:
        (tuple): a tuple containing:

            history: training/validation accuracy and loss across the epochs
            m: trained model
    """
    # start training the model
    history = m.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_valid, y_valid), epochs=epochs)
    return history, m


def train_eval(params, embedding_matrix, folds, output_path):
    """Generates, trains and evaluates a model for each fold.

    Args:
        params: dictionary containing training and evaluation parameters
        embedding_matrix: embedding matrix that is used in the model
        folds: data folds that are used for training and validation
        output_path: path to the folder that will contain the training and evaluation results

    Returns:

    """
    path = save_load.create_ts_path(params, output_path)

    histories = []

    for i, fold in enumerate(folds):
        x_train, y_train = preprocessing.generateXY(fold[0], params['pad_length'])
        x_valid, y_valid = preprocessing.generateXY(fold[1], params['pad_length'])

        m = model.build_lstm_model(y_train.shape, params, embedding_matrix)
        history, trained_model = train_model(x_train, y_train, x_valid, y_valid, m, params['batch_size'], params['epochs'])

        save_load.save_model(trained_model, path, '/model_fold{}.h5'.format(i))
        save_load.save_params(params, path)

        h = save_load.save_history(path+'/history_fold{}.csv'.format(i), history.history)
        histories.append(h)

    save_load.write_final_results(path, histories)


def override_params(perm, params):
    """Overrides the epochs, pad_length, hidden_units, regularization and layers value in the params dictionary.

    Args:
        perm: a tuple containing a single permutation of the parameters
            epochs, pad_length, hidden_units, regularization and layers
        params:

    Returns:
        params: returns the modified parameter dictionary
    """
    e, p, h, r, la = perm
    params['epochs'] = e
    params['pad_length'] = p
    params['hidden_units'] = h
    params['regularization'] = r
    params['layers'] = la

    return params


def hyperparam_train_eval(params, embedding_path, data_path, output_path):
    """

    Args:
        params: dictionary containing the parameter search space
        embedding_path: path to the reduced embedding matrix
        data_path: path to the preprocessed data
        output_path: path to a folder that will contain the evaluation results

    Returns:

    """
    embedding_matrix = save_load.load_embedding(embedding_path)

    folds = save_load.load_data(data_path, params['validation_split'])

    permutations = [(e, p, h, r, l) for e in params['epochs']
                    for p in params['pad_length']
                    for h in params['hidden_units']
                    for r in params['regularization']
                    for l in params['layers']]

    for i, perm in enumerate(permutations):
        ps = override_params(perm, params)
        print('='*120)
        print('permutation {} of {}'.format(i+1, len(permutations)))
        print(ps)
        print('='*120)

        train_eval(ps, embedding_matrix, folds, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', help='path to the preprocessed data hdf', required=True)
    parser.add_argument('-P', '--params', help='path to the json parameter file', default='params.json')
    parser.add_argument('-E', '--embedding', help='path to the hdf embedding file create during preprocessing'
                        , required=True)
    parser.add_argument('-O', '--output', help='folder to save model configurations, model weights as well as training'
                                               ' and validation results', default='../results/')
    args = parser.parse_args()

    training_data_path = args.input
    params_path = args.params
    embedding_path = args.embedding
    output_path = args.output
    params = save_load.load_params(params_path)

    save_load.create_folder(output_path)

    hyperparam_train_eval(params, embedding_path, training_data_path, output_path)
