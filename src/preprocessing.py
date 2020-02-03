"""Preprocessing of complaint management data.

This module preprocesses complaint management data for a classification training.
It requires complaint management data and a word embedding.

Example:
    Example command line to preprocess complaint management data:

    $ python preprocessing.py -I "../text_and_codes.csv" -P "params.json" -E "../wiki-news-300d-1M.vec"

    The input csv should contain two columns separated by a tabstop. The first column contains the complaint text while
    the second contains an error or complaint code. The word embedding is required to use the embedding format of
    the fasttext .vec file (https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip).
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import argparse
import save_load
import pandas as pd
import numpy as np
import collections
import h5py
import random
from sklearn.model_selection import StratifiedKFold, train_test_split


def generate_filters():
    """Function that generates filters that will be applied to the texts for data cleansing.

    Returns:
        (tuple): a tuple containing:

            repl_filter: a dictionary containing strings that are mapped to a string
            rm_filter: a dictionary containing strings that are mapped to an empty string

    """
    # repl_filter contains a mapping of strings and the replacement strings
    repl_filter = {'<br>': ' ', '<br/>': ' ', '</br>': ' ', '<p>': '', '</p>': '', '<p/>': '', '<span/>': '',
                   '</span>': '', '<span>': ''}

    # rm_filter contains strings that will be removed from the textual description
    rm = ['!', '"', '#', '$' '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
          '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    # generate ciphers 0-9
    ciphers = [str(cipher) for cipher in list(range(0, 10))]
    # append ciphers to the filter
    rm += ciphers

    rm_filter = dict()
    for f in rm:
        rm_filter[f] = ''

    return repl_filter, rm_filter


def generateXY(df, pad_length):
    """Splits samples into input (x) and expected output (y) and applies a padding to the input.

    Args:
        df: dataframe containing the evaluation data
        pad_length: the padding length

    Returns:
        (tuple): a tuple containing:
            x: the data that is fed into the network during training and validation
            y: the expected outputs for the inputs x
    """
    # create training sequences
    sequences = df.text.to_list()
    # pad training sequences to length of pad_length
    x = pad_sequences(sequences, maxlen=pad_length)
    # create one hot encoding for each sequence
    y = df.code.to_numpy().reshape(len(sequences), 1)
    y = to_categorical(y)
    return x, y


def apply_filter_column(filt, text_column):
    """Applies filters to a pandas series containing text fields.

    Args:
        filt: dictionary mapping strings to replacement strings.
        text_column: pandas series containing complaint texts

    Returns:
        Returns the modified pandas series.
    """
    for f in filt.keys():
        text_column = text_column.str.replace(f, filt[f])

    return text_column


def filter_html_tags(df):
    """Applies filters to the column named 'text' in a pandas DataFrame.

    Args:
        df: pandas DataFrame containing the complaint texts in a column named 'text'

    Returns:
        Returns the modified pandas DataFrame.
    """
    # Generate filters
    repl, rm = generate_filters()
    # Apply filters
    df.text = apply_filter_column(repl, df.text)
    df.text = apply_filter_column(rm, df.text)
    # transform to lower case characters
    df.text = df.text.str.lower()
    return df


def build_dict(filename, words):
    """Builds a dictionary mapping words to id and reduces the provides embedding matrix to the words contained in
    the provided dataset.

    Args:
        filename: path to the a word embedding
        words: words contained in the dataset

    Returns:
        (tuple): a tuple containing:

            word_to_id: dictionary mapping words to ids
            embedding_matrix: reduced embedding matrix
    """
    # use defaultdict to return 0 for unknown words

    skip_vec = words is None

    # default dict mapping a word to an id
    word_to_id = collections.defaultdict(int)
    embedding = []

    # append zero vector to embedding for unknown words at position 0
    embedding.append(np.zeros((1, 300), dtype=float))
    with open(filename, 'r') as file:
        i = 0
        for line in file.readlines():
            # skip first line, since it only contains meta information
            if i > 0:
                split = line.split(' ')
                word = split[0]
                vec = split[1:]
                if not skip_vec and word in words:
                    word_to_id[word] = i
                    vec = np.array([float(val) for val in vec]).reshape(1, len(vec))
                    embedding.append(vec)
                    i = i + 1
            else:
                i = i + 1
    embedding_matrix = np.array(embedding).squeeze()
    return word_to_id, embedding_matrix


def down_sample(df, samples_per_class):
    """Uniformly samples training, validation and test data from the classes provided in the dataframe.

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame: a pandas DataFrame containing the random samples
    """
    classes = df.groupby(['code'])
    data = []

    for name, code_class in classes:
        items_per_class = len(code_class)
        random_uniform = random.sample(list(range(items_per_class)), min(samples_per_class, items_per_class))
        sam = code_class.iloc[random_uniform].reset_index(drop=True)
        data.append(sam)

    return pd.concat(data, ignore_index=True)


def prepare_data(train_data_path, embedding_path, test_split, validation_splits, output_folder,
                 class_threshold, samples_per_class):
    """Prepares data for training, validation and evaluation. Creates a reduced embedding matrix based on the words
    contained in the provided dataset.

    Args:
        data_path: path to the csv containing complaint messages and error/complaint codes
        embedding_path: path to a word embedding, e.g. the fasttext embedding
            https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
        test_split: fraction of data that is used for testing
        validation_split: fraction of data that is used for validation
        output_folder: path to the folder used for saving the preprocessed data and the embedding matrix
        class_threshold: threshold of samples of an error code to be considered a class during training
        samples_per_class: samples that are drawn from each class for training, validation and testing

    Returns:

    """
    # Load file
    df = pd.read_csv(train_data_path, sep='\t')
    df = filter_html_tags(df)

    # split whitespaces
    df.text = df.text.str.split()

    # create a dictionary mapping codes to frequency
    code_frequency = df.code.value_counts().to_dict()

    # set code to 0 if it the frequency is below the threshold
    df.code = df.code.apply(lambda x: x if code_frequency[x] > class_threshold else 0)

    # compute the frequency of each error code to determine new class identifier of type integer
    code_frequency = df.code.value_counts().to_dict()

    # maps old codes to new codes
    new_codes = dict()
    for i, key in enumerate(code_frequency.keys()):
        new_codes[key] = i
    df.code = df.code.map(new_codes)

    # save the code mapping in a json file
    save_load.save_dictionary(output_folder, new_codes, 'codes_dict.json')

    # generate set of words that are contained in the dataset
    word_list = [word for sublist in df.text.to_list() for word in sublist]
    words = set(word_list)

    # load the embedding matrix and reduce it to the words contained in the dataset
    word_to_id, embedding_matrix = build_dict(embedding_path, words)

    # save the word-to-id dictionary
    save_load.save_dictionary(f'{output_folder}/', word_to_id, 'dictionary.json')

    if samples_per_class is not None:
        df = down_sample(df, samples_per_class)

    # convert words into integer values according to the word_to_id dictionary
    df.text = df.text.apply(lambda wl: [int(word_to_id[word]) for word in wl])

    # divide the dataset in to training and test preserving the original code distribution
    train, test = train_test_split(df, test_size=test_split, shuffle=True, stratify=df.code)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # save the test data
    test.to_hdf(f'{output_folder}/data.h5', key='test')

    # generate 10 stratified folds for model training
    skf = StratifiedKFold(n_splits=validation_splits, shuffle=True)

    for i, (train_index, val_index) in enumerate(skf.split(train.text, train.code)):

        train_i = train.loc[train_index]
        val_i = train.loc[val_index]

        # save fold training and validation data of fold i
        train_i.to_hdf(f'{output_folder}/data.h5', key=f'train_{i}')
        val_i.to_hdf(f'{output_folder}/data.h5', key=f'valid_{i}')

    # save the reduced embedding matrix
    with h5py.File(f'{output_folder}/embedding.h5', 'w') as f:
        f.create_dataset('embedding', data=embedding_matrix)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', help='path to the input csv containing an error code column and a respective'
                                              ' complaint text column', required=True)
    parser.add_argument('-P', '--params', help='path to the json parameter file', default='params.json')
    parser.add_argument('-E', '--embedding', help='path to the fasttext embedding file', required=True)
    parser.add_argument('-O', '--output', help='folder to save training, validation and test data, the dictionary and'
                                               ' the respective embedding matrix', default='../data/')
    parser.add_argument('-T', '--threshold', help='the threshold specifies the number of samples an error code exhibits'
                                                  ' to be considered a separate class during training', type=int,
                        default=500)
    parser.add_argument('-S', '--samples', help='number of samples that are drawn from a class; if None the entire'
                                                'samples are used', type=int, default=None)

    args = parser.parse_args()

    data_path = args.input
    embedding_path = args.embedding
    params_path = args.params
    output_path = args.output
    class_threshold = args.threshold
    samples_per_class = args.samples

    save_load.create_folder(output_path)
    params = save_load.load_params(params_path)

    prepare_data(data_path, embedding_path, params['test_split'], params['validation_splits'], output_path,
                 class_threshold, samples_per_class)



