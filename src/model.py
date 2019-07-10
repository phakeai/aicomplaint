from tensorflow.python.keras.layers import Dense, CuDNNLSTM, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras import regularizers


def build_lstm_model(output_shape, params, embedding_matrix):

    regularizer = None if params['regularization'] == 0 else regularizers.l1(params['regularization'])

    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=params['pad_length'],
                        weights=[embedding_matrix], trainable=True, name="embedding",  activity_regularizer=regularizer))

    for i in range(1, params['layers']):
        model.add(Dropout(0.1))
        model.add(CuDNNLSTM(params['hidden_units'], activity_regularizer=regularizer, return_sequences=True, name='lstm_{}'.format(i)))

    model.add(Dropout(0.1))
    model.add(CuDNNLSTM(params['hidden_units'], activity_regularizer=regularizer))

    model.add(Dense(output_shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
