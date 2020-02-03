from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def build_lstm_model(output_shape, params, embedding_matrix):
    # load embedding matrix
    regularizer = None if params['regularization'] == 0 else regularizers.l1(params['regularization'])

    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=params['pad_length'],
                        weights=[embedding_matrix], trainable=True, name='embedding'))

    i = 0
    for i in range(1, params['layers']):
        model.add(Dropout(0.1))
        model.add(LSTM(params['hidden_units'], activity_regularizer=regularizer, return_sequences=True, name=f'lstm_{i}'))
    model.add(Dropout(0.1))
    model.add(LSTM(params['hidden_units'], activity_regularizer=regularizer, name=f'lstm_{i+1}'))

    model.add(Dense(output_shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model