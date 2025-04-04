import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Bidirectional, Input
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.metrics import Precision, Recall
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import set_random_seed
from sklearn.preprocessing import StandardScaler

from data_preprocessing.data_preprocessing import scale_data, preprocess_patient_data_for_ML_classifier


def preprocess_for_lstm(X_train, X_valid, X_test, sequence_length=49):
    """
    Preprocesses the datasets to prepare them for LSTM input.
    Each patient's data is reshaped into a sequence of 49 time steps.

    Parameters:
    X_train (pd.DataFrame): Training dataset with patient records.
    X_valid (pd.DataFrame): Validation dataset with patient records.
    X_test (pd.DataFrame): Test dataset with patient records.
    sequence_length (int): The number of time steps (default is 49).

    Returns:
    tuple: Preprocessed datasets (X_train_lstm, X_valid_lstm, X_test_lstm)
    """

    def reshape_to_sequences(X):
        # Group by 'recordid' and reshape each patient's data into a sequence
        sequences = X.groupby('recordid').apply(lambda group: group.drop(columns='recordid').values[:sequence_length])

        # Convert to numpy array
        return np.stack(sequences)

    X_train, X_valid, X_test = preprocess_patient_data_for_ML_classifier(X_train, X_valid,
                                                                         X_test)  # Ensure the data is prepared for LSTM

    scaler = StandardScaler()
    X_train, X_valid, X_test = scale_data(X_train, X_valid, X_test, scaler)

    X_train_lstm = reshape_to_sequences(X_train)
    X_valid_lstm = reshape_to_sequences(X_valid)
    X_test_lstm = reshape_to_sequences(X_test)

    return X_train_lstm, X_valid_lstm, X_test_lstm

def build_lstm_model(input_shape):
    """
    Build an LSTM model for binary classification.

    Parameters:
    input_shape (tuple): Shape of the input data.

    Returns:
    model: Compiled LSTM model.
    """
    inputs = Input(shape=input_shape)  # Explicit input layer
    x = LSTM(64, return_sequences=True, kernel_regularizer='l2', recurrent_dropout=0.2, input_shape=input_shape)(inputs)
    x = LSTM(32, return_sequences=True, kernel_regularizer='l2', recurrent_dropout=0.2)(x)

    # Use the last output of the sequence
    x = LSTM(32, return_sequences=False, kernel_regularizer='l2', recurrent_dropout=0.2)(x)

    # Dense layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)  # Sigmoid activation for binary classification

    # Compile the model with binary cross-entropy loss and Adam optimizer
    optimizer = Adam(learning_rate=0.0005)
    model = Model(inputs=inputs, outputs=outputs)  # Functional API
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', Precision(), Recall()]
    )
    return model

def build_bidirectional_lstm_model(input_shape):
    """
    Build a bidirectional LSTM model for binary classification.

    Parameters:
    input_shape (tuple): Shape of the input data.

    Returns:
    model: Compiled bidirectional LSTM model.
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer='l2', recurrent_dropout=0.2, input_shape=input_shape)))
    model.add(Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer='l2', recurrent_dropout=0.2)))

    # Use the last output of the sequence
    model.add(Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer='l2', recurrent_dropout=0.2)))

    # Dense layer for binary classification
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

    # Compile the model with binary cross-entropy loss and Adam optimizer
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', Precision(), Recall()]
    )
    return model

def get_patient_embeddings(model, X):
    """
    Extracts a single vector representation for each patient.
    """
    lstm_layer = keras.Model(inputs=model.input, outputs=model.layers[-2].output)  # Get last LSTM layer output
    embeddings = lstm_layer.predict(X)
    return embeddings
