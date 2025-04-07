import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Bidirectional, Input
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.metrics import Precision, Recall
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import set_random_seed
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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

def build_contrastive_lstm(input_shape, lstm_units=64, projection_dim=32, temperature=0.1):
    """
    Build a self-supervised LSTM model using InfoNCE loss (Temporal Contrastive Learning).

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - lstm_units (int): Number of units in LSTM layers.
    - projection_dim (int): Dimension of the projection head's output.
    - temperature (float): Temperature parameter for InfoNCE loss.

    Returns:
    - model: Compiled contrastive LSTM model.
    - encoder: LSTM encoder (for feature extraction).
    """
    # --- Encoder (LSTM) ---
    inputs = Input(shape=input_shape)
    x = LSTM(
        lstm_units,
        return_sequences=True,
        kernel_regularizer=l2(0.01),
        recurrent_dropout=0.2
    )(inputs)
    x = LSTM(
        lstm_units // 2,
        return_sequences=True,
        kernel_regularizer=l2(0.01),
        recurrent_dropout=0.2
    )(x)
    x = LSTM(
        lstm_units // 2,
        return_sequences=False,
        kernel_regularizer=l2(0.01),
        recurrent_dropout=0.2
    )(x)

    # --- Projection Head (for contrastive learning) ---
    projections = Dense(projection_dim, activation='linear')(x)

    # --- Model ---
    model = Model(inputs=inputs, outputs=projections)

    # --- Custom InfoNCE Loss ---
    def infonce_loss(y_true, y_pred):
        """
        Computes InfoNCE loss for contrastive learning.
        Assumes:
        - y_pred contains stacked [query, positive_key, negative_keys...].
        - Temperature is a hyperparameter.
        """
        batch_size = tf.shape(y_pred)[0] // 2  # Query + positive pairs
        queries = y_pred[:batch_size]
        keys = y_pred[batch_size:]

        # Positive pairs: diagonal elements (query[i] vs key[i])
        pos_sim = tf.reduce_sum(queries * keys[:batch_size], axis=-1)  # Shape: (batch_size,)

        # Negative pairs: query[i] vs key[j] (j != i)
        neg_sim = tf.matmul(queries, keys[batch_size:], transpose_b=True)  # Shape: (batch_size, num_negatives)

        # Combine and compute softmax
        logits = tf.concat([
            tf.expand_dims(pos_sim, -1),
            neg_sim
        ], axis=-1) / temperature

        labels = tf.zeros(batch_size, dtype=tf.int32)  # Positives are at index 0
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        return tf.reduce_mean(loss)

    # Compile with dummy loss (actual loss is computed in a custom training loop)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=infonce_loss)

    return model

