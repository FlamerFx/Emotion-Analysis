import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pickle

# Paths
DATA_PATH = os.path.join("python_ml_server", "data", "test.txt")
GLOVE_PATH = os.path.join("python_ml_server", "data", "glove.6B.300d.txt")
MODEL_PATH = os.path.join("python_ml_server", "model", "model.keras")  # updated to new Keras format
TOKENIZER_PATH = os.path.join("python_ml_server", "model", "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join("python_ml_server", "model", "label_encoder.pkl")
TFLITE_MODEL_PATH = os.path.join("python_ml_server", "model", "model.tflite")  # path for tflite model
SAVED_MODEL_PATH = os.path.join("python_ml_server", "model", "saved_model")    # path for TF SavedModel

# Parameters
MAX_LEN = 100
EMBEDDING_DIM = 300

# Load dataset
print("Loading dataset...")
data = pd.read_csv(DATA_PATH, sep=";", names=["text", "emotion"])
print(data.head())
print(f"Loaded {len(data)} records.")

# Tokenize text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(data["text"])
sequences = tokenizer.texts_to_sequences(data["text"])
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["emotion"])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=0.2, random_state=42)

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path, embedding_dim, word_index):
    print("Loading GloVe embeddings...")
    embeddings_index = {}
    with open(glove_file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("GloVe embeddings loaded.")
    return embedding_matrix

embedding_matrix = load_glove_embeddings(GLOVE_PATH, EMBEDDING_DIM, tokenizer.word_index)

# Build model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_LEN,
                    trainable=False))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
print("Training model...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64, callbacks=[early_stop])

# Evaluate model using classification report
print("Evaluating model on validation set...")
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# Save model and tokenizer
print("Saving model and tokenizer...")
model.save(MODEL_PATH)
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)
with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
print("Model and tokenizer saved.")

# Export model to TensorFlow Lite format for C++ inference
print("Exporting model to TensorFlow Lite format for C++ inference...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Model exported to TensorFlow Lite format at {TFLITE_MODEL_PATH}")

# Export model as TensorFlow SavedModel for TensorFlow C API (C/C++)
print("Exporting model as TensorFlow SavedModel for C/C++ API...")
model.export(SAVED_MODEL_PATH)
print(f"Model exported as TensorFlow SavedModel at {SAVED_MODEL_PATH}")
