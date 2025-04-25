import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Load sarcasm dataset
with open("arcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Tokenize
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=32, padding='post', truncating='post')

# Convert labels to numpy array
import numpy as np
labels = np.array(labels)

# Split dataset (basic)
train_size = int(len(sentences) * 0.8)
X_train = padded_sequences[:train_size]
y_train = labels[:train_size]
X_test = padded_sequences[train_size:]
y_test = labels[train_size:]

# Build a basic model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
