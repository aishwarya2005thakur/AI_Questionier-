
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import json

with open("sarcasm.json",'r') as f:
    datastore = json.load(f)

    sentences = []
    lables = []
    urls = []
    for item in datastore:
        sentences.append(item['headline'])
        lables.append(item['is_sarcastic'])
        urls.append(item['article_link'])
        
#sample sentences
sentences = [
    'I love my dog',
    'I love my cat',
    'I love my parrot',
    'My dog is amazing and very friendly'
]

#defining tensorflow model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()

# Initialize Tokenize
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
#used to introduce new  words in the model = "<OOV>"
tokenizer.fit_on_texts(sentences)

# Convert words to indices
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

# Add padding for consistency in sentence length taking into account the length of the longest sentence length
padded_sequences = pad_sequences(sequences , padding = 'post')

# Print outputs
print("Word Index:", word_index)
print("Sequences:", sequences)
print("Padded Sequences:\n", padded_sequences)
print("tf._version")

# Save only the model weights
model.save_weights('model_weights.h5')

# Save in a smaller precision (quantized model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save the quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_model)
