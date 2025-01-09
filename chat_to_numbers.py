pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentence = [
' I love my dog' ,
' I love my cat' ,
' I love my parrot '
]

tokenizer = Tokenizer( num_words = 100)
tokenizer.fit_on_texts(sentence)
word_index = tokenizer.word_index
print(word_index)