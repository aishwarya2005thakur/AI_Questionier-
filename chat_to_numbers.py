import tensorflow as tf
from tensorlfow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentence = [
' I love my dog' ,
' I love my cat' ,
' I love my parrot '
]

tokenizer = Tokenizer( num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_indxe)