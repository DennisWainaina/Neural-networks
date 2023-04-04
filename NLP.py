x = 5
print(x)

# Natural language processing is a discipline in computing that deals with communication between natural(human) languages
# and computer languages.
# NLP models use recurrent neural networks which shall be discussed later.
# Here we will learn how to use RNN to do the following: Sentiment Analysis and Character generation.
# Sentiment analysis is about figuring out how positive or negative a sentence is.
# Character generation is about generating the next character in a sentence after previous data has been fed into it.
# We can for example use it to generate the next character in a play, after the previous data has been fed into it.
# The first problem is to figure out how we can turn text data into numeric data which the computer can understand.
# A common way of encoding data which people use though it may lead to problems is called bag of words.
# In this algorithm each word is assigned to a number in like a dictionary for example I:0 
# This 0 is something the computer can understand.
# All of these numbers which represent words are placed in something like a bag which is placed in the neural network.
# The neural network can then use this as training data.
# Word embedding is classifying words that are similar with similar numbers.
# This uses vectors to assign similar words.
# The vectors are in 3 dimensions where the hope is similar words are hopefully pointing in the same direction or similar.
# It looks at the angle between each word to see if there are similar at least that's a general idea of how it works.
# Word embeddings are a vectorized representation of words in a given document that places words with similar meanings
# to each other.
# RNN processes it one word at a time in a like internal loop that stores the words in like an internal memory.
# It uses the previous word stored in the internal memory as a basis to understand the next word.
# This is done at different timestamps from time t1 to tn
# The idea is to get a general understanding of the text after all the input has been processed word by word.
# There are types of RNNs for example: Simple Recurrent neural networks in which the current input is based on
# the understanding of the previous input and the output is the understanding of the previous input and current input
# We shall now build our own model for sentiment analysis

# First importing important libraries
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import pad_sequences 
import tensorflow as tf
import keras
import os
import numpy as np

VOCAB_SIZE = 88584  # Varies from 0 to 88584 with 0 being the most popular and 88584 being the least popular

MAXLEN = 250
BATCH_SIZE = 64

# Splitting the data into training and test data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

# Looking at one review
print(train_data[0])
print('Before padding the length is', len(train_data[0]))

# We now do some pre-processing where we change the max words to be 250 such that it may add or remove words to be 250.
# This is because the training data has different lengths hence cannot be fed into the model
train_data = pad_sequences(train_data, maxlen=MAXLEN)
test_data = pad_sequences(test_data, maxlen=MAXLEN)
print('After padding the length is', len(train_data[0]))
print(train_data[0])

# Building the model
model = tf.keras.Sequential(
    [
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 0 for bad 1 for good
    ]
)

# Seeing the summary
model.summary()

# Training the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)  # Validation split being 0.2 means 20% is used to evaluate model

# Evaluating the model
results = model.evaluate(test_data, test_labels)
print(results)

# Before making predictions the data which we want to make a prediction on must be processed the same way as the data which was preprocessed.
# This is because if the ML model thinks the data is different the predictions will not be as accurate.
# For example;
word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], MAXLEN)[0]

text = 'that movie was just amazing, so amazing'
encoded = encode_text(text)
print(encoded)

# One can also decode text such that from numbers to text
reversed_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ''
    for num in integers:
        if num != PAD:
            text = text + reversed_word_index[num] + " "
    
    return text[:-1]

print(decode_integers(encoded))

# Using it to make a prediction
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, MAXLEN))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I ever watched"
predict(negative_review)
