# -*- coding: utf-8 -*-
"""
@author: Burcu İçen, Çağrıhan Günay

Twitter Sentiment Analysis

"""

#Importing necessary libraries:
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk 
nltk.download('stopwords') #for stopword filtering
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer #for stemming process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder #for Label Encoding
import re

#Loading sentiment analysis data:

df = pd.read_csv('Sentiment Analysis data .csv',
                 encoding = 'latin',header=None)
####################Data Preprocessing####################
"""To start exploring and evaulating our model we should start with data preprocessing. Data preprocessing arranges the data 
and gives us the necessery parts of the data. In our data there are 6 columns, but we will only use 2 columns, which are
sentiment and text(tweets).In order to get rid of unnecessary space, we will remove the 4 columns.
"""
#Since column names are (1,2,3,4,5,6), we rename the columns to proper names. As ('sentiment', 'id', 'date', 'query', 'user_id', 'text')

df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']

#Since we are going to train only on text to classify its sentiment. So we can drop the rest of the columns. Because they are unnecessary.

df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)

#In the sentiment column, there are 3 values, 0 indicating that the sentiment is negative,2 indicating that the sentiment is neutral, 4 indicates that sentiment is positive. That's why we change the values(0,4) to (Negative, Positive)

lab_to_sentiment = {0:"Negative",2:"Neutral 4:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))

print("Data Preprocessing completed...")

####################Text Preprocessing####################
"""Tweets often contains mentions, hyperlink texts, emoticons and punctuations. 
To use them for learning, we should eliminate those texts, 
So we have to clean the text data first with various methods: We will use Stemming/ Lematization to preprocess the data first
to do that we will use stop word filtering, we will use NLTK, a python library that has functions to perform text processing task.
"""
stop_words = stopwords.words('english') #a list of stop words(to remove them)
stemmer = SnowballStemmer('english') #stemmer

text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+" #will be used to remove hyperlink texts, emoticons and punctuations from the tweets

#this function performs text proprocessing to tweets data

def preprocess(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip() #to remove hyperlink texts, emoticons and punctuations from the tweets
  tokens = []
  for token in text.split():
    if token not in stop_words:   #to remove stop words
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

#finally we will use the function to preprocess the text data

df.text = df.text.apply(lambda x: preprocess(x))
print("Text Preprocessing completed...")

"""Since we finished preprocessing our data now we can split it to training and testing to start evulating our model
"""
TRAIN_SIZE = 0.8
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 30


train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE,
                                         random_state=7) # Splits Dataset into Training and Testing set

####################Tokenization####################

""" To understand the text data better, we should do the mapping by indexing every words used in tweets. To do that, we will
use a method called "tokenization". tokenization is the task of chopping it up into pieces, called tokens, 
perhaps at the same time throwing away certain characters, such as punctuation.
"""
from keras.preprocessing.text import Tokenizer #we will use special keras library for tokenization process

tokenizer = Tokenizer() # It create tokens for every word in the data and map them to a index using dictionary.
tokenizer.fit_on_texts(train_data.text) #fitting the model

word_index = tokenizer.word_index   #It contains the index for each word
vocab_size = len(tokenizer.word_index) + 1  #It represents the total number of word in the data

"""Now we have the tokenizer, we will use that object to convert any word into a "key" in dictionary. 
Since we are going to build a sequence model, we should be sure that there is no variance in input shapes
It all should be of same length. But texts in tweets have different count of words in it. 
To avoid this, we will make all the sequence in one constant length using pad_sequences from keras library.
"""
from keras.preprocessing.sequence import pad_sequences 

x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text), 
                        maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text),
                       maxlen = MAX_SEQUENCE_LENGTH)

print("Tokenization completed...")

"""Since we are building the model to predict class in enocoded form as binary classification. 
We should encode our training labels to encodings. To do that we will use label encoding.
"""
encoder = LabelEncoder() #building the model
encoder.fit(train_data.sentiment.to_list()) #fitting the model

y_train = encoder.transform(train_data.sentiment.to_list()) #encoding labels of testing and training data
y_test = encoder.transform(test_data.sentiment.to_list())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("Label Encoding completed...")


#As a result our y_train.shape and y_test.shape values are now 1. We encoded labels of testing and training data to use them in our model

####################Word Embedding####################

"""In the next step, we used word embedding to capture context of a word in a document, semantic and syntactic similarity, 
relation with other words, etc. There are many ways to do word embedding. Pretrained models can be used for such a project, 
such as  Word2vec, GloVe Vectors, fastText, We used GloVe Vectors from Stanford University in this project. We downloaded 
the files and unzip them, and put the files to a directory called "GloVe Embedding Stanford AI" in the project files and used
"glove.6B.300d.txt" file for word embedding task. You can find the link here: http://nlp.stanford.edu/data/glove.6B.zip
"""

GLOVE_EMB = "GloVe Embedding Stanford AI\glove.6B.300d.txt" #the path of the txt file
EMBEDDING_DIM = 300 #embedding dimension 
LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 10
embeddings_index = {}

f = open(GLOVE_EMB,encoding="utf8")
#we can update our dictionary with the new word and its correspending vector
for line in f:
  values = line.split()
  word = value = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False) 
print("Word embedding completed...")


