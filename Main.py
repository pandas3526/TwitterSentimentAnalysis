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

lab_to_sentiment = {0:"Negative", 4:"Positive"}
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
"""Now we will implement our model with our findings above. In our data there are some words feature in both Positive and Negative tweets,
    This could be a problem in our learning model. That's why we use Sequence Models. For model architecture, we use
	Embedding Layer - Generates Embedding Vector for each input sequence.
	Conv1D Layer - Its using to convolve data into smaller feature vectors.
	LSTM - Long Short Term Memory, its a variant of RNN which has memory state cell to learn the context of words which are at further along the text to carry contextual meaning rather than just neighbouring words as in case of RNN.
	Dense - Fully Connected Layers for classification
"""
#importing necessary libraries

from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D

#Building Sequence Models

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') #we use input as tokenized and encoded data as starter
embedding_sequences = embedding_layer(sequence_input) #after that we fit our input to embedding layer we created above

#Now we will implement layers to our embedded data by order below

x = SpatialDropout1D(0.2)(embedding_sequences) #it drops entire 1D feature maps

x = Conv1D(64, 5, activation='relu')(x) #to convolve data into smaller feature vectors.

x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x) #we will use bidirectional LSTM to implement our model

x = Dense(512, activation='relu')(x) #by adding dense layer, we feed all outputs from the previous layer to all its neurons, each neuron providing one output to the next layer. (This is enterance dense layer)

x = Dropout(0.5)(x) #to prevent overfitting we use dropout layer(it sets input units to 0 with a frequency of rate at each step during training time)
			#dropout layer can only be used on fully connected layers, it prevents overfitting by deactivating some neurons, Fine Tuning

x = Dense(512, activation='relu')(x) #(This is exit dense layer)
outputs = Dense(1, activation='sigmoid')(x) #final outputs
model = tf.keras.Model(sequence_input, outputs) #fitting model

"""We created our LSTM model, now we will implement Optimization Algorithm with callbacks and start training
For optimization algorithm we will use Adam. We will use 2 callbacks. Callbacks are special functions which are called at the end of an epoch. The callbacks we used here:

LRScheduler - It changes a Learning Rate at specfic epoch to achieve more improved result. In this notebook, the learning rate exponentionally decreases after remaining same for first 10 Epoch.
ModelCheckPoint - It saves best model while training based on some metrics. Here, it saves the model with minimum Validity Loss.""" 

from tensorflow.keras.optimizers import Adam  #Adam optimization algorithm
from tensorflow.keras.callbacks import ReduceLROnPlateau #Callbacks

model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', #Optimizing
              metrics=['accuracy'])
ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,	#Using callbacks
                                     min_lr = 0.01,
                                     monitor = 'val_loss',
                                     verbose = 1)
#We setted everything up, now we will start training
print("Training on GPU...") if tf.config.list_physical_devices('GPU') else print("Training on CPU...")
import time
start = time.time()

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])

print("Training completed...")
stop = time.time()
print("Training time: {}h".format((stop-start)/3600))

"""We finished building our model and trained our model with the data. Now we will test the accuracy of our model. 
Since we trained our model to determine the sentiment of a given sentence, we will test the program in accordance with 
this purpose.
"""

SENTIMENT_THRESHOLDS = (0.4, 0.7) 
"""The function below will be used to give meaning to our trained model with scores. Label will be determined by the comparison
of score and thereshold values above. 
Label is:
    'Neutral' if 0.4<score<0.7 
    'Negative' if score<=0.4 
    'Positive' if score>=0.7      
"""
####################Model Evaluation####################
"""Now that we have trained the model, we can evaluate its performance. 
We will some evaluation metrics and techniques to test the model."""

#Accuracy of our sentiment analysis model
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
#Learning Curve of loss and accuracy of the model on each epoch.
s, (at, al) = plt.subplots(2,1)
at.plot(history.history['accuracy'], c= 'b')
at.plot(history.history['val_accuracy'], c='r')
at.set_title('model accuracy')
at.set_ylabel('accuracy')
at.set_xlabel('epoch')
at.legend(['LSTM_train', 'LSTM_val'], loc='upper left')

al.plot(history.history['loss'], c='m')
al.plot(history.history['val_loss'], c='c')
al.set_title('model loss')
al.set_ylabel('loss')
al.set_xlabel('epoch')
al.legend(['train', 'val'], loc = 'upper left')
plt.close()

####################Model Testing####################
"""In this part, To test accuracy of our model, we will send some sentences as user input to our trained model for testing."""

def decode_sentiment(score, include_neutral=True):
    label = 'Neutral'
    if score <= SENTIMENT_THRESHOLDS[0]:
        label = 'Negative'
    elif score >= SENTIMENT_THRESHOLDS[1]:
        label = "Positive"

    return label
    
"""This function takes text as input and determine if label of the text is Positive, Negative or Neutral"""    
def predict(text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=MAX_SEQUENCE_LENGTH)
    # Predict score
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score)

    return {"label": label, "score": float(score)}

scores = model.predict(x_test, verbose=1, batch_size=10000) 

y_pred_1d = [decode_sentiment(score) for score in scores]

#User interface to use the program; It takes a text input from the user and print its label

def start_program():
    userInput=""
    while(userInput!="q"):
        userInput=input("Enter a sentence: \n Press q to quit program ")
        if(userInput=="q"):
            time.sleep(2)
            print("Exiting program...")
        else:    
            print(predict(userInput))

start_program()      

      


