
# Sentiment Analysis
>Sentiment analysis (or opinion mining) is a natural language processing technique used to determine whether data is positive, negative or neutral.


![](https://i.ibb.co/wLvmwzc/Ads-z-tasar-m-1.png)




# Project Description
The purpose of this project is to train a model with a dataset of 1.6 million tweets to detect the sentiment of a given sentence.

+ Data Preprocessing
+ Text Preprocessing
    + Stop Word Filtering
    + Stemming
+ Tokenization
    * Label Encoding
    * Padding Sequences
+ Word Embeding
    * GloVe Embeding
+ Model Training
    * Sequnce model
         + Embening Layer 
         + Conv1D Layer
         + LSTM
         + Density Layer
         + Optimization
+ Model Evaluation
     * Accuracy Score
     * Learning Curve
+ Model Testing
     *Decoding Sentiment


# 1-)Data Preprocessing
>Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, lacking in certain behaviors or trends, and is likely to contain many errors.
Data preprocessing is a proven method of resolving such issues. Data preprocessing prepares raw data for further processing .That's why we preprocessed our data to make it more useful, and we made 2 columns more useful by deleting 4 out of 6 columns in our data.
# 2-)Text Preprocessing
>To preprocess your text simply means to bring your text into a form that is predictable and analyzable for your task.Therefore, we do text preprocessing by removing unnecessary characters so that the tweets contained in the data can be analyzed clearly.Text Preporcessing was implemented in two steps in this project(Stop Word Filtering, Stemming). We will use Stemming/ Lematization to preprocess the data first, to do that we will use stop word filtering, we will use NLTK, a python library that has functions to perform text processing task.

# 3-)Tokenization
>To understand the text data better, we should do the mapping by indexing every words used in tweets. To do that, we will use a method called "tokenization". tokenization is the task of chopping it up into pieces, called tokens, perhaps at the same time throwing away certain characters, such as punctuation. After that, we will use that object to convert any word into a "key" in dictionary. Since we are going to build a sequence model, we should be sure that there is no variance in input shapes It all should be of same length. But texts in tweets have different count of words in it. To avoid this, we will make all the sequence in one constant length using pad_sequences from keras library. Since we are building the model to predict class in enocoded form as binary classification. We should encode our training labels to encodings. To do that we will use label encoding.
# 4-)Word Embeding
>A word embedding is a learned representation for text where words that have the same meaning have a similar representation so we used word embedding to capture context of a word in a document, semantic and syntactic similarity,  relation with other words, etc. There are many ways to do word embedding. Pretrained models can be used for such a project, such as  Word2vec, GloVe Vectors, fastText, We used GloVe Vectors from Stanford University in this project. We downloaded the files and unzip them, and put the files to a directory called "GloVe Embedding Stanford AI" in the project files and used"glove.6B.300d.txt" file for word embedding task. You can find the link here: http://nlp.stanford.edu/data/glove.6B.zip
# 5-)Model Training
>In our data there are some words feature in both Positive and Negative tweets,This could be a problem in our learning model. That's why we use Sequence Models. For model architecture, we use:
	Embedding Layer - Generates Embedding Vector for each input sequence.
	Conv1D Layer - Its using to convolve data into smaller feature vectors.
	LSTM - Long Short Term Memory, its a variant of RNN which has memory state cell to learn the context of words which are at further along the text to carry contextual meaning rather than just neighbouring words as in case of RNN.
	Dense - Fully Connected Layers for classification
We created our LSTM model, now we will implement Optimization Algorithm with callbacks and start training. For optimization algorithm we will use Adam. We will use 2 callbacks. Callbacks are special functions which are called at the end of an epoch. The callbacks we used here:
LRScheduler - It changes a Learning Rate at specfic epoch to achieve more improved result. In this notebook, the learning rate exponentionally decreases after remaining same for first 10 epoch.
Here is a basic understanding of our model:

![](https://i.ibb.co/dBtN5f4/urban-nature-3.png)


# 6-)Model Testing
>Since we trained our model to determine the sentiment of a given sentence, we will test the program in accordance with  this purpose. Label will be determined by the comparisonof score and thereshold. 

Pseudocode is :

    'Neutral' if 0.4<score<0.7 
    'Negative' if score<=0.4 
    'Positive' if score>=0.7 
>To test accuracy of our model, we will send some sentences as user input to our trained model for testing. The program takes a text input from the user and print its label


### Team&Roles

>Project Team Members: Burcu İÇEN, Çağrıhan GÜNAY

>Project is developed in three parts, First part is preparing the data and text for model training; such as data preprocessing, text preprocessing, tokenization, label encoding, word embeding. This part has been prepared by Çağrıhan GÜNAY. And the second part is, creating the model and training it; such as building sequence model, implementing optimization and callbacks, training the model and finally calculating the accuracy score and plotting learning curve. This part has been prepared by Burcu İÇEN. Last part of the project where user can enter a sentence to decode its sentiment has been prepared by Burcu İÇEN and Çağrıhan GÜNAY together.


### Structure
```
Twitter Sentiment Analysis
│   README.md
│   requirement.txt 
│   main.py 
│   Sentiment Analysis data .csv
└───GloVe Embedding Stanford AI
    │   glove.6B.50d.txt
    │   glove.6B.100d.txt
    │   glove.6B.200d.txt
    │   glove.6B.300d.txt
└───Report
    
```
### requirement.txt: 
>It states dependencies of our project to be interpreted by Python. To load dependencies, you should run following command
```sh
$ python -m pip install -r requirements.txt.
```
### main. py: 
>Main file of the project, It envolves all parts, Data preprocessing to Model Testing. To run the main file you should run following command:
```sh
$ python main.py
```
### Sentiment Analysis data .csv: 
>Data csv file to train the model. Dataset name is, Sentiment140 dataset with 1.6 million tweets and it contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment. Link for the dataset: https://www.kaggle.com/kazanova/sentiment140
### GloVe Embedding Stanford AI files:
>We used GloVe Vectors from Stanford University as word embedding model in this project. We downloaded the files and unzip them, and put the files to a directory called "GloVe Embedding Stanford AI" in the project files and used
"glove.6B.300d.txt" file for word embedding task. You can find the link here: http://nlp.stanford.edu/data/glove.6B.zip

### Report files:
>Detailed report of the project, Main file of the report is Report.tex

### Language, version, and main file
>Language: Python
Version: 3.8.3
Main file: main.py




