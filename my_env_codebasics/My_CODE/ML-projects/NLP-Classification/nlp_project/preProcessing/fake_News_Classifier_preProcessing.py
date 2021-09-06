import pickle

import pandas as pd
import numpy as np
# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# for text processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF library

def Transformations(featureDF, target):
    print(">>feature shape before cleaning: ",featureDF.shape)
    print(">>target shape before cleaning: ",target.shape)
    featureDF = featureDF.drop(['id', 'text', 'author'], axis = 1) # drop unwanted columns
    featureDF = featureDF[featureDF['title'].notnull()] # eliminate null values in title column
    # Duplicate elimination
    featureDF.drop_duplicates(inplace=True)
    # Update y matrix based X
    ## since we've removed some data from X, we need to pass on these updations to y as well, as y doesn't know some of its corresponding X's have been deleted.
    target = target[featureDF.index]
    print(">>feature shape after cleaning: ",featureDF.shape)
    print(">>target shape after cleaning: ",target.shape)
    
    ps = PorterStemmer()
    def text_transform(paragraph):
        text = re.sub(r'\[[0-9]*\]',' ',paragraph).lower()
        text = re.sub(r'\d',' ',text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub(r'\s+',' ',text)
        sentences = nltk.sent_tokenize(text)
        corpus = []
        for i in range(len(sentences)):
            review = sentences[i].split() #converting to list of words
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            corpus.append(review)
        return ' '.join(corpus)
    
    featureDF['title'] = featureDF['title'].map(lambda para: text_transform(para))
    # Update y matrix based X
    ## since we've removed some data from X, we need to pass on these updations to y as well, as y doesn't know some of its corresponding X's have been deleted.
    target = target[featureDF.index]
    print(">>feature shape after preProcessing: ",featureDF.shape)
    print(">>target shape after preProcessing: ",target.shape)
    
    return featureDF, target