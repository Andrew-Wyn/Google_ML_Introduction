#Import libraries
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer



#Tokenization (a list of tokens), will be used as the analyzer
#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words


if __name__ == "__main__":
    #Load the data
    #from google.colab import files # Use to load data on Google Colab
    #uploaded = files.upload() # Use to load data on Google Colab
    df = pd.read_csv('email.csv')
    print(df['text'].head(1))

    #Checking for duplicates and removing them
    df.drop_duplicates(inplace = True)

    #Show the number of missing (NAN, NaN, na) data for each column
    df.isnull().sum()

    #Need to download stopwords
    nltk.download('stopwords')

    #Show the Tokenization (a list of tokens )
    df['text'].head().apply(process_text)

    vectorizer = CountVectorizer(analyzer=process_text)
    # feature
    messages_bow = vectorizer.fit_transform(df['text'].head(2)) # dovrebbe essere una feature impostata come binned/bucketized rivedere parte del corso
    # capire come identificare univocamente le parole

    print(vectorizer.get_feature_names())

    print(len(messages_bow.toarray()))

