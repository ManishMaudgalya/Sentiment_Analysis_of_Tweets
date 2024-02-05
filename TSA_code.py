from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.svm import SVC

def load_data(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset
def preprocessing_tweet(tweet):
    tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    return " ".join(filtered_words)
def get_feature_vector(tf):
    v = TfidfVectorizer(sublinear_tf=True)
    v.fit(tf)
    return v
def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

stop_words = set(stopwords.words('english'))

train_data = load_data("train.csv", ['tweet_id','sentiment', 'tweet_text'])
test_data = load_data("test.csv", ['tweet_id', 'tweet_text'])

n_train_data = remove_unwanted_cols(train_data, ['tweet_id']) 

train_data.text = train_data['tweet_text'].apply(preprocessing_tweet)
test_data.text = test_data['tweet_text'].apply(preprocessing_tweet)

train_tf_vector = get_feature_vector(np.array(train_data.iloc[:, 1]).ravel())
test_tf_vector = get_feature_vector(np.array(train_data.iloc[:, 1]).ravel())
X_train = train_tf_vector.transform(np.array(train_data.iloc[:, 1]).ravel())
y_train = np.array(train_data.iloc[:, 0]).ravel()
X_test = test_tf_vector.transform(np.array(test_data.iloc[:, 1]).ravel())


LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)

