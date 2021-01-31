from pandas import read_csv
import pandas as pd

from nltk.sentiment.util import *
from joblib import dump, load
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import sys

file_name = sys.argv[1]

#load dataset
dataset = read_csv(file_name)
del dataset["reviewTime"]
del dataset["reviewerName"]
del dataset["reviewerID"]
del dataset["asin"]
dataset['reviewText'] = dataset['reviewText'].apply(lambda x: str(x))
dataset['summary'] = dataset['summary'].apply(lambda x: str(x))
dataset['helpful'] = dataset['helpful'].apply(lambda x: eval(x))

new = pd.DataFrame(dataset["helpful"].to_list(), columns=['helpful', 'nothelpful'])
new["helpful_diff"] = new["helpful"] - new["nothelpful"] 

dataset['helpful'] = new['helpful']
dataset['nothelpful'] = new['nothelpful']
dataset['helpful_diff'] = new['helpful_diff']

#calculate reviewText and summary scores
with open('reviewText.pkl', 'rb') as f:
    cv1, clf1 = pickle.load(f)

with open('summary.pkl', 'rb') as f:
    cv2, clf2 = pickle.load(f)

vector = cv1.transform(dataset['reviewText'])
predicted = clf1.predict(vector)
dataset['reviewText'] = predicted

vector = cv2.transform(dataset['summary'])
predicted = clf2.predict(vector)
dataset['summary'] = predicted

#calculate final scores

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
    
predicted = clf.predict(dataset[['helpful', 'reviewText', 'summary', 'unixReviewTime', 'nothelpful', 'helpful_diff']])
print("Accuracy:",metrics.accuracy_score(dataset['score'], predicted))
