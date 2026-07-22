!unzip amazon_baby.csv.zip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import time
from sklearn.linear_model import LogisticRegression
def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
baby_df = pd.read_csv('amazon_baby.csv')
baby_df.head()
#Exercise 1 (data preparation)
#a) Remove punctuation from reviews using the given function.   
#b) Replace all missing (nan) revies with empty "" string.  
#c) Drop all the entries with rating = 3, as they have neutral sentiment.   
#d) Set all positive ($\geq$4) ratings to 1 and negative($\leq$2) to -1.
#a)
baby_df['review'] = baby_df['review'].fillna('')
baby_df['review'] = baby_df['review'].apply(remove_punctuation)
#b)
baby_df['review'] = baby_df['review'].fillna('')
baby_df['review'] = baby_df['review'].apply(remove_punctuation)
#short test:
baby_df["review"][38] == baby_df["review"][38]
#c)
baby_df = baby_df[baby_df["rating"] != 3]
sum(baby_df["rating"] == 3)
#d)
baby_df["rating"] = baby_df["rating"].apply(lambda x: 1 if x >= 4 else -1)
sum(baby_df["rating"]**2 != 1)
# CountVectorizer
#In order to analyze strings, we need to assign them numerical values. We will use one of the simplest string representation, which transforms strings into the $n$ dimensional vectors. The number of dimensions will be the size of our dictionary, and then the values of the vector will represent the number of appereances of the given word in the sentence.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
reviews_train_example = ["We like apples",
                   "We hate oranges",
                   "I adore bananas",
                   "We like like apples and oranges",
                   "They dislike bananas"]
X_train_example = vectorizer.fit_transform(reviews_train_example)
print(vectorizer.get_feature_names_out())
print(X_train_example.todense())
reviews_test_example = ["They like bananas",
                   "We hate oranges bananas and apples",
                   "We love bananas"]
X_test_example = vectorizer.transform(reviews_test_example)
print(X_test_example.todense())
We should acknowledge few facts. Firstly, CountVectorizer does not take order into account. Secondly, it ignores one-letter words (this can be changed during initialization). Finally, for test values, CountVectorizer ignores words which are not in it's dictionary.
# Exercise 2
#a) Split dataset into training and test sets.     
#b) Transform reviews into vectors using CountVectorizer.
#a)
from sklearn.model_selection import train_test_split
reviews = baby_df['review']
ratings = baby_df['rating']
X_train, X_test, y_train, y_test = train_test_split(
    reviews, ratings, test_size=0.2, random_state=42
)
#b)
vectorizer = CountVectorizer()
XX_train_example = vectorizer.fit_transform(reviews)
print(vectorizer.get_feature_names_out())
print(X_train_example.todense())
# Exercise 3
#a) Train LogisticRegression model on training data (reviews processed with CountVectorizer, ratings as they were).   
#b) Print 10 most positive and 10 most negative words.
#a)
from sklearn.linear_model import LogisticRegression
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
#b)
coefs = model.coef_[0]
words = vectorizer.get_feature_names_out()
pos_words = words[coefs > 0][:10]
neg_words = words[coefs < 0][:10]
print("10 słów pozytywnych (coef > 0):")
print(pos_words)
print("\n10 słów negatywnych (coef < 0):")
print(neg_words)
# Exercise 4
#a) Predict the sentiment of test data reviews.   
#b) Predict the sentiment of test data reviews in terms of probability.   
#c) Find five most positive and most negative reviews.   
#d) Calculate the accuracy of predictions.
#a)
y_pred = model.predict(X_test_vec)
print("Przykładowe predykcje:", y_pred[:10])
#b)
y_prob = model.predict_proba(X_test_vec)
print("Przykładowe prawdopodobieństwa:\n", y_prob[:5])
#c)
top5_pos_idx = np.argsort(y_prob[:,1])[-5:][::-1]
top5_neg_idx = np.argsort(y_prob[:,0])[-5:][::-1]
print("5 najbardziej pozytywnych recenzji:")
for i in top5_pos_idx:
    print(X_test.iloc[i])
print("\n5 najbardziej negatywnych recenzji:")
for i in top5_neg_idx:
    print(X_test.iloc[i])
#d)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność predykcji:", accuracy)
# Exercise 5
#In this exercise we will limit the dictionary of CountVectorizer to the set of significant words, defined below.
#a) Redo exercises 2-5 using limited dictionary.   
#b) Check the impact of all the words from the dictionary.   
#c) Compare accuracy of predictions and the time of evaluation.
significant_words = ['love','great','easy','old','little','perfect','loves','well','able','car','broke','less','even','waste','disappointed','work','product','money','would','return']
#a)
vectorizer_lim = CountVectorizer(vocabulary=significant_words, lowercase=True)
X_train_vec_lim = vectorizer_lim.fit_transform(X_train)
X_test_vec_lim = vectorizer_lim.transform(X_test)
model_lim = LogisticRegression(max_iter=1000)
model_lim.fit(X_train_vec_lim, y_train)
y_pred_lim = model_lim.predict(X_test_vec_lim)
y_prob_lim = model_lim.predict_proba(X_test_vec_lim)
top5_pos_idx = np.argsort(y_prob_lim[:,1])[-5:][::-1]
top5_neg_idx = np.argsort(y_prob_lim[:,0])[-5:][::-1]
print("5 najbardziej pozytywnych recenzji:")
for i in top5_pos_idx:
    print(X_test.iloc[i])
print("\n5 najbardziej negatywnych recenzji:")
for i in top5_neg_idx:
    print(X_test.iloc[i])
#b)
coefs_lim = model_lim.coef_[0]
words_lim = vectorizer_lim.get_feature_names_out()
print("\nWpływ słów z ograniczonego słownika:")
for word, coef in zip(words_lim, coefs_lim):
    print(word, coef)
#c)
vectorizer_full = CountVectorizer(lowercase=True)
X_train_vec_full = vectorizer_full.fit_transform(X_train)
X_test_vec_full = vectorizer_full.transform(X_test)
model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_train_vec_full, y_train)
start = time.time()
y_pred_full = model_full.predict(X_test_vec_full)
end = time.time()
print("\nAccuracy (full dictionary):", accuracy_score(y_test, y_pred_full))
print("Time (full dictionary):", end-start)
start = time.time()
y_pred_lim = model_lim.predict(X_test_vec_lim)
end = time.time()
print("Accuracy (limited dictionary):", accuracy_score(y_test, y_pred_lim))
print("Time (limited dictionary):", end-start)
#hint: %time, %timeit


