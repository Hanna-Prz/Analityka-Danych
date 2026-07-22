#When exploring a large set of documents -- such as Wikipedia, news articles, StackOverflow, etc. -- it can be useful to get a list of related material. To find relevant documents you typically
#* Decide on a notion of similarity
#* Find the documents that are most similar
#In the assignment you will
#* Gain intuition for different notions of similarity and practice finding similar documents.
#* Explore the tradeoffs with representing documents using raw word counts and TF-IDF
#* Explore the behavior of different distance metrics by looking at the Wikipedia pages most similar to President Obamaâs page.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
%matplotlib inline
!unzip /content/people_wiki.cvs.zip

def compute_length(row):
    return len(tokenizer(row['text']))

def contains_all_words(text):
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    return all(word in words_in_text for word in top_words_15)

def top_words_tf_idf(name):
    """
    Get a table of the largest TF-IDF words in the given person's Wikipedia page,
    ignoring common English stopwords.
    """
    index = wiki[wiki['name'] == name].index[0]
    text = wiki.loc[index, 'text']
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stopwords_list = list(ENGLISH_STOP_WORDS)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords_list, min_df=1)
    vec = vectorizer.fit_transform([text])
    tfidf = TfidfTransformer(smooth_idf=False, norm=None)
    tfidf_vec = tfidf.fit_transform(vec).toarray().ravel()
    df = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'tf-idf': tfidf_vec
    })
    return df.sort_values(by='tf-idf', ascending=False)

def contains_all_words(text):
    words_in_text = set(re.findall(r'\b\w+\b', text.lower()))
    return all(word in words_in_text for word in top_words_15)
mask = wiki['text'].apply(contains_all_words)
articles = wiki.loc[mask, 'name'].tolist()

def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    index = wiki[wiki['name'] == name].index[0]
    text = wiki.loc[index, 'text']
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, max_features=10000, min_df=1)
    vec=vectorizer.fit_transform([text])
    df = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'count': vec.toarray().ravel()
    })


    return df.sort_values(by='count',ascending=False)
#We will be using the dataset of abridged Wikipedia pages. Each element of the dataset consists of a link to the wikipedia article, the name of the person, and the text of the article (in lowercase).  
wiki = pd.read_csv('people_wiki.csv')
wiki.head()
#As we have seen in Assignment 4, we can extract word count vectors using `CountVectorizer` function.
#- make sure you include words of unit length by using the parameter: `token_pattern=r"(?u)\b\w+\b"`
#- do not use any stopwords
#- take 10000 most frequent words in the corpus
#- explicitly take all the words independent of in how many documents they occur
#- obtain the matrix of word counts
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, max_features=10000, min_df=1)
WCmatrix = vectorizer.fit_transform(wiki['text'])
#a) Start by finding the nearest neighbors of the Barack Obama page using 
#   Element listy
#   Element listy
#the above word count matrix to represent the articles and **Euclidean** distance to measure distance.
#Save the distances in `wiki['BO-eucl']` and look at the top 10 nearest neighbors.
index = wiki[wiki['name'] == 'Barack Obama'].index[0]
row = WCmatrix[index:index+1]
dist = pairwise_distances(row, WCmatrix, metric='euclidean').ravel()
wiki['BO-eucl'] = dist
top10 = wiki.sort_values('BO-eucl').head(10)
print(top10)
#b) Measure the pairwise distance between the Wikipedia pages of Barack Obama, George W. Bush, and Joe Biden. Which of the three pairs has the smallest distance?
index_bush = wiki[wiki['name'] == 'George W. Bush'].index[0]
index_biden = wiki[wiki['name'] == 'Joe Biden'].index[0]
vec_obama=WCmatrix[index]
vec_bush=WCmatrix[index_bush]
vec_biden=WCmatrix[index_biden]
dist_obama_bush = pairwise_distances(vec_obama,vec_bush,  metric='euclidean').ravel()
dist_obama_biden = pairwise_distances(vec_biden,vec_obama,  metric='euclidean').ravel()
dist_bush_biden = pairwise_distances(vec_bush,vec_biden, metric='euclidean').ravel()
distances = {
    'Obama-Bush': dist_obama_bush,
    'Obama-Biden': dist_obama_biden,
    'Bush-Biden': dist_bush_biden
}
closest_pair = min(distances, key=distances.get)
print( closest_pair)
print(dist_obama_bush)
print(dist_obama_biden)
print(dist_bush_biden)
#All of the 10 people from **a)** are politicians, but about half of them have rather tenuous connections with Obama, other than the fact that they are politicians, e.g.,
# Francisco Barrio is a Mexican politician, and a former governor of Chihuahua.
# Walter Mondale and Don Bonker are Democrats who made their career in late 1970s.
#Nearest neighbors with raw word counts got some things right, showing all politicians in the query result, but missed finer and important details.
#c) Let's find out why Francisco Barrio was considered a close neighbor of Obama.
#To do this, look at the most frequently used words in each of Barack Obama and Francisco Barrio's pages.
obama_words = top_words('Barack Obama')
obama_words
barrio_words = top_words('Francisco Barrio')
barrio_words
#d) Extract the list of most frequent **common** words that appear in both Obama's and Barrio's documents and display the five words that appear most often in Barrio's article.
#Use a dataframe operation known as **join**. The **join** operation is very useful when it comes to playing around with data: it lets you combine the content of two tables using a shared column (in this case, the index column of words). See [the documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) for more details.
obama_words_indexed = obama_words.set_index('word')
barrio_words_indexed = barrio_words.set_index('word')
common_words = obama_words_indexed.join(
    barrio_words_indexed,
    how='inner',
    lsuffix='_Obama',
    rsuffix='_Barrio'
)
common_words.sort_values(by='count_Barrio', ascending=False).head(5)
#Collect all words that appear both in Barack Obama and George W. Bush pages.  Out of those words, find the 10 words that show up most often in Obama's page.
obama_words = top_words('Barack Obama')
bush_words = top_words('George W. Bush')
obama_words_indexed = obama_words.set_index('word')
busho_words_indexed = bush_words.set_index('word')
common_words_bush = obama_words_indexed.join(
    barrio_words_indexed,
    how='inner',
    lsuffix='_Obama',
    rsuffix='_Bush'
)
common_words_bush.sort_values(by='count_Obama', ascending=False).head(10)
#e) Among the words that appear in both Barack Obama and Francisco Barrio, take the 15 that appear most frequently in Obama. How many of the articles in the Wikipedia dataset contain all of those 15 words? Which are they?
top_words_15 = common_words.sort_values(by='count_Obama', ascending=False).head(15).index.tolist()
print("Number of articles:", len(articles))
print("Names:", articles)
#Much of the perceived commonalities between Obama and Barrio were due to occurrences of extremely frequent words, such as "the", "and", and "his". So nearest neighbors is recommending plausible results sometimes for the wrong reasons.
#To retrieve articles that are more relevant, we should focus more on rare words that don't happen in every article. **TF-IDF** (term frequencyâinverse document frequency) is a feature representation that penalizes words that are too common.
#a) Repeat the search for the 10 nearest neighbors of Barack Obama with Euclidean distance of TF-IDF. This time do not limit to only 10000 most frequent words, but take all of them.
vectorizer = CountVectorizer(min_df=1)
WCmatrix = vectorizer.fit_transform(wiki['text'])
tfidf=TfidfTransformer(smooth_idf=False, norm=None)
TFIDFmatrix = tfidf.fit_transform(WCmatrix)
index_obama = wiki[wiki['name'] == 'Barack Obama'].index[0]
dist = pairwise_distances(TFIDFmatrix[index_obama], TFIDFmatrix, metric='euclidean').ravel()
wiki['BO-eucl-TF-IDF'] = dist
top10 = wiki.sort_values('BO-eucl-TF-IDF').head(10)
top10
#b) Sort the words in Obama's article by their TF-IDF weights; do the same for Schiliro's article as well.
Using the **join** operation we learned earlier, compute the common words shared by Obama's and Schiliro's articles.
Sort the common words by their TF-IDF weights in Obama's document.
obama_tf_idf = top_words_tf_idf('Barack Obama')
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
obama_indexed = obama_tf_idf.set_index('word')
schiliro_indexed = schiliro_tf_idf.set_index('word')
common_words = obama_indexed.join(schiliro_indexed, how='inner', lsuffix='_Obama', rsuffix='_Schiliro')
common_words_sorted = common_words.sort_values(by='tf-idf_Obama', ascending=False)
common_words_sorted.head(10)
#c)Among the words that appear in both Barack Obama and Phil Schiliro, take the 15 that have largest weights in Obama. How many of the articles in the Wikipedia dataset contain all of those 15 words? Which are they?
top_words_15 = common_words_sorted.head(15).index.tolist()
mask = wiki['text'].apply(contains_all_words)
articles = wiki.loc[mask, 'name'].tolist()
print("Number of articles:", len(articles))
print("Names:", articles)
#a)Compute the Euclidean distance between TF-IDF features of Obama and Biden.
index_obama = wiki[wiki['name'] == 'Barack Obama'].index[0]
index_biden = wiki[wiki['name'] == 'Joe Biden'].index[0]
dist= pairwise_distances(
    TFIDFmatrix[index_obama],
    TFIDFmatrix[index_biden],
    metric='euclidean'
).ravel()[0]
print(dist_obama_biden)
wiki.sort_values(by='BO-eucl-TF-IDF',ascending=True)[['name','BO-eucl-TF-IDF']][0:10]
#b) Let us compute the length of each Wikipedia document, and examine the document lengths for the 100 nearest neighbors to Obama's page. To compute text length use the same splitting rules you used in `vectorizer`.
token_pattern = r"(?u)\b\w+\b"
tokenizer = re.compile(token_pattern).findall
wiki['length'] = wiki.apply(compute_length, axis=1)
wiki[['name', 'length']].head()
wiki[['name', 'length']].sort_values(by='length', ascending=True).head(100)
index_obama = wiki[wiki['name'] == 'Barack Obama'].index[0]
X = vectorizer.fit_transform(wiki['text'])
distances = pairwise_distances(X[index_obama], X, metric='euclidean').ravel()
sorted_indices = np.argsort(distances)
nearest_indices = sorted_indices[1:101]
nearest_neighbors_euclidean = wiki.iloc[nearest_indices][['name', 'length']].copy()
nearest_neighbors_euclidean['distance'] = distances[nearest_indices]
nearest_neighbors_euclidean.head()
plt.figure(figsize=(10.5, 4.5))
plt.hist(nearest_neighbors_euclidean['length'], bins=20, alpha=0.6, label="Obama 100 NN")
plt.hist(wiki['length'], bins=50, alpha=0.4, label="All articles")
plt.axvline(nearest_neighbors_euclidean['length'].mean(), color='red', linestyle='dashed', linewidth=2, label="Mean NN length")
plt.xlim(150, 700)
plt.xlabel("Document length (words)")
plt.ylabel("Number of documents")
plt.title("Comparison of document lengths")
plt.legend()
plt.tight_layout()
plt.show()
#c)To see how these document lengths compare to the lengths of other documents in the corpus, make a histogram of the document lengths of Obama's 100 nearest neighbors and compare to a histogram of document lengths for all documents.
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], bins=50, alpha=0.3, color='black', label="Entire Wikipedia")
plt.hist(nearest_neighbors_euclidean['length'], bins=20, alpha=0.6, color='red', label="100 NNs of Obama (Euclidean)")
plt.hist(nearest_neighbors_cosine['length'], bins=20, alpha=0.6, color='blue', label="100 NNs of Obama (Cosine)")
plt.axvline(wiki.loc[wiki['name'] == 'Barack Obama', 'length'].values[0], color='black', linestyle='--', linewidth=2, label="Length of Barack Obama")
plt.axvline(wiki.loc[wiki['name'] == 'Joe Biden', 'length'].values[0], color='green', linestyle='--', linewidth=2, label="Length of Joe Biden")
plt.xlim(150,1000)
plt.xlabel("Document length (words)")
plt.ylabel("Number of articles")
plt.title("Document lengths comparison: Obama nearest neighbors vs Entire Wikipedia")
plt.legend()
plt.tight_layout()
plt.show()
#d) Train a new nearest neighbor model, this time with cosine distances.  Then repeat the search for Obama's 100 nearest neighbors and make a plot to better visualize the effect of having used cosine distance in place of Euclidean on our TF-IDF vectors.
distances_cosine = cosine_distances(TFIDFmatrix[index_obama], TFIDFmatrix).ravel()
sorted_indices_cosine = np.argsort(distances_cosine)
nearest_indices_cosine = sorted_indices_cosine[1:101]
nearest_neighbors_cosine = wiki.iloc[nearest_indices_cosine][['name', 'length']].copy()
nearest_neighbors_cosine['distance_cosine'] = distances_cosine[nearest_indices_cosine]
nearest_neighbors_cosine.head()
plt.figure(figsize=(10.5,4.5))
plt.hist(wiki['length'], bins=50, alpha=0.3, color='black', label="Entire Wikipedia")
plt.hist(nearest_neighbors_euclidean['length'], bins=20, alpha=0.6, color='red', label="100 NNs of Obama (Euclidean)")
plt.hist(nearest_neighbors_cosine['length'], bins=20, alpha=0.6, color='blue', label="100 NNs of Obama (Cosine)")
plt.axvline(wiki.loc[wiki['name'] == 'Barack Obama', 'length'].values[0], color='black', linestyle='--', linewidth=2, label="Length of Barack Obama")
plt.axvline(wiki.loc[wiki['name'] == 'Joe Biden', 'length'].values[0], color='green', linestyle='--', linewidth=2, label="Length of Joe Biden")
plt.xlabel("Document length (words)")
plt.xlim(150,1000)
plt.ylabel("Number of articles")
plt.title("Document lengths comparison: Obama nearest neighbors vs Entire Wikipedia")
plt.legend()
plt.tight_layout()
plt.show()
#a) Transform the tweet into TF-IDF features, using the fit to the Wikipedia dataset. (That is, let's treat this tweet as an article in our Wikipedia dataset and see what happens.) How similar is this tweet to Barack Obama's Wikipedia article?
df = pd.DataFrame({'text': ['democratic governments control law in response to popular act']})
tweet = df.loc[0, "text"]
word_counts = Counter(tweet.split())
vectorizer = TfidfVectorizer(
    norm=None,
    sublinear_tf=True,
    smooth_idf=True
)
TFIDFmatrix = vectorizer.fit_transform(wiki["text"])
tweet_tfidf = vectorizer.transform([tweet]).toarray()[0]
words = vectorizer.get_feature_names_out()
rows = []
for w in word_counts:
    if w in words:
        idx = list(words).index(w)
        rows.append({
            "word": w,
            "word_count": word_counts[w],
            "tf_idf": tweet_tfidf[idx]
        })
result = pd.DataFrame(rows)
print(result)
#b) Now, compute the cosine distance between the Barack Obama article and this tweet:
tweet_vector = vectorizer.transform([tweet])
index_obama = wiki[wiki["name"] == "Barack Obama"].index[0]
obama_vector = TFIDFmatrix[index_obama]
distance = cosine_distances(tweet_vector, obama_vector)[0][0]
print("Cosine distance between the tweet and Barack Obama's article:")
print(distance)
nearest_neighbors_cosine[0:23]

