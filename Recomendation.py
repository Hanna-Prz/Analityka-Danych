import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
!wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip
!ls ml-latest-small

def predict_rating(user_id, movie_id):
    if movie_id not in user_movie_matrix.columns:
        return np.nan
    user_idx = user_id - 1
    movie_idx = list(user_movie_matrix.columns).index(movie_id)
    sim_scores = user_similarity[user_idx]
    movie_ratings = user_movie_matrix_filled.iloc[:, movie_idx]
    return np.dot(sim_scores, movie_ratings) / np.sum(np.abs(sim_scores))

def recommend_user_based(user_id, n=5):
    user_index = user_id - 1
    similar_users = user_similarity[user_index]
    scores = similar_users.dot(user_movie_matrix_filled.values)
    scores = scores / np.array([np.abs(similar_users).sum()])
    recommendations = pd.Series(scores.flatten(), index=user_movie_matrix.columns)
    already_seen = user_movie_matrix.loc[user_id].dropna().index
    recommendations = recommendations.drop(already_seen)
    top_movies = recommendations.sort_values(ascending=False).head(n)
    return movies[movies['movieId'].isin(top_movies.index)]

def recommend_item_based(movie_id, n=5):
    idx = list(user_movie_matrix.columns).index(movie_id)
    similar_scores = item_similarity[idx]
    similar_movies = pd.Series(similar_scores, index=user_movie_matrix.columns)
    top_movies = similar_movies.sort_values(ascending=False).iloc[1:n+1]
    return movies[movies['movieId'].isin(top_movies.index)]

def recommend_by_genre(genre, n=10):
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    genre_movies = genre_movies.merge(movie_stats, on='title')
    return genre_movies.sort_values(
        by=['mean_rating', 'rating_count'],
        ascending=False
    ).head(n)

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")
print("Ratings:")
display(ratings.head())
print("\nMovies:")
display(movies.head())
print("Number of ratings:", len(ratings))
print("Number of useres:", ratings['userId'].nunique())
print("Number of movies:", ratings['movieId'].nunique())
display(ratings['rating'].describe())
plt.figure(figsize=(10,5))
sns.countplot(x='rating', data=ratings)
plt.title("Distrybution")
plt.xlabel("Ratings")
plt.ylabel("Number")
plt.show()
movie_counts = ratings['movieId'].value_counts().head(20)
plt.figure(figsize=(10,6))
movie_counts.plot(kind='bar')
plt.title("20 the most popular movies to mark")
plt.ylabel("Numbrer of marks")
plt.show()
data = ratings.merge(movies, on="movieId")
display(data.head())
movie_stats = data.groupby("title").agg(
    mean_rating=('rating', 'mean'),
    rating_count=('rating', 'count')
).sort_values(by='rating_count', ascending=False)
display(movie_stats.head(10))
plt.figure(figsize=(10,6))
plt.scatter(movie_stats['rating_count'], movie_stats['mean_rating'], alpha=0.3)
plt.xlabel("Number of ratings")
plt.ylabel("Mean mark")
plt.title("Mean mark vs popularity")
plt.show()
popular_recommendations = movie_stats[
    movie_stats['rating_count'] > 50
].sort_values(by='mean_rating', ascending=False)
print("TOP 10 movies (popularity):")
display(popular_recommendations.head(10))
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)
print("user-movie matrix")
print(user_movie_matrix.shape)
display(user_movie_matrix.head())
user_movie_matrix_filled = user_movie_matrix.fillna(0)
display(user_movie_matrix_filled.head())
user_similarity = cosine_similarity(user_movie_matrix_filled)
print("user similarity matrix")
print(user_similarity.shape)
plt.figure(figsize=(8,6))
sns.heatmap(user_similarity[:20, :20], cmap='coolwarm')
plt.title("User similarity (snippet)")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(user_similarity, cmap='coolwarm')
plt.title("User similarity ")
plt.show()
print("Recommendations for user 1:")
display(recommend_user_based(1, 10))
item_similarity = cosine_similarity(user_movie_matrix_filled.T)
print("Movie similarity matrix:")
print(item_similarity.shape)
plt.figure(figsize=(8,6))
sns.heatmap(item_similarity[:20, :20], cmap='viridis')
plt.title("Similarity of the films (excerpt)")
plt.show()
plt.figure(figsize=(8,6))
sns.heatmap(item_similarity, cmap='viridis')
plt.title("Similarity of the films ")
plt.show()
example_movie = ratings['movieId'].iloc[0]
print("Base film:")
display(movies[movies['movieId'] == example_movie])
print("\nSimilar movies:")
display(recommend_item_based(example_movie, 10))
svd = TruncatedSVD(n_components=20, random_state=42)
latent_matrix = svd.fit_transform(user_movie_matrix_filled)
print("Latent matrix:")
print(latent_matrix.shape)
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(svd.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance")
plt.title("SVD – explained variance")
plt.show()
latent_similarity = cosine_similarity(latent_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(latent_similarity[:20, :20], cmap='magma')
plt.title("User similarity – SVD")
plt.show()
total_cells = user_movie_matrix.shape[0] * user_movie_matrix.shape[1]
non_zero = np.count_nonzero(user_movie_matrix.values)
sparsity = 1 - (non_zero / total_cells)
print("RARENESS OF THE GRADE MATRIX")
print(f"Total number of fields: {total_cells}")
print(f"Number of ratings: {non_zero}")
print(f"Sparsity: {sparsity:.4f}")
user_activity = ratings.groupby("userId").size()
print("User activity statistics:")
display(user_activity.describe())
plt.figure(figsize=(10,5))
user_activity.hist(bins=50)
plt.title("Rozkład liczby ocen na użytkownika")
plt.xlabel("Liczba ocen")
plt.ylabel("Liczba użytkowników")
plt.show()
genres = movies['genres'].str.get_dummies('|')
print("Matrix of genres:")
display(genres.head())
genre_popularity = genres.sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
genre_popularity.plot(kind='bar')
plt.title("Popularity of film genres")
plt.ylabel("Number of movies")
plt.show()
print("Recommendations – Sci-Fi genre:")
display(recommend_by_genre("Sci-Fi", 10))
train, test = train_test_split(ratings, test_size=0.2, random_state=42)
print("Train size:", train.shape)
print("Test size:", test.shape)
predictions = []
true_values = []
for _, row in test.iterrows():
    # Cast userId and movieId to int before passing to predict_rating
    pred = predict_rating(int(row['userId']), int(row['movieId']))
    if not np.isnan(pred):
        predictions.append(pred)
        true_values.append(row['rating'])
rmse = np.sqrt(mean_squared_error(true_values, predictions))
print("RMSE (user-based):", rmse)
user_id = 10
print("Movies watched by the user:")
display(data[data['userId'] == user_id].sort_values(by='rating', ascending=False).head())
print("\nRecommendations:")
display(recommend_user_based(user_id, 10))
