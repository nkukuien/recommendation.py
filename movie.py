from sklearn.cluster import KMeans

movies = [
    {"title": "The Avengers", "features": [9, 1, 8, 7]},
    {"title": "Inception", "features": [8, 2, 9, 6]},
    {"title": "The Shawshank Redemption", "features": [3, 7, 2, 9]},
    {"title": "Pulp Fiction", "features": [1, 9, 5, 3]},
    {"title": "The Matrix", "features": [7, 3, 9, 2]},
    {"title": "Interstellar", "features": [9, 8, 6, 5]},
    {"title": "Fight Club", "features": [2, 9, 3, 8]},
    {"title": "The Godfather", "features": [4, 6, 9, 1]},
    {"title": "The Dark Knight", "features": [8, 9, 2, 5]},
    {"title": "Goodfellas", "features": [3, 8, 4, 7]}
]

# Extract the feature vectors and apply k-means clustering
feature_vectors = [movie["features"] for movie in movies]
kmeans = KMeans(n_clusters=2).fit(feature_vectors)

# Now, suppose a user likes "The Shawshank Redemption". We can recommend another movie from the same cluster.
liked_movie = "The Shawshank Redemption"
liked_movie_features = next(movie["features"] for movie in movies if movie["title"] == liked_movie)
liked_movie_cluster = kmeans.predict([liked_movie_features])[0]

# Find another movie from the same cluster
recommended_movie = next(movie for movie in movies if movie["title"] != liked_movie and
                        kmeans.predict([movie["features"]])[0] == liked_movie_cluster)

print("Because you liked", liked_movie + ",", "we recommend:", recommended_movie["title"])
