import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load song data
df = pd.read_csv('collage.csv')

# Combine all your features into a single string
df['combined_features'] = df['collages'] + ' ' + df['country'] + ' ' + df['best ranks']# Add more features if you have

# Create a TF-IDF vectorizer. This will convert your combined features into a matrix of TF-IDF features.
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Initialize a NearestNeighbors model. We set n_neighbors to 10 as we want to recommend 10 songs.
knn = NearestNeighbors(n_neighbors=3, metric='cosine')

# Train the model
knn.fit(tfidf_matrix)

# Let's say the user has selected songs with indices 1, 2, and 3 in the dataframe
selected_collages = [9 ,4]

# We'll store the recommended songs in this set (sets automatically remove any duplicates)
recommended_collages = set()

# For each selected song, find the 10 nearest neighbors (most similar songs) and add them to the set
for collage in selected_collages:
    distances,indices=knn.kneighbors(tfidf_matrix[collage])
    recommended_collages.update(indices.flatten())

# Remove the selected songs from the set of recommended songs
recommended_collages.difference_update(selected_collages)

# Print the recommended songs
print(df.loc[list(recommended_collages)])

#
