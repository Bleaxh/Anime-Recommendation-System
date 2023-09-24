import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your anime dataframe
anime_df = pd.read_csv("./dataset/anime.csv")  # Replace with your actual file path

# Preprocessing
anime_df.dropna(subset=['name', 'genre'], inplace=True)  # Drop rows with missing values
anime_df['genre'] = anime_df['genre'].str.replace(",", " ")
anime_df['name'] = anime_df['name'].str.replace('.',' ')

# Check if input anime name exists in the dataframe
def anime_exists(anime_name):
    return anime_df['name'].str.strip().str.lower().eq(anime_name).any()

# Function to find similar anime based on genre
def find_similar_anime(query_anime_name, num_similar=6):
    query_anime_name = query_anime_name.strip().lower()
    similar_anime = []
    for anime_name in anime_df['name']:
        if query_anime_name in anime_name.lower():
            similar_anime.append(anime_name)

    # similar_anime = similar_anime[:3]
    
    if anime_exists(query_anime_name):
        anime_row = anime_df[anime_df['name'].str.strip().str.lower() == query_anime_name]
        anime_vector = tfidf_vectorizer.transform(anime_row['genre'])

        # Calculate cosine similarity between the anime vector and all anime vectors
        similarity_scores = cosine_similarity(anime_vector, tfidf_matrix)
        similar_indices = similarity_scores.argsort()[0][-num_similar-1:-1][::-1]

        for idx in similar_indices:
            similar_anime.append(anime_df.iloc[idx]['name'])

        print(f"Anime similar to {anime_df[anime_df['name'].str.strip().str.lower() == query_anime_name]['name'].values[0]}:")
        for idx, anime in enumerate(similar_anime, start=1):
            print(f"{idx}. {anime}")
    else:
        print("Invalid anime name.")
    
    return similar_anime[:num_similar]



# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['genre'])

# Example usage
# query_anime_name = input("Enter the anime name: ")
# 


