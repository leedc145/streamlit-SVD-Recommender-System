# Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd

# Suppress warning message "RuntimeWarning: invalid value encountered in true_divide"
np.seterr(invalid='ignore')

# Reading dataset (MovieLens 1M movie ratings dataset: downloaded from https://grouplens.org/datasets/movielens/1m/)
@st.cache
def fetch_data():
    data = pd.io.parsers.read_csv('ratings.dat', 
                                  names=['user_id', 'movie_id', 'rating', 'time'],
                                  engine='python', delimiter='::')
    return data
data = fetch_data()

@st.cache
def fetch_movie_data():
    movie_data = pd.io.parsers.read_csv('movies.dat',
                                        names=['movie_id', 'title', 'genre'],
                                        engine='python', delimiter='::')
    return movie_data
movie_data = fetch_movie_data()

@st.cache
def compute_USV():
    # Creating the rating matrix (rows as movies, columns as users)
    ratings_mat = np.ndarray(shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
                             dtype=np.uint8)

    ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

    # Normalizing the matrix(subtract mean off)
    normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

    # Computing the Singular Value Decomposition (SVD)
    A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
    U, S, V = np.linalg.svd(A)
    return U, S, V
U, S, V = compute_USV()

# Function to calculate the cosine similarity (sorting by most similar and returning the top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie id starts from 1 in the dataset
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

# Function to print top N similar movies
def print_similar_movies(movie_data, movie_id, top_indexes):
    for id in top_indexes + 1:
        movie_list.append(movie_data[movie_data.movie_id == id].title.values[0])

# Display title and subheader
st.title('Movies Recommender System')
st.subheader('This app shows most similar movies based on selected movie.')

# Let user choose what similar movie to display for recommendations
option_movie_title = st.sidebar.selectbox('Please choose a movie: ', 
                                  list(movie_data.title))
# Let user choose how many recommended movies to display
option_display = st.sidebar.selectbox('Please choose how many recommended movies to display: ', 
                              [10, 20, 30, 40, 50]) #default index=0

st.write('Below shows the recommended movies for:', option_movie_title)

# Find the movie ID of the selected option
option_movie_id = movie_data[movie_data.title == option_movie_title].movie_id.values[0]

# k-principal components to represent movies, movie_id to find recommendations, top_n print n results        
k = 50
sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, option_movie_id, option_display)

#Printing the top N similar movies
movie_list = []
print_similar_movies(movie_data, option_movie_id, indexes)

# Create rank from 1 until max option value
rank = []
for i in range(option_display):
    rank.append(i+1)

# Final dataframe of recommended movies output
result = pd.DataFrame(list(zip(rank, movie_list)), columns =['Rank', 'Movie Title']).set_index('Rank')

# Display to streamlit
st.dataframe(result, 1000, 1000)
