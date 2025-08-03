# importing basic modules required
import re
import ast
import pickle
import numpy as np
import pandas as pd
import streamlit as st  

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings ('ignore')

#load datasets
movies= pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')

#merge columns
movies= movies.merge(credits, on='title')

#best features for rec - movie_id, title, overview, genres, keywords, cast, crew
df= movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def fetch_genres(text):
    genres= []
    for i in ast.literal_eval(text):
        genres.append(i['name'])
    return genres
df['genres']= df['genres'].apply(fetch_genres)

def fetch_keywords(text):
    keywords= []
    for i in ast.literal_eval(text):
        keywords.append(i['name'])
    return keywords
df['keywords']= df['keywords'].apply(fetch_keywords)


def fetch_cast(text):
    cast= []
    counter= 0
    for i in ast.literal_eval(text):
        if counter != 3:
            cast.append(i['name'])
            counter += 1
        else:
            break
    return cast
df['cast']= df['cast'].apply(fetch_cast)

def fetch_director(text):
    director= []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            director.append(i['name'])
    return director
df['crew']= df['crew'].apply(fetch_director)


df['overview'].fillna(' ', inplace=True)
df['overview'] = df['overview'].apply(lambda x: x.split())

df['tags']= df['overview']+ df['genres']+ df['keywords']+ df['cast']+ df['crew']

data= df[['movie_id', 'title', 'tags']]
data['tags']= data['tags'].apply(lambda x: [i.replace(' ','') for i in x])
data['tags']= data['tags'].apply(lambda x: " ".join(x))

#NLP FOR TEXT PREPROCESSING - tokenization, lower case, stemming and lemmetization, stopwords removal, vector encoding
ps= PorterStemmer()
def text_preprocessing(text):
    cleaned_text = []
    for i in text.split():
        lower_data= i.lower()
        stem_text = ps.stem(lower_data)
        cleaned_text.append(stem_text)
    return " ".join(cleaned_text)

data['tags']= data['tags'].apply(text_preprocessing)

#BOW encoding text to numbers
cv= CountVectorizer(max_features=5000, stop_words='english')
cv.fit_transform(data['tags']).toarray()
vectors= cv.fit_transform(data['tags']).toarray()

#cosine similarity
similarity= cosine_similarity(vectors)

#sorting 
#print(sorted(enumerate(similarity[64]), reverse=True, key=lambda x:x[1])[1:6])

#final recommend system 
def recommend(movie):
    
    recommended_movies=[]
    
    movie_index =data[data['title'] == movie].index[0]
    distance= similarity[movie_index]
    movie_list = sorted(enumerate(distance), reverse=True, key= lambda x: x[1])[1:6]
    
    for i in movie_list:
        recommended_movies.append(data.iloc[i[0]].title)
        
    return recommended_movies


#print (recommend('Avatar'))

#streamlit web-app
st.title(':blue[Movie Recommendation System]')

#selectbox
selected_movie = st.selectbox("Please select a movie to get similar recommendations:", list(data['title'].values))

btn= st.button('Recommend')

if btn:
    top5movie=recommend(selected_movie)
    for i in top5movie:
        st.write(i)
        
