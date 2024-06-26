import streamlit as st
import pickle
import requests

movies = pickle.load(open('movies.pkl', 'rb'))
movies_list = movies['title'].values

similarity = pickle.load(open('similarity.pkl', 'rb'))


def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=35551ce8318af74a8b0d9856acabfb7a'.format(movie_id),verify=False)
    data = response.json()
    if data['poster_path']:
        return "https://image.tmdb.org/t/p/original/"+data['poster_path']
    else:
        return "default.jpg"

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_recommend = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_list = []
    recommended_movies_poster =[]
    for i in movie_recommend:
        movie_id = movies.iloc[i[0]].id
        recommended_list.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_list, recommended_movies_poster


st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Which Movie recommendation you would like to check?',
    movies_list

)

if st.button('Recommended'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
