import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


movie = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

##print(movie.head(1),movie.columns,movie.info(),credit.head(1),credit.columns,credit.info())
##print(credit.head(1)['cast'].values)

##print(f"Movies Data Shape: {movie.shape} \nCredits Data Shape: {credit.shape}")

dt = movie.merge(credit, on = 'title')
##print(dt.shape,dt.columns)


dt = dt[['id','title','genres','overview','keywords','cast', 'crew']]
##print(f"GENRE: \n{dt.head(1)['genres'].values}\nOVERVIEW : \n{dt.head(1)['overview'].values}\KEYWORDS : \n{dt.head(1)['keywords'].values},\nCAST :  \n{dt.head(1)['cast'].values},\nCREW :  \n{dt.head(1)['crew'].values}")
##print(dt.info())
##
## 0   id        4809 non-null   int64 
## 1   title     4809 non-null   object
## 2   genres    4809 non-null   object
## 3   overview  4806 non-null   object
## 4   keywords  4809 non-null   object
## 5   cast      4809 non-null   object
## 6   crew      4809 non-null   object

#Checking shape and total isnull before dropping:
##print(dt.shape)
##print(dt.isnull().sum())

#Dropping:
dt.dropna(inplace = True)

#Checking shape and total isnull after dropping:
##print(f"SHAPE: {dt.shape}\nISNULL CHECK: \n{dt.isnull().sum()}")

#Checking for total number of duplicate values in each column:
##print(dt.duplicated().sum())

#print(type(dt),type(dt.head(1)['genres'].values),type(dt.iloc[0].genres))


def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

dt['genres'] = dt['genres'].apply(convert)
dt['keywords'] = dt['keywords'].apply(convert)

##print(f"GENRE: \n{dt.head(1)['genres'].values}\nOVERVIEW : \n{dt.head(1)['overview'].values}\nKEYWORDS : \n{dt.head(1)['keywords'].values},\nCAST :  \n{dt.head(1)['cast'].values},\nCREW :  \n{dt.head(1)['crew'].values}")

#print(dt.iloc[0]['cast'])


def convert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l


dt['cast'] = dt['cast'].apply(convert3)
##print(dt.iloc[0]['cast'])

#print(dt.iloc[0]['crew'])



def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l
            
                

dt['crew'] = dt['crew'].apply(fetch_director)
##print(dt.iloc[0]['crew'])


##print(f"GENRE: \n{dt.head(1)['genres'].values}\nOVERVIEW : \n{dt.head(1)['overview'].values}\nKEYWORDS : \n{dt.head(1)['keywords'].values},\nCAST :  \n{dt.head(1)['cast'].values},\nCREW :  \n{dt.head(1)['crew'].values}")

dt['overview'] = dt['overview'].apply(lambda x:x.split())
##print(dt.iloc[0]['overview'])

dt['genres'] = dt['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
##print(dt.iloc[0]['genres'])


dt['keywords'] = dt['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
dt['cast'] = dt['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
dt['crew'] = dt['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
##print(f"OVERVIEW : \n{dt.head(1)['overview'].values}\nKEYWORDS : \n{dt.head(1)['keywords'].values},\nCAST :  \n{dt.head(1)['cast'].values},\nCREW :  \n{dt.head(1)['crew'].values}")

dt['tags'] = dt['overview']+dt['genres']+dt['keywords']+dt['cast']+dt['crew']
#print(dt.columns)
data = dt.copy()
data = data[['id','title','tags']]
#print(data.iloc[0]['tags'])

data['tags'] = data['tags'].apply(lambda x:" ".join(x))
#print(data.iloc[0]['tags'])

data['tags'] = data['tags'].apply(lambda x:x.lower())
#print(data.iloc[0]['tags'])

ps = PorterStemmer() #Initializing

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

data['tags'] = data['tags'].apply(stem)
#print(data['tags'][499])

cv = CountVectorizer(max_features = 5000, stop_words = 'english')    #initializing
##vectors = cv.fit_transform(data['tags']).toarray().shape     #to check shape as number of movies, number of words.
vectors = cv.fit_transform(data['tags']).toarray()

#print(cv.get_feature_names_out())    # to see top 5000 most similar/repeated words
#print(data['tags'][499])

similarity = cosine_similarity(vectors)
##print(similarity[0])

def recommendation(movie):
    movie_index = data[data['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_recommend = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_recommend:
        print(data.iloc[i[0]].title)
        #print(i)
    


##print(data[data['title']=='Avatar'].index[0])  ### getting index of recommended movie for recommendation (helper function)
##print(sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6])  # ----- using to sort similarity vector with holding its index place and get top 5
#print(data.iloc[1216].title)# --- using to get title name for top 5 movies name
#recommendation('Hulk')

pickle.dump(data,open('movies.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb')) 


















