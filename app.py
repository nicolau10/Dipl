import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from ast import literal_eval

st.title("Система рекомендаций фильмов")
st.image("economizar-cinema.jpg")


st.subheader("Фильм какого жанра вы хотите посмотреть?")
lis = st.selectbox("Жанр", ['Action', 'Science Fiction', 'Drama', 'Adventure', 'History', 'Crime',
       'Thrillers', 'Comedy', 'Documentary'])


st.subheader("Фильм какого года вы хотите посмотреть")  #Year
year = st.slider("Год", 1984, 2001)

st.subheader("Выберите рейтинг фильма")  #Year
rate = st.slider("Рейтинг", 0, 7)

films = pd.read_csv('datasetedited.csv')
films.pop('Unnamed: 0')
films = films.sort_values(by = 'id')
rates = pd.read_csv('ratings_small.csv')
rates = rates.rename(columns = {'movieId':'id'})
rates.pop('timestamp')

combined = pd.merge(rates, films, on='id')

rates = rates[rates.id.isin(films['id'])]

ans = combined.loc[(combined.genres == lis) & (combined.vote_average >= rate) & (combined.year >= year),['title','vote_average','year']]

names = ans['title'].tolist()
x = np.array(names)
ans1 = np.unique(x)

finallist = ""

bruh = st.checkbox("Выберите фильм")
if bruh == True:
    finallist = st.selectbox('Фильм',ans1)

##### IMPLEMENTING RECOMMENDER ######
dataset = rates.pivot_table(index='id', columns = 'userId', values='rating')
dataset.fillna(0,inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)
dataset.sort_values(by = 'id')

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Film_Name):
    n = 10
    FilmList = films[films['title'].str.contains(Film_Name)] 
    FilmList = FilmList.sort_values(by = 'id')  
    
    if len(FilmList):        
        Filmi= FilmList.iloc[0]['id']
        Filmi = dataset[dataset['id'] == Filmi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Filmi],n_neighbors=n+1)

    
        
         
        Film_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]

        
        Recommendations = []
        for val in Film_indices:
            Filmi = dataset.iloc[val[0]]['id']
            #print(Filmi)
            i = films[films['id'] == Filmi].index
            #print(films[films['id'] == Filmi])
            Recommendations.append({'title':films.iloc[i]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['title']
    else:
        return "Нет похожих фильмов."


display = food_recommendation(finallist)

if bruh == True:
    bruh1 = st.checkbox("Новые рекомендации: ")
    if bruh1 == True:
        for i in display:
            st.write(i)