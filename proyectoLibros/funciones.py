import requests as req
from bs4 import BeautifulSoup
import numpy as np
import re
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Con esta funcion obtengo los datos de una url que recibo por parametro.
def getData(url):
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'}
    r = req.get(url,headers = headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

# Con esta funcion obtengo el link de la sieguiente pagina de reseñas.
def getNextPage(soup):
    page = soup.find('ul',{'class': 'a-pagination'})
    if not page.find('li', {'class':'a-disabled a-last'}):
        url = 'https://www.amazon.es' + str(page.find('li',{'class':'a-last'}).find('a')['href'])
        return url
    else :
        return

# Con esta funcion leemos todos los comentarios de una pagina
def getComentarios(soup):
    comentarios = []
    aux = soup.findAll('span',{'class': 'a-size-base review-text review-text-content'})
    for i in aux:
        comentarios.append(i.get_text(strip=True))
    return comentarios

def getEstrellas(soup):
    estrellas = []
    #de la pagina me quedo solo con la parte donde estan los comentarios (no nos interesa la media
    # de estrellas por ejemplo)
    aux = soup.findAll('div',{'class': 'a-section celwidget'})
    for i in aux:
        iconoEstrellas = i.find('span',{'class': 'a-icon-alt'})
        estrellas.append(re.sub(',\d de 5 estrellas','',iconoEstrellas.get_text(strip=True))) 
    return estrellas

def tokenizar(comentario):
    tokensComentario = word_tokenize(comentario)
    return tokensComentario

def crearCorpus(listadoResenias):
    stemmer = SnowballStemmer('spanish')
    for i,resenia in enumerate(listadoResenias):
        tokensResenia = tokenizar(resenia)
        reseniaStemmizada = ""
        for token in tokensResenia:
            tokenStemmizado = stemmer.stem(token)
            reseniaStemmizada += " "+tokenStemmizado
        listadoResenias[i] = reseniaStemmizada
    return listadoResenias

# Con esta funcion leemos todas las reseñas del libro de la url.
def leerLibro(url):
    soup = getData(url)
    todosLosComentarios = []
    todasLasEstrellas = []
    while (True):
        soup = getData(url)
        todosLosComentarios.append(getComentarios(soup))
        todasLasEstrellas.append(getEstrellas(soup))
        url = getNextPage(soup)
        if not url:
            break
    return todosLosComentarios,todasLasEstrellas

def preprocesar(X_train):
    #quito todo lo que no sean letras, incluyendo con tildes
    #no funciona bien
    for i,resenia in enumerate(X_train):
        resenia = re.sub(r'[^\w\s]','',resenia)
        X_train[i] = re.sub(r'\d','',resenia)
    return X_train