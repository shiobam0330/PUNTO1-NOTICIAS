import streamlit as st
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import spacy
import language_tool_python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import FastText
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import gensim
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer,WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

path = os.path.join(os.getcwd(), "PUNTO 1 NEWS")

st.title("Clasificacion de noticias")

@st.cache_data
def models():
    import pickle
    with open("kmeans_bert.pkl", "rb") as file:
        kmeans = pickle.load(file)
    with open("pca_fast.pkl", "rb") as file:
        pca = pickle.load(file)
    with open("scaler_fast.pkl", "rb") as file:
        scaler = pickle.load(file)
    with open("km_fast.pkl", "rb") as file:
        km_fast = pickle.load(file)
    modelo_ft = gensim.models.fasttext.load_facebook_vectors(path+'modelo_fasttext.bin')
    lda_model = gensim.models.LdaModel.load(path + "modelo_lda_wo.bin")
    lda_model_tfidf = gensim.models.LdaModel.load(path + "modelo_lda_tfidf.bin")
    return kmeans, pca, scaler, km_fast, modelo_ft ,lda_model,lda_model_tfidf

def inic():
    tool = language_tool_python.LanguageTool("en")
    nlp = spacy.load("en_core_web_lg")
    stop_nltk = stopwords.words('english')
    stop_words = set(stopwords.words('english'))
    stop_spacy = nlp.Defaults.stop_words
    stop_todas = list(stop_spacy.union(set(stop_nltk)))
    dictionary = gensim.corpora.Dictionary.load("dictionary.dict")
    return tool, nlp, stop_todas, dictionary, stop_words

def proce_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

@st.cache_data
def get_bert_embedding(textos, tokenizer, model):
    inputs = tokenizer(textos, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings


def procesamiento(noticia):
    nuevos = pd.DataFrame({"headline_text": [noticia]})
    nuevos["matches"] = nuevos["headline_text"].apply(lambda x: tool.check(x))
    nuevos["corrected"] = nuevos.apply(lambda c: language_tool_python.utils.correct(c["headline_text"], c["matches"]), axis=1)

    # Procesar en lotes usando nlp.pipe() de SpaCy
    tokens = [doc for doc in nlp.pipe(nuevos['corrected'], batch_size=1000)]
    nuevos['tokens'] = [[token.text for token in doc] for doc in tokens]

    nuevos['processed_text'] = nuevos.apply(lambda row: ' '.join(token.lemma_ for token in nlp(row["corrected"]).sents), axis=1)
    nuevos['processed_text'] = nuevos['processed_text'].str.lower()
    nuevos['processed_text'] = nuevos['processed_text'].replace(list('áéíóú'), list('aeiou'), regex=True)
    nuevos['processed_text'] = nuevos['processed_text'].str.replace('[^\w\s]', '', regex=True)
    nuevos['processed_text'] = nuevos['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_todas]))

    processed_nuevo = nuevos['processed_text'].map(preprocesamiento)
    bow_corpus_new = [dictionary.doc2bow(doc) for doc in processed_nuevo]

    return nuevos, bow_corpus_new


def topicos(individ, datos):
    topics = []
    for y in range(datos.shape[0]):
        if individ[y]:
            valid_sublist = [sublist for sublist in individ[y] if len(sublist) > 1]
            if valid_sublist:
                max_index = np.argmax([sublist[1] for sublist in valid_sublist])
                topics.append(valid_sublist[max_index][0])
            else:
                topics.append(None)  
        else:
            topics.append(None) 

    return topics

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.0

    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector = np.add(feature_vector, model[word])  # Acceso directo a las palabras
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector

def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.key_to_index)  # Corregido: usando 'key_to_index' en FastText
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]

    return np.array(features)


def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocesamiento(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result
dictionary = gensim.corpora.Dictionary.load(path+"dictionary.dict")

kmeans, pca, scaler, km_fast, modelo_ft ,lda_model,lda_model_tfidf = models()
tool, nlp, stop_todas, dictionary, stop_words = inic()
tokenizer, model = proce_bert()
nuevos = None

noticias_procesadas = set()

entrada = st.text_area("Ingrese una noticia en inglés.", value="Tropical Storm Sara causes heavy rain in parts of Honduras")
modelo_cla = st.selectbox("Seleccione la forma de clasificar su noticia", ["", "Modelo Bert", "Modelo Fast Text", "LDA sin TF-IDF", "LDA con TF-IDF"])

metodos_dict = {
    "LDA sin TF-IDF": {
        "modelo": lda_model,
        "datos": None, 
        "temas": ["Politicas Publicas", "Internacional", "Climas Extremos"]
    },
    "LDA con TF-IDF": {
        "modelo": lda_model_tfidf,
        "datos": None, 
        "temas": ["Internacional", "Politicas Publicas", "Climas Extremos"]
    },
    "Modelo Bert": {
        "modelo": kmeans,
        "datos": None,
        "temas": ["Politicas Publicas", "Internacional", "Climas Extremos"]
    },
    "Modelo Fast Text": {
        "modelo": km_fast,
        "datos": None, 
        "temas": ["Politicas Publicas", "Climas Extremos", "Internacional"]
    }
}

if modelo_cla == "":
    st.warning("Por favor, selecciona un método válido.")
else:
    if st.button("Clasificar noticia"):
        if entrada not in noticias_procesadas:
            nuevos, bow_corpus_new = procesamiento(entrada)
            noticias_procesadas.add(entrada)
            
            if modelo_cla == "Modelo Bert":
                nuevos['bert_embedding'] = nuevos['processed_text'].apply(lambda x: get_bert_embedding(x, tokenizer, model))
                metodos_dict["Modelo Bert"]["datos"] = np.stack(nuevos['bert_embedding'].values) 
            elif modelo_cla == "Modelo Fast Text":
                input_data = averaged_word_vectorizer(corpus=nuevos['tokens'], model=modelo_ft, num_features=modelo_ft.vector_size)
                metodos_dict["Modelo Fast Text"]["datos"] = input_data 

            metodos_dict["LDA sin TF-IDF"]["datos"] = bow_corpus_new
            metodos_dict["LDA con TF-IDF"]["datos"] = bow_corpus_new

        else:
            st.warning("Esta noticia ya ha sido procesada.")

        if modelo_cla in metodos_dict:
            model = metodos_dict[modelo_cla]["modelo"]
            input_data = metodos_dict[modelo_cla]["datos"]
            temas_lista = metodos_dict[modelo_cla]["temas"]
            
            if modelo_cla == "LDA sin TF-IDF":
                ind_nuevo = lda_model[input_data]
                topics = topicos(ind_nuevo, nuevos)
                resultado = temas_lista[topics[0]]
                st.success(f"El tema de la noticia es: {resultado}")

            elif modelo_cla == "LDA con TF-IDF":
                ind_tfidf_nuevo = lda_model_tfidf[input_data]
                topics_tfidf = topicos(ind_tfidf_nuevo, nuevos)
                resultado = temas_lista[topics_tfidf[0]]
                st.success(f"El tema de la noticia es: {resultado}")

            elif modelo_cla == "Modelo Bert":
                embeddings_nuevo = np.stack(nuevos['bert_embedding'].values)
                predicted_class = kmeans.predict(embeddings_nuevo)[0]
                resultado = temas_lista[predicted_class]
                st.success(f"El tema de la noticia es: {resultado}")

            elif modelo_cla == "Modelo Fast Text":
                ftext_nuevo = np.array(input_data)
                pcs_new = pca.transform(ftext_nuevo)
                doc_embedding_new = pd.DataFrame(pcs_new)
                scaler_new = scaler.transform(doc_embedding_new)
                fast = km_fast.predict(scaler_new)
                resultado = temas_lista[fast[0]]
                st.success(f"El tema de la noticia es: {resultado}")


