import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

import nltk
from nltk.stem.lancaster import LancasterStemmer

from stopwords import worte


@st.cache(suppress_st_warning=True)
def download_punkt():
    nltk.download("punkt")


@st.cache(suppress_st_warning=True)
def load_data_from_json():
    # st.write("Loading data from json")
    with open("chabodoc/intents.json") as file:
        data = json.load(file)
    return data


@st.cache(suppress_st_warning=True)
def prepare_data(STEMMER, data):
    # st.write("Prepare data")
    words = []  # Wörter, die der Chatbot erkennen können soll
    labels = []  # zugehörige Labels (siehe Output unten)
    docs_x = []  # Trainingsgedöhns
    docs_y = []

    # Durchlaufe die Intents
    for intent in data["intents"]:
        # Speichere Pattern-Token (gekürzte Wörter) mit zugehörigen Labeln
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [
        w for w in words if not w in worte
    ]  # Schmeiße Stopwords raus (sowas wie "als" oder "habe"), die irrelevant für die Klassifizierung sind
    words = [STEMMER.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    return words, labels, docs_x, docs_y


def bagofwords(STEMMER, s, words):
    # Input: Satz s (User-Input), Liste bekannter Wörter words
    # Output: Vektor mit Nullen und Einsen
    bag = [0 for _ in range((len(words)))]
    s_words_tokenize = nltk.word_tokenize(
        s
    )  # Splitte Satz auf in einzelne Wörter und Satzzeichen
    s_words = [
        STEMMER.stem(word.lower()) for word in s_words_tokenize
    ]  # "Kürze" Wörter gemäß Lancaster-Stemmer
    # Wenn Wort in Wortliste enthalten, setze 1, sonst 0
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return (
        s_words_tokenize,
        s_words,
        torch.tensor(bag).float(),
    )


def app():
    st.markdown("## 3. Bag of Words")
    st.write("""Hier kann die Bag-of-Words-Funktion getestet werden.""")

    download_punkt()
    STEMMER = LancasterStemmer()

    data = load_data_from_json()
    words, *_ = prepare_data(STEMMER, data)

    with st.form("bag_of_words_form", clear_on_submit=True):
        input_sentence = st.text_area("Nutzereingabe:", value="Hi Melinda, wie geht es dir heute?", key="input_sentence")
        submit = st.form_submit_button(label="Los")

    st.markdown("---")

    if submit:
        message = input_sentence.lower()
        s_words_tokenize, s_words, bagofwords_output = bagofwords(
            STEMMER, message, words
        )

        indices_tensor = torch.arange(len(words))[bagofwords_output == 1.0]
        indices_words = []
        for idx in indices_tensor:
            indices_words.append(words[idx])

        st.markdown("**Eingabe: **")
        st.code(input_sentence)
        st.markdown("---")
        st.markdown("**Token: **")
        st.code(s_words_tokenize)
        st.markdown("---")
        st.markdown("**Wortstamm: **")
        st.code(s_words)
        st.markdown("---")
        st.markdown("**Indizes (Wert 1): **")
        st.code(indices_tensor)
        st.markdown("---")
        st.markdown("**Wörter an den Indizes mit Wert 1: **")
        st.code(indices_words)

