import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

#import json

import nltk
from nltk.stem.lancaster import LancasterStemmer

import pandas as pd

#from utils import Classifier
#import matplotlib

PATHDATA = "./chabodoc/data/"
PATHSTOP = PATHDATA+"stopwords/"
PATHWORDS = PATHDATA+"words/"
PATHNET = PATHDATA+"networks/"


@st.cache(suppress_st_warning=True)
def download_punkt():
    nltk.download("punkt")

class Classifier(nn.Module):
    def __init__(self, dims=[]):
        super().__init__()
        layerlist = []
        for i in range(len(dims) - 1):
            layerlist.append(nn.Linear(dims[i], dims[i + 1]))
            layerlist.append(nn.ReLU())
        self.layers = nn.Sequential(*(layerlist[:-1]))

    def forward(self, x):
        out = self.layers(x)
        return out


def bagofwords(s, words, stopwords):
    # Input: Satz s (User-Input), Liste bekannter Wörter words
    # Output: Vektor mit Nullen und Einsen
    STEMMER = LancasterStemmer()
    bag = [0 for _ in range((len(words)))]
    #print("baglen", len(bag))
    s_words_tokenize = nltk.word_tokenize(
        s
    )  # Splitte Satz auf in einzelne Wörter und Satzzeichen
    s_words = [
        STEMMER.stem(word.lower())
        for word in s_words_tokenize
        if word.lower() not in stopwords
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


class Testgruppe:
    def __init__(self, name, suffix=""):
        self.name = name
        self.suffix = suffix

        self.words = ["gut", "schlecht", "neutral"]
        self.stopwords = ["ich", "mir", "geht", "es"]

        #print(len(self.words))
        self.network = Classifier(dims=[len(self.words), int(len(self.words) / 2), 3])

    def predict(self, input):
        *_, bag = bagofwords(input.lower(), self.words, self.stopwords)
        result = F.softmax(self.network(bag), dim=-1)
        return result


class Gruppe():
    def __init__(self, name, suffix=""):
        self.name = name
        self.suffix = suffix

        with open(
            PATHWORDS + self.name + self.suffix + "_words.txt", "r", encoding="utf-8"
        ) as file:
            self.words = [w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"]

        with open(PATHSTOP + "stopwords.txt", "r", encoding="utf-8") as file:
            self.stopwords = [w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"]

        with open(
            PATHSTOP + self.name + self.suffix + "_stop.txt", "r", encoding="utf-8"
        ) as file:
            self.stopwords.extend([w.replace("\n", "").strip() for w in file.readlines() if w != "" or w != "\n"])

        #print(len(self.words))
        """for w in self.words:
            if w.lower() in self.stopwords:
                print(w)
        self.words = [w.lower() for w in self.words if not w.lower() in self.stopwords]"""
        #print(len(self.words))
        #self.network = torch.load(PATHNET + self.name + self.suffix + "_model.pt")
        self.network = Classifier(dims=[len(self.words), int(len(self.words) / 2), 3])
        self.network.load_state_dict(
            torch.load(PATHNET + self.name + self.suffix + ".pt", map_location=torch.device("cpu"))
        )

    def predict(self, input):
        *_, bag = bagofwords(input.lower(), self.words, self.stopwords)
        result = F.softmax(self.network(bag), dim=-1)
        return result


def app():
    st.markdown("## 5. Vergleich der ChatBots")

    st.markdown(
        "Hier können die ChatBots verschiedener Gruppen geladen und getestet werden."
    )

    st.markdown("---")

    # TODO read group names from some file or anything similar
    group_list_dropdown = ["Melinda", "Salzwerk", "Gruppe", "MarzInator", "LuSo", "Supernet"]

    chatbot_option = st.selectbox(
        "ChatBot Auswahl",
        group_list_dropdown,
    )

    # TODO take Gruppe(...) instead of Testgruppe
    print("Loading", chatbot_option)
    current_group = Gruppe(chatbot_option)

    table_current_group_input = chatbot_option + "user_table_entries"
    table_current_group_good = chatbot_option + "good_table_entries"
    table_current_group_bad = chatbot_option + "bad_table_entries"
    table_current_group_neutral = chatbot_option + "neutral_table_entries"
    if table_current_group_input not in st.session_state:
        st.write("Group not in session state")
        st.session_state[table_current_group_input] = []
        st.session_state[table_current_group_good] = []
        st.session_state[table_current_group_bad] = []
        st.session_state[table_current_group_neutral] = []

    st.markdown("---")

    st.markdown("Chatbot ("+current_group.name+"): Wie geht es dir heute?")

    with st.form("user_input", clear_on_submit=True):
        user_input = st.text_input("Nutzer:", key="input_sentence")
        submit = st.form_submit_button(label="Senden")

    st.markdown("---")

    if submit:
        result = current_group.predict(user_input)

        st.session_state[table_current_group_input].append(user_input)
        st.session_state[table_current_group_good].append(result[1].item())
        st.session_state[table_current_group_bad].append(result[0].item())
        st.session_state[table_current_group_neutral].append(result[2].item())

        st.markdown("Details zu Antwort")
        result_table = {
            "Nutzer": st.session_state[table_current_group_input],
            "gut": st.session_state[table_current_group_good],
            "schlecht": st.session_state[table_current_group_bad],
            "neutral": st.session_state[table_current_group_neutral],
        }

        result_table = pd.DataFrame.from_dict(result_table)

        st.table(result_table.style.background_gradient(axis=None, cmap="Blues"))

    #st.sidebar.image("./images/Logo_Uni_Luebeck_600dpi.png", use_column_width=True)
    #st.sidebar.image("./images/Logo_UKT.png", use_column_width=True)
