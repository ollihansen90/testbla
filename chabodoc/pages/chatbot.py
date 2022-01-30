import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

import nltk
from nltk.stem.lancaster import LancasterStemmer

from stopwords import worte
from random import choice
from pages.chat_tree import answer_tree

import os
import dropbox


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


@st.cache(suppress_st_warning=True)
def download_punkt():
    nltk.download("punkt")


@st.cache(suppress_st_warning=True)
def load_data_from_json():
    # st.write("Loading data from json")
    with open("/app/chabodoc/intents.json", encoding="utf-8") as file:
        data = json.load(file)
    return data


def bagofwords(STEMMER, s, words):
    # Input: Satz s (User-Input), Liste bekannter Wörter words
    # Output: Vektor mit Nullen und Einsen
    bag = [0 for _ in range((len(words)))]
    s_words = nltk.word_tokenize(
        s
    )  # Splitte Satz auf in einzelne Wörter und Satzzeichen
    s_words = [
        STEMMER.stem(word.lower()) for word in s_words
    ]  # "Kürze" Wörter gemäß Lancaster-Stemmer

    # Wenn Wort in Wortliste enthalten, setze 1, sonst 0
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return torch.tensor(bag).float()


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


def predict(STEMMER, message, model, words, labels, data, device):
    message = message.lower()
    result = F.softmax(model(bagofwords(STEMMER, message, words).to(device)), dim=0)
    result_index = torch.argmax(result)
    tags = labels#[result_index]
    tag = labels[result_index]

    # Wie sicher ist sich der Chatbot? 0.9 ist schon ziemlich sicher.
    if result[result_index] > 0.9:
        for tg in data["intents"]:
            if tg["tag"] == tag:
                responses = tg["responses"]
        response = choice(responses)
    else:
        # st.write("Chatbot ist sich etwas unsicher.", result[result_index].item())
        response = "Das habe ich leider nicht verstanden!"
    return tags, response, result


def app():
    st.markdown("## 2. ChatBot")

    st.markdown("""Hier kannst du mit **Melinda** chatten.""")

    st.markdown("""---""")

    download_punkt()
    STEMMER = LancasterStemmer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_data_from_json()
    words, labels, docs_x, docs_y = prepare_data(STEMMER, data)

    if "chatbot_model_trained" not in st.session_state:
        dims = [507, 253, 14]
        st.session_state["chatbot_model_trained"] = Classifier(dims).to(device)
        st.session_state["chatbot_model_trained"].load_state_dict(
            torch.load("/app/chabodoc/chatbot_model_trained.pth")
        )

    st.session_state["chatbot_model_trained"].eval()

    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []
        st.session_state["conversation"].append(
            "Melinda: Hi, ich bin Melinda! Ich freue mich, dass wir hier chatten können, und würde dir gerne ein paar Fragen stellen. Wie geht es dir gerade?"
        )
        st.session_state["tag"] = []
        st.session_state["tag"].append(" ")
        st.session_state["sicher"] = []
        st.session_state["sicher"].append(" ")

        st.session_state["tree_id"] = 1
        st.session_state["case"] = 0

        st.session_state["finished_chat"] = False

    col1, col2 = st.columns([9, 1])

    st.markdown("***")

    form_placeholder = st.empty()
    with form_placeholder.form("chat_form", clear_on_submit=True):
        placeholder = st.empty()
        user_input = placeholder.text_input("Nutzer:", key="user_input")
        submit = st.form_submit_button(label="Senden")

    if submit:
        input_string = "Nutzer: " + user_input
        st.session_state["conversation"].append(input_string)

        tags, prediction, sicherheiten = predict(
            STEMMER,
            user_input,
            st.session_state["chatbot_model_trained"],
            words,
            labels,
            data,
            device,
        )

        indices = sicherheiten.argsort(descending=True)
        print(indices)
        print(tags)
        topindex = indices[0].item()
        sicher = sicherheiten[topindex].item()
        tags = [tags[i] for i in indices]
        sicherheiten = sicherheiten[indices]
        tag = tags[0]

        (
            response,
            st.session_state["tree_id"],
            st.session_state["case"],
            st.session_state["finished_chat"],
        ) = answer_tree(
            st.session_state["tree_id"],
            st.session_state["case"],
            tag,
            prediction,
            st.session_state["finished_chat"],
        )

        response_string = "Melinda: " + response
        st.session_state["conversation"].append(response_string)
        st.session_state["tag"].append(tag)
        st.session_state["sicher"].append(sicher)

    if st.session_state["finished_chat"]:
        form_placeholder.empty()
        st.markdown("**Wenn du erneut chatten möchtest, lade bitte den Tab neu.**")
        try:
            with open("testfile.txt", "wb") as file:
                file.writelines(st.session_state["conversation"])
                file.writelines([i for i in zip(st.session_state["tag"],st.session_state["sicher"])])
            with open("testfile.txt", "rb") as file:
                dbx = dropbox.Dropbox(os.environ["ACC_TOKEN"])
                dbx.files_upload(file.read(), "/testfile.txt")
        except:
            print("Variable nicht gesetzt")

    with col1:
        for entry in st.session_state["conversation"]:
            if "Nutzer" in entry:
                markdown_string = "<p style='text-align: right;'>" + entry + "</p>"
                st.markdown(markdown_string, unsafe_allow_html=True)
            elif "Melinda" in entry:
                markdown_string = "<p style='text-align: left;'>" + entry + "</p>"
                st.markdown(markdown_string, unsafe_allow_html=True)

    with st.expander("Details zu aktueller Antwort von Melinda"):
        tabelle = {"Label": tags, "Sicherheit": [str(i.item()) for i in sicherheiten]}
        st.table(tabelle)
        """tag_string = "Tag: " + str(st.session_state["tag"][-1])
        st.markdown(tag_string)
        sicher_string = "Sicherheit: " + str(st.session_state["sicher"][-1])
        st.markdown(sicher_string)"""
        
