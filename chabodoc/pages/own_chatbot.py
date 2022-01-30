import streamlit as st
import json
import dropbox
import os

def get_list_of_words(words):
    output = words.split("\n")
    output = [word.strip() for word in output]
    output = list(filter(None, output))
    return output


def get_dict(tag, patterns):
    json_dict = {
        "tag": tag,
        "patterns": patterns,
        "responses": [],
        "context_set": "",
    }

    return json_dict


def app():
    st.markdown("## 4. Ein eigener ChatBot")

    st.markdown("""Hier könnt ihr einen eigenen Chatbot erstellen.""")

    st.markdown(
        """Vergebt zunächst bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht. 
        Anschließend könnt ihr euch "gute", "neutrale" sowie "schlechte" Wörter überlegen und in die dafür vorgesehenen Bereichen eintragen. 
        Beachtet dabei, dass ihr einzelne Wörter (z.B. "gut" als gutes Wort) oder auch zusammengehörende Wörter (z.B. "sehr gut" als gutes Wort) jeweils in einzelne Zeilen schreiben und dann eine neue Zeile beginnt. 
        Zum Schluss, wenn ihr fertig seid, klickt auf "Abgeben"."""
    )
    st.markdown(
        """Der Chatbot erhält beim Training eure gelabelten Daten. Ziel ist es, dass er hinterher möglichst gut auf Nutzereingaben reagieren kann. Es sollte bei den Daten also darauf geachtet werden, dass möglichst "natürliche" Eingaben gegeben werden sollten.
        **Hinweis**: Die einzelnen Eingaben müssen durch Zeilenumbrüche getrennt sein."""
    )

    with st.form("data_for_own_chatbot", clear_on_submit=False):
        group_name = st.text_input("Gruppenname")
        st.markdown("---")
        good_words = st.text_area("Gute Wörter", value="Mir geht es gut")
        st.markdown("---")
        neutral_words = st.text_area("Neutrale Wörter", value="so lala")
        st.markdown("---")
        bad_words = st.text_area("Schlechte Wörter", value="Mir geht es schlecht")
        st.markdown("---")
        submit = st.form_submit_button(label="Abgeben")

    if submit:
        group_name = group_name.strip()
        if not group_name:
            st.error(
                "ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht."
            )
            return
        elif not group_name.isalpha():
            st.error(
                "ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht."
            )
            return

        list_good = get_list_of_words(good_words)
        list_neutral = get_list_of_words(neutral_words)
        list_bad = get_list_of_words(bad_words)

        good_dict = get_dict("good words", list_good)
        neutral_dict = get_dict("neutral words", list_neutral)
        bad_dict = get_dict("bad words", list_bad)

        intents_dict = {
            "intents": [good_dict, neutral_dict, bad_dict],
        }

        st.write(intents_dict)
        
        
        with open(group_name+".json", "w") as file:
            json.dump(intents_dict, file)
        with open(group_name+".json", "rb") as file:
            dbx = dropbox.Dropbox(os.environ["ACC_TOKEN"])
            dbx.files_upload(file.read(), "/intentsfiles/"+group_name+".json", mode=dropbox.files.WriteMode.overwrite)
    """    except:
            print("Variable nicht gesetzt")
"""

    st.markdown("---")

    st.markdown(
        """Optional könnt ihr hier zusätzlich zu den [bereits vorhandenen Stopwords](https://pastebin.com/raw/N0v76srz) selbst noch eigenen Stopwords hinzufügen. 
        Wählt dazu zunächst die Checkbox aus. Anschließend öffnet sich wieder ein Bereich, in dem ihr eure Daten eingeben könnt. 
        Gebt dazu erneut euren Gruppennamen an, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht. 
        Bei den Stop Wörtern gilt ebenfalls, wie bereits oben erläutert, dass neue Wörter in eine neue Zeile geschrieben werden müssen. 
        Wenn ihr fertig seid, klickt auf "Abgeben"."""
    )

    optional_stopwords = st.checkbox("Eigene Stopwords hinzufügen")

    if optional_stopwords:
        with st.form("stopwords_own_chatbot", clear_on_submit=False):
            #group_name = st.text_input("Gruppenname")
            #st.markdown("---")
            stop_words = st.text_area("Stopwords")
            st.markdown("---")
            submit_optional = st.form_submit_button(label="Abgeben")

        if submit_optional:
            group_name = group_name.strip()
            if not group_name:
                st.error(
                    "ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht."
                )
                return
            elif not group_name.isalpha():
                st.error(
                    "ERROR: Vergebt bitte einen Gruppennamen, der aus mindestens einem Zeichen und nur Buchstaben (ohne Sonderzeichen, Zahlen, Leerzeichen) besteht."
                )
                return

            list_stop = get_list_of_words(stop_words)
            with open(group_name+".txt", "w") as file:
                file.writelines(list_stop)
            with open(group_name+".txt", "rb") as file:
                dbx = dropbox.Dropbox(os.environ["ACC_TOKEN"])
                dbx.files_upload(file.read(), "/stopfiles/"+group_name+".txt", mode=dropbox.files.WriteMode.overwrite)


