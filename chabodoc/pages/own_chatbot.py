import streamlit as st


def get_list_of_words(words):
    print(words)
    output = words.split("\n")
    output = [word.strip() for word in output]
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
    st.markdown("## 3. Mein eigener ChatBot")

    st.markdown("""Hier könnt Ihr einen eigenen Chatbot erstellen.""")

    st.markdown(
        """Bitte separiert alle Wörter, die ihr eingebt mit Komma voneinander, sonst kann es zu Problemen kommen."""
    )

    with st.form("Daten für den eigenen ChatBot", clear_on_submit=False):
        group_name = st.text_input("Gruppenname")
        st.markdown("---")
        good_words = st.text_area("Gute Wörter")
        st.markdown("---")
        neutral_words = st.text_area("Neutrale Wörter")
        st.markdown("---")
        bad_words = st.text_area("Schlechte Wörter")
        st.markdown("---")
        submit = st.form_submit_button(label="Abgeben")

    if submit:
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

    st.sidebar.image("./images/Logo_Uni_Luebeck_600dpi.png", use_column_width=True)
