import os
import streamlit as st

# Custom imports
from multipage import MultiPage
from pages import (
    main_page,
    general_information,
    chatbot,
)  # import your pages here

# Create an instance of the app
app = MultiPage()

st.set_page_config(
    page_title="ChaBoDoc",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Title of the main page
st.markdown("## ChaBoDoc")

# Add all your application here
app.add_page("Startseite", main_page.app)
app.add_page("Informationen", general_information.app)
app.add_page("ChatBot", chatbot.app)

# The main app
app.run()
