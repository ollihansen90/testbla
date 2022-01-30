"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework. 
(taken from: https://github.com/prakharrathi25/data-storyteller)
"""

import streamlit as st

# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args:
            title ([str]): The title of page which we are adding to the list of apps

            func: Python function to render this page in Streamlit
        """

        self.pages.append({"title": title, "function": func})

    def run(self):
        """with st.header("Hallo"):
            col1,  col2 = st.columns((1,1.5))
            with col1:
                st.image("./chabodoc/images/Logo_UzL.png", use_column_width=True)
            with col2:
                st.image("./chabodoc/images/Logo_UKT.png", use_column_width=True)"""
        

        st.sidebar.title("ChaBoDoc")
        # Drodown to select the page to run
        page = st.sidebar.selectbox(
            "Navigation", self.pages, format_func=lambda page: page["title"]
        )

        with st.container():
            st.sidebar.image("./chabodoc/images/Logo_UzL.png", use_column_width=True)
            st.sidebar.image("./chabodoc/images/Logo_UKT.png", use_column_width=True)
        # run the app function
        page["function"]()
