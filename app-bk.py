import streamlit as st
from multiapp import MultiApp
from apps import (
    housing
)

st.set_page_config(layout="wide")


apps = MultiApp()

# Add all your application here
apps.add_app("U.S. Real Estate Data", housing.app)


# The main app
apps.run()
