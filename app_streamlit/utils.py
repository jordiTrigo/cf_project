import streamlit as st
import pandas as pd

@st.cache_data  # para mejorar la velocidad de carga
def load_data():
    my_df = pd.read_csv('data/data_recipes.csv')
    return my_df