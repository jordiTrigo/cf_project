import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer

import io
from utils import load_data


st.title("Analizando el tráfico hacia las recetas")

st.markdown('''
    ### Exploración de Datos
''')

st.markdown('''
    Realizamos la carga de los datos del dataset de recetas.
''')

df = load_data()

st.markdown('''
    Realizamos una visualización de las primeras filas del DataFrame 
    para inspeccionar los datos iniciales.
''')
st.write(df.head())  # Visualización de los primeros registros

st.markdown('''
    Se obtiene el número de filas y columnas del DataFrame.
''')
st.write(df.shape)

st.markdown('''
    Información general del Dataframe. Se muestra información general sobre las columnas,
    incluyendo los tipos de datos y la presencia de valores nulos.
''')

buffer = io.StringIO()
df.info(buf=buffer)
df_info = buffer.getvalue()
with st.expander("Ver info del DataFrame"):
    st.text(df_info)

