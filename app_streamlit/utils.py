import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import io

import seaborn as sns
import altair as alt

from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix



@st.cache_data  # para mejorar la velocidad de carga
def load_data():
    my_df = pd.read_csv('data/data_recipes.csv')
    return my_df


def do_show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    with st.expander("Ver info del DataFrame", expanded=True):
        st.text(df_info)


def do_show_missingno_matrix():
    # Convertir la figura a una imagen
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Mostrar la matriz en Streamlit usando st.image
    st.image(buf, use_column_width=True)


def do_show_heatmap_correlation(correlation_matrix):
    # Aumentamos el tamaño del gráfico para mostrar los valores de correlación de manera más clara
    plt.figure(figsize=(8, 6))

    # Creamos el mapa de calor con anotaciones de valores
    sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap="coolwarm", cbar=True)

    # Establece el título
    plt.title("Correlación entre las Variables Numéricas")

    # Mostramos el gráfico en Streamlit
    st.pyplot(plt)

    # Cerrar la figura de Matplotlib para evitar problemas de memoria
    plt.close()


def do_show_chart_circle_plot(my_df):

    chart_circle = alt.Chart(my_df).mark_circle(size=60).encode(
        x=alt.X(alt.repeat("column"), type='quantitative'),
        y=alt.Y(alt.repeat("row"), type='quantitative'),
        color=alt.Color('high_traffic:N', scale=alt.Scale(scheme='set1'))
    ).properties(
        width=150,
        height=150
    ).repeat(
        row=['calories', 'carbohydrate', 'sugar', 'protein'],
        column=['calories', 'carbohydrate', 'sugar', 'protein']
    ).interactive()

    st.altair_chart(chart_circle, use_container_width=True)


def do_show_scatter_plots(my_df):

    # Crear scatter plots para cada combinación
    for variable in ['protein', 'sugar', 'carbohydrate']:
        scatter_plot = alt.Chart(my_df).mark_circle().encode(
            x=alt.X('calories:Q', title='Calories'),
            y=alt.Y(f'{variable}:Q', title=variable),
            color=alt.Color('high_traffic:N', scale=alt.Scale(scheme='tableau10'), legend=alt.Legend(title='High Traffic'))
        ).properties(
            width=400,
            height=300,
            title=f'Calories vs {variable}'
        ).interactive()

        st.altair_chart(scatter_plot, use_container_width=False)


def test_transformers(my_df, columns):

    pt = PowerTransformer()
    transformed_data = pt.fit_transform(my_df[columns])

    transformed_df = pd.DataFrame(transformed_data, columns=columns)

    for col in columns:
        
        original_distribution = alt.Chart(my_df).mark_bar().encode(
            alt.X(f'{col}:Q', bin=alt.Bin(maxbins=30)),
            alt.Y('count()')
        ).properties(
            title=f'Original Distribution of {col}'
        ).interactive()

        transformed_distribution = alt.Chart(transformed_df).mark_bar().encode(
            alt.X(f'{col}:Q', bin=alt.Bin(maxbins=30)),
            alt.Y('count()')
        ).properties(
            title=f'Transformed Distribution of {col}'
        ).interactive()

        st.altair_chart(original_distribution | transformed_distribution, use_container_width=True)

    
def do_prepare_hot_encoder(my_df, categoric_cols):

    from sklearn.preprocessing import OneHotEncoder

    # Instanciamos un objeto de la clase OneHotEncoder
    ohe = OneHotEncoder()

    # Transformación de las columnas categóricas
    encoded_cols = ohe.fit_transform(my_df[categoric_cols])
    encoded_df = pd.DataFrame(encoded_cols.toarray(),
                              columns=ohe.get_feature_names_out(categoric_cols))
    return encoded_df


def get_apply_power_transformer(my_df_enc):

    # Ahora vamos a aplicar la transformación PowerTransformer
    my_pt = PowerTransformer()
    my_df_enc[['calories','protein','carbohydrate','sugar']] = my_pt.fit_transform(my_df_enc[['calories','protein','carbohydrate','sugar']])

    # Creamos las variables correspondientes. En el axis=1 eliminamos la que hace referencia a
    # 'high_traffic' y añadimos la 'y' que contiene a 'high_traffic'.
    X = my_df_enc.drop('high_traffic', axis=1)
    y = my_df_enc['high_traffic']

    return X, y


def get_train_test_split(X, y):

    # Separacion de los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    return X_train, X_test, y_train, y_test


def do_apply_linear_regression(X, y, X_train, X_test, y_train, y_test):

    from sklearn.linear_model import LogisticRegression

    # Definimos el flujo o pipeline que usamos para aplicar el modelo
    my_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif)),
        ('classifier', LogisticRegression())
    ])

    # Definición de los hyperparametros para GridSearchCV
    parameters = {
        'selector__k': ['all'],
        'classifier__C': [0.001, 0.1, 1, 10, 100, 1000],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear','saga']
    }

    # Instanciamos GridSearchCV con 5-fold cross-validation
    my_grid_search = GridSearchCV(
                    my_pipeline,
                    parameters,
                    cv=5,
                    scoring='precision'
                    )

    # Con los datos de train realizamos un fit de GridSearchCV
    my_grid_search.fit(X_train, y_train)

    # Obtenemos el mejor estimador
    my_best_estimator = my_grid_search.best_estimator_

    # Mostramos los parámetros del estimador obtenido
    print("Parámetros del mejor estimador:", my_best_estimator.get_params())

    # Ahora vamos a evaluar el estimador usando los datos de test
    y_prediccion_proba = my_best_estimator.predict_proba(X_test)
    my_threshold = 0.6
    y_prediccion = (y_prediccion_proba[:,1] > my_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_prediccion)
    f1 = f1_score(y_test, y_prediccion)
    precision = precision_score(y_test, y_prediccion)
    recall = recall_score(y_test, y_prediccion)

    # Mostramos los resultados de la Regresión Logística
    with st.expander("**Resultados de la Regresión Logística:**", expanded=True):
        st.text(f'Accuracy: {accuracy}')
        st.text(f'F1: {f1}')
        st.text(f'Precision: {precision}')
        st.text(f'Recall: {recall}')

    # Obtenemos los indices de las recetas con 'high_traffic' respecto a los datos de test
    my_high_traffic_indexs = np.where(y_prediccion == 1)[0]

    # Conseguimos las recetas (recipes) que tienen un 'high_traffic' alto
    my_high_traffic_recipes = X_test.iloc[my_high_traffic_indexs]

    # Vamos a graficar las puntuaciones respecto a las características seleccionadas
    selector = my_best_estimator.named_steps['selector']
    selected_indices = selector.get_support(indices=True)

    # 'selected_scores': Contiene las puntuaciones (scores) de las características (features) seleccionadas por el
    # modelo de Regresión Logística. Estas puntuaciones indican el peso de las características para predecir
    # si una receta generará tráfico alto o bajo.
    selected_scores = selector.scores_[selected_indices]

    # 'selected_features': Aquí tenemos las características (features) seleccionadas que se corresponden con
    # las columnas del dataset original que se utilizaron como características para entrenar el modelo.
    # Cada característica se asocia con su respectiva puntuación.
    selected_features = X.columns[selected_indices]

    # Mostramos un gráfico de tipo bar plot con las puntuaciones respecto a las características estudidadas
    plt.figure(figsize=(8, 6))
    
    # Define una paleta de colores personalizada
    custom_palette = ["#586ba4", "#324376", "#f5dd90", "#f68e5f", "#f76c5e"]
    sns.barplot(x=selected_scores, y=selected_features, palette=custom_palette)

    plt.title('Puntuaciones de las características via el modelo de Regresión Logística')
    plt.xlabel('Puntuación')
    plt.ylabel('Característica')
    st.pyplot(plt)

    # Matriz de confusión del modelo SVM
    lr_confusion = confusion_matrix(y_test, y_prediccion)

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        lr_confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['High Traffic', 'Low Traffic'],
        yticklabels=['High Traffic', 'Low Traffic']
    )

    plt.title("Matriz de Confusión - Regresión Logística")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    st.pyplot(plt)

    return my_high_traffic_recipes


def do_show_recipes_high_traffic(my_high_traffic_recipes, categorical_var):
    
    # Columna/Variable categóricas: categorical_var
    df_categoricas = pd.DataFrame(
                        my_high_traffic_recipes.filter(like=categorical_var)
                                            .sum()
                                            .sort_values(ascending=False)
                    )

    df_categoricas.rename(columns={0:'Count'}, inplace=True)

    # Mostramos la media de las categorías
    plt.figure(figsize=(8, 6))

    # Define una paleta de colores personalizada
    custom_palette = ["#493657", "#ce7da5", "#bee5bf", "#dff3e3", "#ffd1ba"]
    sns.barplot(x=df_categoricas.index, y='Count', data=df_categoricas, palette=custom_palette)

    plt.xticks(rotation=45)
    st.pyplot(plt)


def do_apply_svm(X, y, X_train, X_test, y_train, y_test):
    
    from sklearn.svm import SVC

    # Definir un flujo de trabajo para SVM
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif)),
        ('classifier', SVC(probability=True))
    ])

    # Definir parámetros para búsqueda en cuadrícula
    svm_parameters = {
        'selector__k': ['all'],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
    }

    # Instanciar GridSearchCV
    svm_grid_search = GridSearchCV(svm_pipeline, svm_parameters, cv=5, scoring='precision')

    # Entrenar el modelo SVM
    svm_grid_search.fit(X_train, y_train)

    # Evaluar el modelo SVM en el conjunto de prueba
    svm_best_estimator = svm_grid_search.best_estimator_
    svm_y_pred_prob = svm_best_estimator.predict_proba(X_test)[:, 1]
    my_threshold = 0.55
    svm_y_pred = (svm_y_pred_prob >= my_threshold).astype(int)

    # Calcular métricas para el modelo SVM
    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    svm_f1 = f1_score(y_test, svm_y_pred)
    svm_precision = precision_score(y_test, svm_y_pred)
    svm_recall = recall_score(y_test, svm_y_pred)

    # Mostrar resultados del modelo SVM
    with st.expander("**Resultados del modelo SVM:**", expanded=True):
        st.text(f'Accuracy: {svm_accuracy}')
        st.text(f'F1: {svm_f1}')
        st.text(f'Precision: {svm_precision}')
        st.text(f'Recall: {svm_recall}')

    # Índices de las recetas con tráfico alto respecto el conjunto de test
    high_traffic_inds = np.where(svm_y_pred == 1)[0]

    # Recetas con un tráfico alto
    high_traffic_recs = X_test.iloc[high_traffic_inds]

    # Vamos a graficar las puntuaciones respecto a las características seleccionadas
    selector_s = svm_best_estimator.named_steps['selector']
    selected_inds = selector_s.get_support(indices=True)
    selected_scores_s = selector_s.scores_[selected_inds]
    selected_features_s = X.columns[selected_inds]

    # Mostramos un gráfico de tipo bar plot con las puntuaciones respecto a las características estudidadas
    plt.figure(figsize=(8, 6))
    
    # Define una paleta de colores personalizada
    custom_palette = ["#586ba4", "#324376", "#f5dd90", "#f68e5f", "#f76c5e"]
    sns.barplot(x=selected_scores_s, y=selected_features_s, palette=custom_palette)

    plt.title('Puntuaciones de las características via el modelo SVM')
    plt.xlabel('Puntuación')
    plt.ylabel('Característica')
    st.pyplot(plt)


    # Matriz de confusión del modelo SVM
    svm_confusion = confusion_matrix(y_test, svm_y_pred)

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        svm_confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=['High Traffic', 'Low Traffic'],
        yticklabels=['High Traffic', 'Low Traffic']
    )

    plt.title("Matriz de Confusión - SVM")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    st.pyplot(plt)

    return high_traffic_recs