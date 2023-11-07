import streamlit as st

import pandas as pd
import numpy as np
import missingno as msno


from utils import (
    load_data, 
    do_show_info, 
    do_show_missingno_matrix, 
    do_show_heatmap_correlation,
    do_show_chart_circle_plot,
    do_show_scatter_plots,
    test_transformers,
    do_prepare_hot_encoder,
    get_apply_power_transformer,
    get_train_test_split,
    do_apply_linear_regression,
    do_show_recipes_high_traffic,
    do_apply_svm,
)



st.set_page_config(layout="wide")

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

do_show_info(df)


st.markdown('''
    Comprobamos el número de valores únicos que contiene la columna 'recipe'.
    Verificamos que la columna 'recipe' contiene 947 valores únicos.
''')
st.write(df['recipe'].nunique())


st.markdown('''
    Eliminamos la columna 'recipe' ya que podemos utilizar el índice del Dataframe:
''')
st.code("df.drop(['recipe'],axis=1, inplace=True)", language="python")
df.drop(['recipe'],axis=1, inplace=True)


st.markdown('''
    La columna 'servings' debería ser numérica, pero nos retorna que es de tipo object (dtype=object)
    Investigamos el porqué.
''')
st.dataframe(df['servings'].unique())
        

st.markdown('''
    Comprobamos que contiene el sufijo ' as a snack'. Eliminamos el sufijo ' as a snack' ya que no aporta nada
    y transformamos la columna via astype en un tipo categórico (category).
''')

df['servings'] = df['servings'].str.replace(' as a snack','').astype('category')
st.dataframe(df['servings'])

st.markdown('''
    Comprobamos cuáles son los valores únicos para la columna 'category':
''')
st.dataframe(df['category'].unique())

st.markdown('''
    Agrupamos las categorías que representan carne (meat) y las que representan vegetables.
    Se observan varias categorías relacionadas con alimentos. Decidimos agrupar 'Pork',
    'Chicken Breast', y 'Chicken' bajo la categoría 'Meat', y 'Potato' bajo la categoría 'Vegetable'.
''')

my_code = '''
    df['category'] = df['category'].replace(['Pork','Chicken Breast','Chicken'],'Meat')
    df['category'] = df['category'].replace('Potato','Vegetable')
'''
st.code(my_code, language="python")
df['category'] = df['category'].replace(['Pork','Chicken Breast','Chicken'],'Meat')
df['category'] = df['category'].replace('Potato','Vegetable')
st.dataframe(df['category'])


st.markdown('''
    Transfomamos la columna 'high_traffic' como una columna categórica (categorical column).
''')
my_code = '''
    df['category'] = df['category'].astype('category')
'''
st.code(my_code, language="python")
df['category'] = df['category'].astype('category')
st.dataframe(df['category'])


st.markdown('''
    Utilizamos el método describe() para comprobar las variables numéricas y observamos que no contienen
    valores negativos.
''')
st.write(df.describe())


st.markdown('''
    Comprobamos los valores únicos para la columna 'high_traffic'.
    Se observa que esta columna contiene el valor 'High' y valores nulos (NaN).
''')
st.write(df['high_traffic'].unique())


st.markdown('''
    Modificamos los valores de la columna 'high_traffic' de manera que los valores con el literal 'High' los
    transformamos al valor 1, y los valores nulos al valor 0.
    Finalmente modificamos la columna 'high_traffic' a un tipo categórico.
''')
my_code = '''
    reps = {'High':1, np.nan:0}
    df['high_traffic'] = df['high_traffic'].replace(reps).astype('category')
'''
st.code(my_code, language="python")

reps = {'High':1, np.nan:0}
df['high_traffic'] = df['high_traffic'].replace(reps).astype('category')

st.markdown('''
    Mostramos la columna 'high_traffic' tras la transformación:
''')
st.dataframe(df['high_traffic'])


st.markdown('''
    Visualizamos la distribution de los valores de la columna 'high_traffic':
''')
st.dataframe(df['high_traffic'].value_counts())


st.markdown('''
    Comprobamos si los valores nulos en las columnas 'calories', 'carbohydrate', 'sugar' y 'protein'
    se encuentran en las mismas filas y, vemos que se verifica que los valores nulos de estas columnas
    están en las mismas filas.
''')
st.dataframe(df[df['calories'].isna()].head(10))


st.markdown('''
    Utilizamos la libreria 'missingno' para visualizar los valores nulos en el Dataframe.
''')
my_code = '''
    msno.matrix(df) 
'''
st.code(my_code, language="python")

# Crear la matriz de missingno
msno.matrix(df)

do_show_missingno_matrix()


st.markdown('''
    Como hemos observado, los valores nulos se encuentran ubicados en las columnas
    'calories','carbohydrate','sugar' y 'protein'.
    
    Para enriquecer el Dataframe y no perder filas, imputamos los valores nulos que encontramos en las
    columnas 'calories','carbohydrate','sugar' y 'protein' por la media de las columnas resultante de
    agrupar por la columna 'category'.
''')

my_code = '''
    cols_na = ['calories','carbohydrate','sugar','protein']

    for col in cols_na:
        df[col] = df.groupby('category', observed=True)[col].transform(lambda x: x.fillna(x.mean()))
'''
st.code(my_code, language="python")

cols_na = ['calories','carbohydrate','sugar','protein']

for col in cols_na:
    df[col] = df.groupby('category', observed=True)[col].transform(lambda x: x.fillna(x.mean()))

st.dataframe(df)

st.markdown('''
    Observamos el Dataframe resultante después de aplicar estas modificaciones.
''')
my_code = '''
    df.info()
'''
st.code(my_code, language="python")

do_show_info(df)


st.markdown('''
    Estos pasos nos han permitido limpiar y preparar nuestro DataFrame para su posterior análisis y modelado.
    Miramos las primeras 10 filas del DataFrame.
''')
st.dataframe(df.head(10))


st.markdown('''
    Usamos un gráfico de tipo Heatmap de la librería Seaborn para comprobar la correlación entre
    las columnas numéricas.
''')

numeric = df[['calories', 'carbohydrate', 'sugar', 'protein']]
correlation_matrix = numeric.corr()

do_show_heatmap_correlation(correlation_matrix)


st.markdown('''
    El gráfico nos proporciona información sobre la distribución de los datos de cada característica
    respecto al volúmen de tráfico.
    
    Podemos observar de manera efectiva dimensiones adicionales y patrones dentro del gráfico.
''')

do_show_chart_circle_plot(df)


st.markdown('''
    Mostramos la distribución de la columna 'calories' respecto las otras columnas numéricas ie, 
    'protein', 'sugar' y 'carbohydrate':
''')

do_show_scatter_plots(df)


st.markdown('''
    Ahora realizamos la transformacion de las variables aplicando PowerTransformer y 'normalizando'
    así su distribución.
    
    Mostramos las distribuciones antes y después de aplicar el Power Transform
''')

cols = ['calories', 'sugar', 'protein', 'carbohydrate']
test_transformers(df, cols)


st.markdown('''

    ## Resúmen

    En esta parte del proyecto, hemos realizado un análisis exploratorio del dataset y hemos aplicado 
    una transformación a las variables numéricas utilizando PowerTransformer de scikit-learn para poder 
    ajustar sus distribuciones.

    Algunas observaciones clave:

    1. Matriz de Correlación: He utilizado un heatmap para visualizar la correlación entre las variables numéricas. 
        Esto es útil para identificar relaciones entre las variables. He observado que las correlaciones son en su mayoría débiles, 
        lo que indica que no hay una dependencia lineal fuerte entre estas variables. He comprobado que la mayor correlación se da 
        entre las variables 'calories' y 'protein'.

    2. Gráfico Pair Plot: El gráfico pair plot es una herramienta útil para explorar la distribución de datos y observar cómo se 
        relacionan entre sí. Lo he utilizado para visualizar la distribución de las variables numéricas y cómo se relacionan con 
        la variable objetivo 'high_traffic'. Esto puede ayudarnos a identificar patrones visuales en los datos.

    3. Scatter Plots: La creación de scatter plots me ha permitido analizar la relación entre la variable 'calories' y las otras 
        variables numéricas ('protein', 'sugar' y 'carbohydrate'). Estos gráficos proporcionan una representación visual de cómo 
        estas variables se relacionan entre sí. Por ejemplo, he observado cómo la variable 'calories' se relaciona con las variables 
        'protein', 'sugar' y 'carbohydrate'.

    4. Transformación PowerTransformer: He aplicado PowerTransformer a las variables numéricas para ajustar sus distribuciones. Esto es 
        importante ya que muchos algoritmos de machine learning funcionan mejor con datos que siguen una distribución gaussiana o normal. 
        La comparación de las distribuciones antes y después de la transformación muestra claramente cómo esta técnica puede mejorar la 
        simetría de los datos.

    En general, este análisis exploratorio es esencial para comprender los datos y cómo se distribuyen. La transformación de las variables 
        numéricas es un paso importante para preparar los datos para la construcción de modelos de machine learning.
            
    
    ## Desarrollo del modelo

    Nuestro objetivo es comprobar si podemos clasificar o no una receta en función del tráfico (high_traffic) que recibe (alto: 1 o bajo: 0), 
        es decir lo que quiero es construir modelos de machine learning para predecir si una receta generará un tráfico alto o no. Por lo tanto, 
        estamos ante un caso de clasificación (tráfico alto o bajo) y algunos modelos usados para la clasificación binaria son la Logistic Regression, 
        Máquinas de Soporte Vectorial (SVM), Bosques Aleatorios, y otros. Utilizaré para ello como modelo base la Logistic Regression y después el 
        modelo SVM y, finalmente los compararé.

    Para evaluar los modelos, he decido utilizar las siguientes métricas: accuracy, F1, precision and recall. En las siguientes líneas, veremos cómo 
        he preparado los datos para poder aplicar los modelos de Machine Learning, el ajuste del modelo y los resultados obtenidos. Observaremos también 
        qué tipos de recetas tienen éxito al aumentar el tráfico a la web.

    He utilizado un flujo de trabajo (pipeline en el argot) que contiene una transformación a nivel de escala, una instancia del modelo y un selector 
        para obtener las características más relevantes. He escogido también una lista de parámetros adaptados para la Logistic Regression.

    Luego obtuve las métricas (F1, recall, precision, accuracy), grafiqué el peso de todas las características utilizadas y, finalmente, filtré los valores 
        de predicción asociados a un tráfico alto con el fin de mostrar qué tipo de recetas tienen más popularidad y éxito, usando para ello un gráfico de 
        barras que muestra las categorías más comunes.
            
    
    ## Modificando y transformando los datos: Preparación

    Aplicaremos diferentes métodos de preparación y transformación al constatar que tenemos diferentes tipos de variables, por un lado tenemos variables 
        numéricas y por otro categóricas.

    Para las variables categóricas, utilizaremos la codificación one-hot, ya que proporciona una mayor precisión al aplicar los modelos.

    En el caso de las variables numéricas y después de los resultados que hemos observado anteriormente, utilizaremos PowerTransformer.
''')

# Las columnas categóricas son
categoric_cols = ['category','servings']

encoded_df = do_prepare_hot_encoder(df, categoric_cols)

df = pd.concat([df, encoded_df], axis=1)

# Eliminamos las columnas categóricas iniciales
df_enc = df.drop(categoric_cols, axis=1)

# Mostramos el Dataframe después de las transformaciones aplicadas sobre él
st.dataframe(df_enc.head())

X, y = get_apply_power_transformer(df_enc)

st.markdown('''
    
    ## Aplicamos el modelo de Logistic Regression

    Consideraciones a tener en cuenta.

    ### GridSearchCV

    GridSearchCV es una técnica utilizamos para buscar los mejores hiperparámetros asociados a un modelo de machine learning. Los hiperparámetros son 
        configuraciones que pueden ser modificadas tales como la elección del kernel, la tasa de aprendizaje en redes neuronales o la profundidad del 
        árbol en árboles de decisión.

    El papel de GridSearchCV es vital para ayudar a encontrar la combinación óptima de hiperparámetros que maximice el rendimiento del modelo. 
        Funciona de la siguiente manera:

    Define un conjunto de hiperparámetros y sus valores posibles. Por ejemplo, los hiperparámetros podrían incluir el tipo de kernel, el valor de C (regularización), 
        y otros parámetros específicos del modelo.

    GridSearchCV realiza una búsqueda exhaustiva a través de todas las combinaciones posibles de valores de hiperparámetros. Puedes especificar qué métrica de evaluación 
        deseas optimizar, como 'precision', 'recall', 'accuracy', etc.

    Se entrena y evalúa el modelo con cada combinación de hiperparámetros utilizando una validación cruzada.

    Al final, GridSearchCV devuelve la mejor combinación de hiperparámetros que maximiza la métrica de evaluación especificada.

    Utilizaré GridSearchCV para encontrar la mejor combinación de hiperparámetros para el modelo de Regresión Logística.

    ## Matriz de confusión

    La matriz de confusión es una herramienta fundamental para evaluar el rendimiento de un modelo de clasificación. Muestra la relación entre las predicciones del modelo 
        y el valor real en el conjunto de datos de prueba.

    La matriz de confusión se divide en cuatro partes:

        1. Verdaderos positivos (True Positives - TP): Representa los casos en los cuáles el modelo predijo correctamente el valor como positivo (high_traffic = 1) cuando 
            el verdadero valor era positivo.

        2. Falsos positivos (False Positives - FP): Representa los casos en los que el modelo predijo incorrectamente el valor como positivo (high_traffic = 1) cuando el 
            valor real era negativo (high_traffic = 0).

        3. Verdaderos negativos (True Negatives - TN): Representa los casos en los que el modelo predijo correctamente el valor como negativo (high_traffic = 0) cuando el 
            valor real era negativo.

        4. Falsos negativos (False Negatives - FN): Representa los casos en los que el modelo predijo incorrectamente el valor como negativo (high_traffic = 0) cuando el 
            valor real era positivo (high_traffic = 1).

    La matriz de confusión nos permite evaluar aspectos como la precisión (accuracy), la precisión (precision) y el recall (también llamado sensibilidad). Estas métricas 
        se calculan a partir de los valores obtenidos en la matriz de confusión y ayudan a entender el rendimiento del modelo en tareas de clasificación.

        - Precisión (Precision): Mide la proporción de casos positivos predichos correctamente en comparación con todos los casos positivos predichos. Se calcula como TP / (TP + FP).

        - Recall (Sensibilidad): Mide la proporción de casos positivos predichos correctamente en comparación con todos los casos positivos reales en el conjunto de datos. 
            Se calcula como TP / (TP + FN).

        - Exactitud (Accuracy): Mide la proporción de predicciones correctas en comparación con todas las predicciones. Se calcula como (TP + TN) / (TP + TN + FP + FN).

    La matriz de confusión y estas métricas son esenciales para identificar si el modelo tiende a cometer errores de falsos positivos o falsos negativos y ajustarlo en consecuencia.

    Utilizaré la matriz de confusión para visualizar cómo se distribuyen las predicciones del modelo de Regresión Logística en comparación con los valores reales en los 
        datos de prueba. Esto me permitirá evaluar y comprender mejor el rendimiento del modelo según verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.
''')
# Separacion de los datos en train y test
X_train, X_test, y_train, y_test = get_train_test_split(X, y)

my_high_traffic_recipes = do_apply_linear_regression(X, y, X_train, X_test, y_train, y_test)

st.markdown('''
    ## **Evaluando el modelo y sus resultados**

    Vemos que modelo de Regresión Logística ha tenido buenos resultados.

    Vemos algo más a fondo estos resultados:

    - **Accuracy**: el modelo de Regresión Logística ha obtenido alrededor de un 73% de precisión (accuracy). Debemos tener en cuenta que estamos priorizando la precisión sobre la exactitud.
    - **Precision**: el modelo de Regresión Logística obtuvo cerca del 79%.
    - **Recall**: el modelo de Regresión Logística ha obtenido alrededor del 80%.
    - **F1**: al ser la media entre las métricas precision y recall obtiene un valor un poco superior al 80%.
''')

st.markdown('''
    #### **Mostramos gráficos con las variables categóricas**

    Ahora analizo las categorías de las recetas con tráfico alto ('high_traffic') y visualizo con la ayuda de un barplot/gráfico de barras la cantidad de recetas que pertenecen a cada categoría 
        ('category_') y después a cada 'servings_'.

    Para ello creamos un nuevo DataFrame: df_categoricas. Este contiene la información sobre las diferentes variables de tipo 'category_'. Asignamos a este DataFrame los datos de las recetas que 
        tenemos con tráfico alto (my_high_traffic_recipes) y se filtra por 'category' para poder crear una gráfica con sólo las variables categóricas que referencian a 'category_'.

    A continuación, aplicamos el método .sum() a este DataFrame filtrado que nos permite obtener el total de la suma del número de recetas con tráfico alto por cada categoría.

    El resultado anterior es un objeto Serie de Pandas que ordenamos de manera descendente (de mayor a menor) según el número de recetas para cada categoría así veremos en orden de mayor a menor 
        las categorías, primero la que tiene la suma total más alta de recetas hasta la última que tiene la suma menor de recetas por esa categoría.

    Renombramos la serie a 'Count' para que el eje Y sea más descriptivo. Utilizamos el parámetro inplace=True para actualizar el DataFrame df_categoricas directamente.

    Finalmente, utilizamos  Seaborn (sns) para crear el gráfico de barras. En el eje X (x), constan las categorías (los índices del DataFrame df_categoricas), y en el eje Y (y), tenemos el total 
        de recetas para cada categoría.

    El resultado es un gráfico de barras que muestra las categorías en el eje X y el número de recetas en cada categoría en el eje Y. Esto permite visualizar cuáles son las categorías más comunes 
        entre las recetas con tráfico alto.

    Usamos el mismo procedimiento para poder mostrar una gráfica de barras para el caso de las variables categóricas 'servings_'.
''')

do_show_recipes_high_traffic(my_high_traffic_recipes, 'category')
do_show_recipes_high_traffic(my_high_traffic_recipes, 'servings')


st.markdown('''
    ### Mostramos las primeras 15 filas que contienen recetas con un alto 'high_traffic'
''')
st.dataframe(my_high_traffic_recipes.head(15))


st.markdown('''
    ## Modelo SVM

    El Support Vector Machine (Máquina de Vectores de Soporte) es un algoritmo de Machine Learning utilizado para tareas de clasificación y regresión. En el contexto de clasificación, SVM se utiliza 
        para separar dos clases distintas de datos, lo que lo convierte en una herramienta eficaz para la clasificación binaria. Además, SVM ofrece un alto grado de precisión y es una excelente opción 
            cuando la precisión es una prioridad.

    Utilizaré SVM para clasificar recetas en función de su tráfico, es decir, para predecir si una receta generará un tráfico alto o bajo.
''')

high_traffic_recs = do_apply_svm(X, y, X_train, X_test, y_train, y_test)

do_show_recipes_high_traffic(high_traffic_recs, 'category')
do_show_recipes_high_traffic(high_traffic_recs, 'servings')