import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import gzip
import pickle
import xgboost as xgb
from joblib import load
from tensorflow.keras.models import load_model
import tensorflow as tf



# Mostrar la imagen solo en la página de inicio
st.title("Análisis de Detección de Ocupación")
st.write("Grupo: Yulisa Ortiz Giraldo y Juan Pablo Noreña Londoño")
if "image_displayed" not in st.session_state:
    st.image("image1.jpg", use_container_width=True)
    st.session_state["image_displayed"] = True  # Marcar que la imagen ya se mostró

# Crear una tabla de contenido en la barra lateral
seccion = st.sidebar.radio("Tabla de Contenidos", 
                           ["Vista previa de los datos", 
                            "Información del dataset", 
                            "Análisis Descriptivo", 
                            "Mapa de calor de correlaciones", 
                            "Distribución de la variable objetivo", 
                            "Boxplots", 
                            "Conclusión: Selección del Mejor Modelo",  # Nueva ubicación
                            "Modelo Random Forest",  # Nueva sección
                            "Modelo de redes neuronales"])
# Cargar datos
@st.cache_data
def load_data():
    df_train = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatrain.csv")
    df_test = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatest.csv")
    return df_train, df_test

df_train, df_test = load_data()

# Preprocesamiento de datos
for df in [df_train, df_test]:
    df.drop(columns=["id", "date"], inplace=True, errors='ignore')

def preprocess_data(train_df, test_df):
    X_train = train_df.drop(columns=["Occupancy"], errors='ignore')
    y_train = train_df["Occupancy"]
    X_test = test_df.drop(columns=["Occupancy"], errors='ignore')
    y_test = test_df["Occupancy"]
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

X_train, X_test, y_train, y_test, scaler = preprocess_data(df_train, df_test)


# Mostrar contenido basado en la selección
if seccion == "Vista previa de los datos":
    st.subheader("Vista previa de los datos")
    st.write(df.head())

elif seccion == "Información del dataset":
    st.subheader("Información del dataset")
    st.write(df.info())
    st.write("La base de datos seleccionada para el desarrollo de la aplicación corresponde a un estudio diseñado para optimizar actividades de clasificación binaria para determinar sí una habitación está ocupada o no. Dentro de sus características, se recopilan mediciones ambientales tales como la temperatura, la humedad del ambiente, la luz o nivel de luminosidad, y niveles de CO2, donde, con base a estas se determina sí la habitación está ocupada. La información de ocupación se obtuvo mediante la obtención de imágenes capturadas por minuto, garantizando etiquetas precisas para la clasificación. Este conjunto de datos resulta muy importante y útil para la investigación basada en la detección ambiental y el diseño de sistemas de edificios inteligentes según sea el interés del usuario.")
    st.write("La base cuenta con un total de 17.895 datos con un total de 8 variables, sin embargo, se utilizará una cantidad reducida de variables debido a que aquellas como “ID” y “Fecha” no aportan información relevante para la aplicación de los temas anteriormente tratados.")
    st.write("El conjunto de datos fue obtenido del repositorio público Kaggle, ampliamente utilizado en investigaciones relacionadas con sistemas inteligentes y monitoreo ambiental. La fuente original corresponde al trabajo disponible en el siguiente enlace: https://www.kaggle.com/datasets/pooriamst/occupancy-detection.")

elif seccion == "Análisis Descriptivo":
    st.subheader("Resumen de los datos")
    st.write(df.describe())
    st.subheader("Histograma de Temperature")
    # Temperatura
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["Temperature"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Temperature')
    st.pyplot(fig)
    st.write("Del histograma anterior, se denota que la mayoría de imágenes tomadas de la habitación captaron una temperatura de entre 20°C y 21°C, siendo una temperatura ambiente la que más predomina en el conjunto de datos. Además, se observa que la temperatura mínima registrada es de 19°C y la máxima es un poco superior a 24°C. Por tanto, en la habitación no hay presencia de temperaturas que se consideren bajas o altas.")
    #  Humidity
    st.subheader("Histograma de Humidity")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["Humidity"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Humidity')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Humidity')
    st.pyplot(fig)
    st.write("De la variable “Humidity”, se observa que la humedad se encuentra entre aproximadamente un 16% y un 40%. Para su interpretación en este caso, se debe conocer cuáles son los valores de humedad normales en una habitación, para ello, la empresa Philips (sin fecha) en su publicación “¿Cómo medir la humedad recomendada en casa?” afirma que la humedad ideal debe encontrarse entre 30% y 60% para la conservación de los materiales de las paredes y el piso; por otra parte, en el blog Siber. (n.d.) mencionan que el ser humano puede estar en espacios con una humedad de 20% a 75%. Teniendo en cuenta lo anterior, se puede afirmar que la humedad en la mayoría de los datos es adecuada para las personas, para los casos cuyo valor de humedad es menor a 20% no resulta ideal pero no debería ser un inconveniente significativo.")
    # HumidityRatio
    st.subheader("Histograma de HumidityRatio")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["HumidityRatio"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('HumidityRatio')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de HumidityRatio')
    st.pyplot(fig)
    st.write("Este histograma corresponde a la cantidad derivada de la temperatura y la humedad relativa dada en kilogramo de vapor de agua por kilogramo de aire, los valores se encuentran entre 0.002 kg vapor de agua/kg de aire hasta 0.0065 kg vapor de agua/ kg de aire aproximadamente. Según la explicación de la variable anterior, los resultados de la relación se encuentran en un rango adecuado.")
    # Light
    st.subheader("Histograma de Light")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["Light"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Light')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Light')
    st.pyplot(fig)
    st.write("De la variable Light, se observa que en la gran mayoría de los datos no hubo presencia de luz, no obstante, se denota el incremento en los valores cercanos a 500lux, esto indica que en estos casos sí se hizo uso de la luz eléctrica en la habitación debido al flujo luminoso provocado por el bombillo. Este podría ser un factor importante en la determinación de sí la habitación está ocupada o no, pero esto se confirmará más adelante en los resultados.")
    # CO2
    st.subheader("Histograma de CO2")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["CO2"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('CO2')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de CO2')
    st.pyplot(fig)
    st.write("Para la variable de CO2, se observa que los niveles de CO2 dados en ppm (partículas por millón) de aproximadamente 400 a 700pm son los más presentes en el conjunto de datos. Se registran más casos donde los niveles de CO2 son mucho mayores a los recurrentes, llegando hasta los 2000ppm. Para comprender la tolerancia de una persona hacia el CO2, la empresa Enectiva (2017) en su publicación “Efectos de la concentración de CO₂ para la salud humana” expone que las habitaciones deben tener niveles de CO2 máximo recomendado en 1200-1500ppm, a partir de este valor pueden presentarse efectos secundarios sobre las personas, como la fatiga y la pérdida de concentración; a niveles mayores a los presentes en el histograma puede provocar aumento del ritmo cardíaco, dificultades respiratorias, náuseas, e inclusive la pérdida de la consciencia. Los niveles de CO2 pueden ser un indicativo clave para determinar sí la habitación está ocupada o no debido a la naturaleza del ser humano de expulsar dióxido de carbono “CO2” en su exhalación, aunque debe tenerse en cuenta que un nivel elevado de CO2 puede deberse a razones diferentes del proceso de respiración de la persona.")
    
elif seccion == "Distribución de la variable objetivo":
    st.subheader("Distribución de la variable objetivo")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Occupancy"], ax=ax)
    st.pyplot(fig)
    st.write("De la variable respuesta “Occupancy”, se obtiene que en su mayoría de casos se tiene como resultado que la habitación no se encuentra ocupada, denotada con el valor de cero y por el valor 1 en el caso contrario. Se obtuvo que en el 78.9% de los casos la habitación está vacía, y en el 21.1% se encuentra ocupada.")

elif seccion == "Mapa de calor de correlaciones":
    st.subheader("Mapa de calor de correlaciones")
    st.write("Se plantea la matriz de correlación de las variables mencionadas para verificar qué tan relacionadas se encuentran con la variable respuesta de “Occupancy” y así observar cuáles tendrían mayor incidencia en la toma de decisión:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)
    st.write("Según la matriz, la variable que más se correlaciona con la variable respuesta es la luz (“Light”), pues es una determinante importante en la ocupación de una habitación; seguido de ésta, se denotan las variables de temperatura y CO2, cuyas características se encuentran estrechamente relacionadas con la presencia de personas en un espacio. Por último, debe mencionarse que las variables relacionadas con la humedad presentan una muy baja correlación con la ocupación de una habitación, esto debe tenerse en cuenta en la formulación del modelo para la aplicación y considerar sí se eliminan estas variables dependiendo de los resultados que se obtengan.")

elif seccion == "Boxplots":
    st.subheader("Conjunto de boxplots")
    st.image("Boxplots.jpeg", use_container_width=True)
    st.write("""
    ### Análisis de Variables

    #### CO2 (Dióxido de carbono):
    - **Habitación vacía (rojo):** Niveles considerablemente más bajos, con una mediana en torno a 500ppm.
    - **Habitación ocupada (verde):** Niveles mucho más altos, con una mediana cerca de 1000ppm.
    - El nivel de CO2 aumenta notablemente con la ocupación, posiblemente debido a la respiración de las personas.

    #### Humidity (Humedad):
    - **Habitación vacía (rojo):** Mediana ligeramente por encima de 25, con dispersión moderada.
    - **Habitación ocupada (verde):** Mediana cerca de 30, con valores más altos.
    - La ocupación no parece variar mucho la humedad, en línea con la matriz de correlaciones.

    #### HumidityRatio (Proporción de humedad):
    - **Habitación vacía (rojo):** Valores concentrados alrededor de 0.0035.
    - **Habitación ocupada (verde):** Valores ligeramente más altos, alrededor de 0.004.
    - Aunque las diferencias no son grandes, la ocupación está asociada con un pequeño incremento en la proporción de humedad.

    #### Light (Luz):
    - **Habitación vacía (rojo):** Gran dispersión, con valores extremos muy altos.
    - **Habitación ocupada (verde):** Valores más bajos y concentrados.
    - La ocupación 0 (habitación vacía) está asociada con niveles de luz más altos y variables, posiblemente por la ausencia de personas que reduzcan el uso de iluminación artificial.

    #### Temperature (Temperatura):
    - **Habitación vacía (rojo):** Mediana cerca de 20°C, con dispersión moderada.
    - **Habitación ocupada (verde):** Mediana ligeramente más alta, alrededor de 22°C.
    - La temperatura es más alta con ocupación, posiblemente por el calor generado por las personas o el uso de calefacción.

     #### Conclusión:
    La ocupación tiene un impacto claro en **CO2, Light (luz) y Temperature (temperatura)**, aumentando sus valores en comparación con la falta de ocupación. En particular, la luz tiende a ser más alta y variable cuando hay ocupación. Otras variables como la humedad presentan cambios menores, pero no son significativos.
    """)

# Nueva sección: Conclusión sobre la selección del mejor modelo
elif seccion == "Conclusión: Selección del Mejor Modelo":
    st.subheader("Conclusión: Selección del Mejor Modelo (Random Forest)")
    st.markdown("""
    Después de evaluar varios modelos de machine learning para la tarea de predecir la ocupación de habitaciones, se determinó que el **Random Forest Classifier** es el modelo más adecuado para este problema. A continuación, se detallan las razones por las que se seleccionó este modelo y por qué los otros no fueron la mejor opción:

    #### Razones para elegir Random Forest:
    1. **Alto Rendimiento en Precisión y F1-Score**:
       - Random Forest demostró un excelente rendimiento en términos de precisión y F1-Score, logrando un buen equilibrio entre la clasificación de habitaciones ocupadas y desocupadas.

    2. **Manejo de Desequilibrio de Clases**:
       - Random Forest maneja bien el desequilibrio de clases gracias a su estructura de ensamble de múltiples árboles de decisión, mejorando la generalización.

    3. **Interpretabilidad de las Características**:
       - Random Forest proporciona una interpretación clara de la importancia de las características, permitiendo identificar variables clave como el nivel de CO2, la luz y la humedad en la predicción de ocupación.

    4. **Robustez ante Overfitting**:
       - Debido a su naturaleza basada en ensambles de múltiples árboles, Random Forest es menos propenso al sobreajuste en comparación con modelos individuales como Decision Tree.

    5. **Eficiencia y Escalabilidad**:
       - Aunque puede ser computacionalmente más costoso que algunos modelos más simples, su rendimiento general lo hace una excelente opción para conjuntos de datos medianos y grandes.

    #### Razones por las que otros modelos no fueron seleccionados:
    - **XGBoost**: Aunque XGBoost es un modelo muy potente, en este caso Random Forest logró un rendimiento similar con menor complejidad y menor necesidad de ajuste de hiperparámetros.
    
    - **Decision Tree**: Es propenso al overfitting y no generaliza tan bien como Random Forest, ya que se basa en un solo árbol en lugar de un ensamble.
    
    - **K-Nearest Neighbors (KNN)**: Requiere mucho cálculo a medida que aumenta el tamaño de los datos y no maneja bien el desequilibrio de clases.
    
    - **Red Neuronal**: Aunque las redes neuronales pueden ser muy poderosas, en este caso el modelo Random Forest alcanzó una mejor interpretabilidad y robustez sin necesidad de ajustes complejos.

    ### Conclusión Final:
    El **Random Forest Classifier** fue seleccionado como el mejor modelo debido a su alto rendimiento, capacidad para manejar el desequilibrio de clases, interpretabilidad de las características y robustez ante el overfitting. Estos factores lo convierten en la opción más adecuada para la tarea de predecir la ocupación de habitaciones, superando a otros modelos como XGBoost, Decision Tree, KNN y la red neuronal en este contexto específico.
    """)

# ==============================
# SECCIÓN: RANDOM FOREST
# ==============================
if seccion == "Modelo Random Forest":
    st.subheader("Modelo Random Forest: Predicción de Ocupación")
    st.markdown("""
    En esta sección, exploraremos el modelo **Random Forest** para predecir la ocupación de habitaciones basándonos en las siguientes variables:
    - **Temperature**: Temperatura en la habitación.
    - **Humidity**: Humedad en la habitación.
    - **Light**: Nivel de luz en la habitación.
    - **CO2**: Nivel de dióxido de carbono en la habitación.
    - **HumidityRatio**: Relación de humedad en la habitación.
    - **Occupancy**: Variable objetivo que indica si la habitación está ocupada (1) o no (0).
     """)

    # Cargar el modelo
    def load_model():
        try:
            with gzip.open('random_forest_model.pkl.gz', 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            return None

    # Verificar si el modelo está cargado en la sesión
    if "model" not in st.session_state:
        st.session_state.model = load_model()

    if st.session_state.model is not None:
        st.success("Modelo cargado correctamente.")
        model = st.session_state.model

        # **Solo aparece en la sección Random Forest**
        st.markdown("### Hacer una predicción")
        st.write("Introduce valores para hacer una predicción:")
        inputs = {}

        # Diccionario con valores mínimos y máximos de cada variable
        min_max_dict = {
            'Temperature': (19.0, 25.0),
            'Humidity': (20.0, 60.0),
            'Light': (0.0, 1500.0),
            'CO2': (400.0, 1200.0),
            'HumidityRatio': (0.003, 0.007)
        }

        columnas_modelo = list(min_max_dict.keys())

        for col in columnas_modelo:
            min_val, max_val = min_max_dict[col]
            min_val, max_val = float(min_val), float(max_val)
            inputs[col] = st.number_input(
                f"{col} ({min_val} - {max_val})", 
                min_value=min_val, 
                max_value=max_val, 
                value=(min_val + max_val) / 2
            )

        if st.button("Predecir"):
            input_df = pd.DataFrame([inputs])
            prediccion = model.predict(input_df)
            ocupacion = "Ocupada" if prediccion[0] == 1 else "No Ocupada"
            st.success(f"La predicción de ocupación es: {ocupacion}")

            # Importancia de las variables
            st.markdown("### Importancia de las variables")
            importancia = model.feature_importances_
            imp_df = pd.DataFrame({'Variable': columnas_modelo, 'Importancia': importancia})
            imp_df = imp_df.sort_values(by='Importancia', ascending=False)

            plt.figure(figsize=(8, 5))
            sns.barplot(x='Importancia', y='Variable', data=imp_df, palette='viridis')
            st.pyplot(plt)

    else:
        st.error("No se pudo cargar el modelo. Verifica el archivo.")

    st.sidebar.info("Esta aplicación predice la ocupación de una habitación usando un modelo Random Forest.")

# ==============================
# SECCIÓN: REDES NEURONALES
# ==============================
if seccion == "Modelo de redes neuronales":
    # --- Cargar modelo y scaler ---
    @st.cache_resource
    def load_assets():
        model = tf.keras.models.load_model("best_model.h5")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    model, scaler = load_assets()

    # --- Título de la aplicación ---
    st.title("🔍 Predicción de Ocupación con Redes Neuronales")

    st.markdown("Ingrese los valores de las variables para hacer una predicción:")

    # --- Entrada de usuario con sliders ---
    temperature = st.slider("Temperature (°C)", 19.0, 25.0, 22.0)
    humidity = st.slider("Humidity (%)", 20.0, 60.0, 40.0)
    light = st.slider("Light (lux)", 0.0, 1500.0, 750.0)
    co2 = st.slider("CO2 (ppm)", 400.0, 1200.0, 800.0)
    humidity_ratio = st.slider("Humidity Ratio", 0.003, 0.007, 0.005)

    # --- Inicializar historial de predicciones ---
    if "history" not in st.session_state:
        st.session_state.history = []

    # --- Sección de Redes Neuronales ---
    st.subheader("🤖 Predicción con Redes Neuronales")

    # --- Botón de predicción ---
    if st.button("Predecir con Redes Neuronales"):
        # Crear array con los valores ingresados
        input_data = np.array([[temperature, humidity, light, co2, humidity_ratio]])
        
        # Escalar los valores de entrada
        input_scaled = scaler.transform(input_data)
        
        # Hacer la predicción con el modelo
        prediction = model.predict(input_scaled)

        # Verificar si la predicción tiene la forma esperada
        if prediction.ndim == 2 and prediction.shape[1] > 1:
            predicted_class = np.argmax(prediction)  # Softmax
        else:
            predicted_class = int(round(prediction[0]))  # Salida escalar
        
        # Guardar en el historial solo para la sección de redes neuronales
        st.session_state.history.append({
            "Temperature": temperature,
            "Humidity": humidity,
            "Light": light,
            "CO2": co2,
            "Humidity Ratio": humidity_ratio,
            "Prediction": "Ocupada" if predicted_class == 1 else "Desocupada"
        })
        
        # Mostrar el resultado
        st.subheader("🧠 Resultado de la Predicción:")
        if predicted_class == 1:
            st.success("✅ La sala está ocupada.")
        else:
            st.warning("❌ La sala está desocupada.")

        # Mostrar probabilidades de salida
        st.write("📊 **Predicción cruda (probabilidades softmax):**", prediction)

# --- Mostrar hiperparámetros del modelo ---
    st.subheader("📌 Hiperparámetros del Modelo")
    st.write({
        "Capas Ocultas": 1,
        "Neuronas en capa oculta": 176,
        "Función de Activación": "ReLU",
        "Optimizador": "RMSprop",
        "Learning Rate": 0.065,
        "Batch Size": 24,
        "Epochs": 5
    })



