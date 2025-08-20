import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# CREAR DATASET SIMULADO
# -------------------------------
@st.cache_data
def generar_datos():
    X, y = make_classification(
        n_samples=300,   # al menos 300 muestras
        n_features=6,    # al menos 6 columnas
        n_informative=4, # 4 columnas útiles
        n_redundant=1,   # 1 redundante
        n_classes=2,     # problema binario
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 7)])
    df["Target"] = y
    return df

# -------------------------------
# ENTRENAR Y EVALUAR MODELOS
# -------------------------------
def entrenar_modelos(X_train, X_test, y_train, y_test):
    modelos = {
        "KNN": KNeighborsClassifier(),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
        "Regresión Logística": LogisticRegression(max_iter=1000)
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        resultados[nombre] = {
            "modelo": modelo,
            "accuracy": acc,
            "report": report,
            "y_pred": y_pred
        }
    return resultados

# -------------------------------
# APP STREAMLIT
# -------------------------------
st.title("📊 Comparación de Modelos de Clasificación")
st.write("Se genera un dataset simulado con 300 muestras y 6 características para comparar **KNN, Árbol de Decisión y Regresión Logística**.")

# Generar datos
df = generar_datos()
st.subheader("Vista previa del Dataset")
st.write(df.head())

# Separar en train/test
X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelos
resultados = entrenar_modelos(X_train, X_test, y_train, y_test)

# Mostrar resultados
for nombre, res in resultados.items():
    st.subheader(f"🔹 {nombre}")
    st.write(f"**Accuracy:** {res['accuracy']:.2f}")
    st.text("Reporte de Clasificación:")
    st.json(res["report"])

    # Matriz de confusión
    cm = confusion_matrix(y_test, res["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de Confusión - {nombre}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)
