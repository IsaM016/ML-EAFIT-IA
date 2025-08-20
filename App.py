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
def generar_datos(n_samples, n_features, n_classes):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=1,
        n_classes=n_classes,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, n_features + 1)])
    df["Target"] = y
    return df

# -------------------------------
# ENTRENAR Y EVALUAR MODELOS
# -------------------------------
def entrenar_modelos(X_train, X_test, y_train, y_test, k, depth, c):
    modelos = {
        "KNN": KNeighborsClassifier(n_neighbors=k),
        "Árbol de Decisión": DecisionTreeClassifier(max_depth=depth, random_state=42),
        "Regresión Logística": LogisticRegression(max_iter=1000, C=c)
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
st.title("📊 Clasificación Dinámica con KNN, Árbol de Decisión y Regresión Logística")

# ---- Parámetros dataset ----
st.sidebar.header("📌 Configuración del Dataset")
n_samples = st.sidebar.slider("Número de muestras", 100, 1000, 300, step=50)
n_features = st.sidebar.slider("Número de características", 6, 15, 6)
n_classes = st.sidebar.radio("Número de clases", [2, 3])

# ---- Parámetros modelos ----
st.sidebar.header("⚙️ Hiperparámetros de Modelos")
k = st.sidebar.slider("Número de vecinos (KNN)", 1, 20, 5)
depth = st.sidebar.slider("Profundidad máxima (Árbol)", 1, 20, 5)
c = st.sidebar.slider("C (Regularización Regresión Logística)", 0.01, 10.0, 1.0)

# Generar datos
df = generar_datos(n_samples, n_features, n_classes)
st.subheader("👀 Vista previa del Dataset")
st.write(df.head())

# Separar en train/test
X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelos
resultados = entrenar_modelos(X_train, X_test, y_train, y_test, k, depth, c)

# Mostrar resultados
accuracies = {}
for nombre, res in resultados.items():
    st.subheader(f"🔹 {nombre}")
    st.write(f"**Accuracy:** {res['accuracy']:.2f}")
    st.json(res["report"])

    accuracies[nombre] = res["accuracy"]

    # Matriz de confusión
    cm = confusion_matrix(y_test, res["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Matriz de Confusión - {nombre}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

# Comparación de modelos
st.subheader("📊 Comparación de Accuracy")
fig, ax = plt.subplots()
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="Set2", ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
st.pyplot(fig)
