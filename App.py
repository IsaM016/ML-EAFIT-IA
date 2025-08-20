import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import graphviz

# ----------------- CONFIG STREAMLIT -----------------
st.set_page_config(page_title="🌸 ML Playground", layout="wide")
st.title("🌸 Machine Learning Playground")
st.markdown("Explora un dataset simulado, haz un **EDA girly 💖** y entrena clasificadores interactivos ✨")

# ----------------- CREAR DATASET -----------------
st.sidebar.header("⚙️ Parámetros del Dataset")

n_samples = st.sidebar.slider("Número de muestras", 100, 1000, 300, 50)
n_features = st.sidebar.slider("Número de características", 6, 12, 6, 1)
n_classes = st.sidebar.slider("Número de clases", 2, 5, 3, 1)

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=4,
    n_redundant=0,
    n_classes=n_classes,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
df["target"] = y

st.subheader("📊 Vista previa de los datos")
st.dataframe(df.head())

# ----------------- EDA -----------------
tab1, tab2, tab3 = st.tabs(["🌸 EDA", "🤖 Modelos", "🌳 Árbol de Decisión"])

with tab1:
    st.header("✨ Exploratory Data Analysis (EDA)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribución de Clases")
        fig, ax = plt.subplots()
        sns.countplot(x="target", data=df, palette="pink")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Mapa de calor de correlaciones")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
        st.pyplot(fig)

    st.success("💡 Consejo: explora bien tus datos antes de entrenar modelos.")

# ----------------- SPLIT DATA -----------------
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- PARÁMETROS MODELOS -----------------
st.sidebar.header("🤖 Parámetros de Modelos")

# KNN
st.sidebar.subheader("KNN")
n_neighbors = st.sidebar.slider("Número de vecinos", 1, 20, 5)

# Árbol de Decisión
st.sidebar.subheader("Árbol de Decisión")
criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy", "log_loss"])
max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
min_samples_split = st.sidebar.slider("Mínimo muestras split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Mínimo muestras hoja", 1, 20, 1)

# ----------------- ENTRENAMIENTO -----------------
modelos = {
    "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Árbol de Decisión": DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
}

resultados = {}

with tab2:
    st.header("🤖 Resultados de Modelos")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        resultados[nombre] = {
            "modelo": modelo,
            "accuracy": acc,
            "reporte": classification_report(y_test, y_pred, output_dict=True),
            "matriz": confusion_matrix(y_test, y_pred)
        }

        with st.expander(f"🌸 Resultados de {nombre}"):
            st.write(f"**Accuracy:** `{acc:.2f}` 🌟")

            # Matriz de confusión
            fig, ax = plt.subplots()
            sns.heatmap(resultados[nombre]["matriz"], annot=True, fmt="d", cmap="RdPu")
            st.pyplot(fig)

            st.json(classification_report(y_test, y_pred, output_dict=True))

with tab3:
    st.header("🌳 Visualización del Árbol de Decisión")
    if "Árbol de Decisión" in resultados:
        dot_data = export_graphviz(
            resultados["Árbol de Decisión"]["modelo"],
            out_file=None,
            feature_names=X.columns,
            class_names=[str(c) for c in np.unique(y)],
            filled=True,
            rounded=True,
            special_characters=True
        )
        st.graphviz_chart(dot_data)
    else:
        st.warning("⚠️ Entrena primero el modelo de Árbol de Decisión desde la pestaña de Modelos.")
