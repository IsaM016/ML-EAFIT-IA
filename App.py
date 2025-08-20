import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz
import pydotplus
from PIL import Image
import io

# ===========================
# 🎀 Configuración inicial
# ===========================
st.set_page_config(page_title="Clasificación con ML", page_icon="🌸", layout="wide")
st.title("🌸 Clasificación Interactiva con ML 🌸")
st.markdown("Experimenta con **KNN** y **Árbol de Decisión**, visualiza datos y explora el árbol 🌳✨")

# ===========================
# 🎲 Generar dataset simulado
# ===========================
X, y = make_classification(
    n_samples=300, n_features=6, n_informative=4, n_classes=3,
    n_clusters_per_class=1, random_state=42
)
X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 7)])
y = pd.Series(y, name="Target")

# ===========================
# 🎨 Sidebar - Parámetros
# ===========================
st.sidebar.header("⚙️ Parámetros del modelo")
test_size = st.sidebar.slider("Tamaño del test (%)", 10, 50, 20, step=5) / 100

# KNN
st.sidebar.subheader("KNN")
n_neighbors = st.sidebar.slider("Número de vecinos (k)", 1, 15, 5)

# Árbol
st.sidebar.subheader("Árbol de Decisión")
max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)
criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy"])

# ===========================
# 📊 Separar dataset
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# ===========================
# 🤖 Entrenamiento de modelos
# ===========================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "modelo": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred)
    }

# ===========================
# 📌 Tabs
# ===========================
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Modelos", "🌳 Árbol de Decisión"])

# ----------- TAB 1 EDA -----------
with tab1:
    st.subheader("🔍 Exploratory Data Analysis")
    st.write("Primeras filas del dataset:")
    st.dataframe(pd.concat([X, y], axis=1).head())

    # Distribución de clases
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette="pastel", ax=ax)
    ax.set_title("Distribución de Clases")
    st.pyplot(fig)

    # Correlaciones
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Mapa de Correlación")
    st.pyplot(fig)

# ----------- TAB 2 MODELOS -----------
with tab2:
    st.subheader("📈 Resultados de Modelos")
    for name, result in results.items():
        st.markdown(f"### {name}")
        st.write(f"**Accuracy:** {result['accuracy']:.2f}")

        # Reporte
        st.write("Reporte de Clasificación:")
        st.json(result["report"])

        # Matriz de confusión
        fig, ax = plt.subplots()
        sns.heatmap(result["confusion"], annot=True, fmt="d", cmap="pink", cbar=False, ax=ax)
        ax.set_title(f"Matriz de Confusión - {name}")
        st.pyplot(fig)

# ----------- TAB 3 ÁRBOL -----------
with tab3:
    st.subheader("🌳 Visualización del Árbol de Decisión")

    dot_data = export_graphviz(
        results["Árbol de Decisión"]["modelo"],
        out_file=None,
        feature_names=X.columns,
        class_names=[str(c) for c in np.unique(y)],
        filled=True,
        rounded=True,
        special_characters=True
    )

    # Convertir a imagen con pydotplus
    graph = pydotplus.graph_from_dot_data(dot_data)
    png_bytes = graph.create_png()
    image = Image.open(io.BytesIO(png_bytes))

    st.image(image, caption="Árbol de Decisión con Nodos y Ramas 🌸", use_container_width=True)
