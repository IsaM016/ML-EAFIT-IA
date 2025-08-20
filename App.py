import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus
from PIL import Image
import io

# 🎀 Estilo
st.set_page_config(page_title="EDA + Modelos IA", layout="wide")
st.markdown("<h1 style='text-align:center; color:#e75480;'>🌸 Machine Learning Girly App 🌸</h1>", unsafe_allow_html=True)

# ================================
# Generación de datos simulados
# ================================
X, y = make_classification(
    n_samples=300, n_features=6, n_informative=4, 
    n_redundant=0, n_classes=3, random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
df["target"] = y

# ================================
# Sidebar dinámico
# ================================
st.sidebar.header("⚙️ Parámetros")

test_size = st.sidebar.slider("Proporción de test", 0.1, 0.5, 0.3, step=0.05)

# Parámetros KNN
st.sidebar.subheader("🔮 KNN")
n_neighbors = st.sidebar.slider("Número de vecinos", 1, 15, 5)

# Parámetros Árbol
st.sidebar.subheader("🌳 Árbol de Decisión")
max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)
criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy", "log_loss"])

# ================================
# Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=test_size, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# Tabs
# ================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Modelos", "🌳 Árbol Visual", "📈 Métricas"])

# ================================
# EDA
# ================================
with tab1:
    st.subheader("📊 Exploratory Data Analysis")
    st.dataframe(df.head())

    st.markdown("### Distribución de Clases")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, palette="pastel", ax=ax)
    st.pyplot(fig)

    st.markdown("### Heatmap de Correlaciones")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ================================
# Modelos
# ================================
resultados = {}

with tab2:
    # KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    resultados["KNN"] = accuracy_score(y_test, y_pred_knn)

    # Árbol
    tree = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    resultados["Árbol"] = accuracy_score(y_test, y_pred_tree)

    st.subheader("📊 Resultados")
    for modelo, acc in resultados.items():
        st.markdown(f"**{modelo}:** {acc:.2f}")

# ================================
# Árbol de Decisión Gráfico
# ================================
with tab3:
    st.subheader("🌳 Visualización del Árbol de Decisión")
    
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=[f"feature_{i}" for i in range(1, 7)],
        class_names=[str(c) for c in np.unique(y)],
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    graph = pydotplus.graph_from_dot_data(dot_data)
    png_bytes = graph.create_png()
    image = Image.open(io.BytesIO(png_bytes))
    st.image(image, caption="Árbol de Decisión", use_container_width=True)

# ================================
# Métricas
# ================================
with tab4:
    st.subheader("📈 Reporte de Clasificación - Árbol de Decisión")
    report = classification_report(y_test, y_pred_tree, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("📈 Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt="d", cmap="Purples", ax=ax)
    st.pyplot(fig)
