import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 🎀 Estilo
st.set_page_config(page_title="EDA + Modelos IA", layout="wide")
st.markdown("<h1 style='text-align:center; color:#e75480;'>🌸 Machine Learning Girly App 🌸</h1>", unsafe_allow_html=True)

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
# Lógica de Carga de Datos (Nueva)
# ================================
st.subheader("📁 Cargar tu propio archivo CSV")
uploaded_file = st.file_uploader("Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer el archivo CSV si se subió uno
    df = pd.read_csv(uploaded_file)
    st.success("✅ ¡Archivo cargado exitosamente!")
else:
    # Generación de datos simulados si no hay archivo
    st.info("ℹ️ No se ha subido ningún archivo. Se usarán datos simulados.")
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4, 
        n_redundant=0, n_classes=3, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(1, 7)])
    df["target"] = y

# ================================
# Selección de la columna Target
# ================================
st.markdown("---")
st.subheader("🎯 Seleccionar la columna objetivo (Target)")
if "target" in df.columns:
    target_column = st.selectbox(
        "Elige la columna que deseas predecir:",
        options=df.columns,
        index=df.columns.get_loc("target")
    )
else:
    target_column = st.selectbox(
        "Elige la columna que deseas predecir:",
        options=df.columns
    )

# ================================
# Split
# ================================
X = df.drop(columns=[target_column])
y = df[target_column]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
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
    sns.countplot(x=target_column, data=df, palette="pastel", ax=ax)
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
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    plot_tree(
        tree,
        filled=True,
        feature_names=feature_names, # Se usa la lista de nombres de las características
        class_names=[str(c) for c in np.unique(y)],
        rounded=True,
        ax=ax
    )
    
    st.pyplot(fig)

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
