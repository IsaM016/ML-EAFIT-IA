import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
st.set_page_config(page_title="🌸 Clasificación Girly + IA 🌸", layout="wide")

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
def entrenar_modelos(X_train, X_test, y_train, y_test, modelos):
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
# APP
# -------------------------------
st.title("🌸✨ Clasificación Dinámica con IA ✨🌸")
st.markdown("Explora un dataset, prueba modelos de **IA/ML** y disfruta de gráficos interactivos 💖")

# ---- Sidebar ----
st.sidebar.header("📌 Configuración Dataset")
tipo_datos = st.sidebar.radio("Elige tu dataset", ["Simulado", "Subir CSV"])

if tipo_datos == "Simulado":
    n_samples = st.sidebar.slider("Número de muestras", 100, 2000, 300, step=100)
    n_features = st.sidebar.slider("Número de características", 6, 15, 6)
    n_classes = st.sidebar.radio("Número de clases", [2, 3])
    df = generar_datos(n_samples, n_features, n_classes)
else:
    archivo = st.sidebar.file_uploader("📂 Sube tu CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
    else:
        st.warning("Por favor, sube un archivo CSV para continuar.")
        st.stop()

# Validación de target
if "Target" not in df.columns:
    st.error("El dataset debe contener una columna llamada **Target** para entrenar los modelos.")
    st.stop()

X = df.drop("Target", axis=1)
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---- Parámetros modelos ----
st.sidebar.header("⚙️ Selección de Modelos IA")
modelos_disp = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True)
}
modelos_selec = st.sidebar.multiselect("Elige modelos a probar", list(modelos_disp.keys()), ["KNN", "Árbol de Decisión", "Regresión Logística"])

# Entrenar
modelos = {m: modelos_disp[m] for m in modelos_selec}
resultados = entrenar_modelos(X_train, X_test, y_train, y_test, modelos)

# -------------------------------
# TABS
# -------------------------------
eda_tab, modelos_tab, comparacion_tab = st.tabs(["🔎 EDA", "🤖 Modelos", "📊 Comparación"])

# ---- EDA ----
with eda_tab:
    st.header("🔎 Exploración de Datos")
    st.write("👀 Vista previa del dataset")
    st.write(df.head())

    st.subheader("📊 Estadísticas descriptivas")
    st.write(df.describe())

    st.subheader("🎀 Distribución de la variable objetivo")
    fig = px.histogram(df, x="Target", color="Target", color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💖 Correlación entre variables")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdPu")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌸 Visualización dinámica (Scatter)")
    col_x = st.selectbox("Eje X", df.columns[:-1])
    col_y = st.selectbox("Eje Y", df.columns[:-1], index=1)
    fig = px.scatter(df, x=col_x, y=col_y, color="Target", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

# ---- Modelos ----
with modelos_tab:
    st.header("🤖 Resultados de Modelos IA")
    for nombre, res in resultados.items():
        st.subheader(f"✨ {nombre}")
        st.write(f"**Accuracy:** 🌸 {res['accuracy']:.2f}")
        with st.expander("Ver reporte de clasificación"):
            st.json(res["report"])

        # Matriz de confusión interactiva
        cm = confusion_matrix(y_test, res["y_pred"])
        z = cm[::-1]
        x = [f"Pred {i}" for i in range(cm.shape[0])]
        y_labels = [f"Real {i}" for i in range(cm.shape[0])]
        fig = ff.create_annotated_heatmap(z, x=x, y=y_labels[::-1], colorscale="Pinkyl")
        st.plotly_chart(fig, use_container_width=True)

# ---- Comparación ----
with comparacion_tab:
    st.header("📊 Comparación de Accuracy")
    accuracies = {nombre: res["accuracy"] for nombre, res in resultados.items()}
    acc_df = pd.DataFrame({"Modelo": list(accuracies.keys()), "Accuracy": list(accuracies.values())})
    fig = px.bar(acc_df, x="Modelo", y="Accuracy", color="Modelo", text="Accuracy", 
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)
