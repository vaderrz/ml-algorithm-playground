import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

st.set_page_config(page_title="ML Algorithm Playground", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Algorithm Playground")
st.write("Makine öğrenmesi algoritmalarının parametrelerini değiştirerek veriler üzerindeki etkilerini görselleştirin!")

# Sol Menü - Görev ve Model Seçimi
task_type = st.sidebar.selectbox("Görev Türü (Task)", ["Sınıflandırma (Classification)", "Kümeleme (Clustering)"])

if task_type == "Sınıflandırma (Classification)":
    st.sidebar.subheader("Veri ve Model Ayarları")
    classifier_name = st.sidebar.selectbox("Algoritma", ["Random Forest", "SVM", "KNN"])
    
    # Hiperparametreler
    n_samples = st.sidebar.slider("Örnek Sayısı", 100, 1000, 300)
    
    if classifier_name == "Random Forest":
        max_depth = st.sidebar.slider("Max Depth", 1, 15, 3)
        model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        model = SVC(C=C)
    else:
        k_neighbors = st.sidebar.slider("K Değeri (KNN)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k_neighbors)

    # Veri Üretme
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    st.subheader(f"📊 Model: {classifier_name} | Test Başarısı (Accuracy): `{acc:.2f}`")

    # Görselleştirme
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title("Veri Dağılımı ve Sınıflandırma")
    st.pyplot(fig)

else:
    st.sidebar.subheader("Kümeleme Ayarları")
    n_clusters = st.sidebar.slider("Küme Sayısı (K)", 2, 8, 3)
    
    X, _ = make_blobs(n_samples=300, centers=n_clusters, cluster_std=0.60, random_state=42)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)
    
    st.subheader(f"📊 K-Means Kümeleme (K = {n_clusters})")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Merkezler')
    ax.legend()
    st.pyplot(fig)
