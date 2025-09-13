import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.title("Clustering ")
st.write("Upload your  dataset and explore clustering algorithms interactively.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=None, engine='python')
    st.write("### Data Preview", data.head())
    if 'quality' in data.columns:
        feature_set = data.drop("quality", axis=1)
    else:
        feature_set = data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_set)
    st.write("### Select Clustering Algorithm")
    algo = st.selectbox("Algorithm", ["KMeans", "DBSCAN", "GMM"])
    if algo == "KMeans":
        n_clusters = st.slider("Number of clusters", 2, 15, 4)
        km = KMeans(n_clusters=n_clusters, random_state=42)
        labels = km.fit_predict(X_scaled)
        st.write(f"KMeans inertia: {km.inertia_}")
    elif algo == "DBSCAN":
        eps = st.slider("EPS (distance threshold)", 0.1, 5.0, 1.0)
        min_samples = st.slider("Min samples", 1, 10, 3)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        st.write(f"DBSCAN clusters found: {n_clusters}")
    else:
        n_components = st.slider("Number of components", 2, 20, 4)
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(X_scaled)
        st.write(f"GMM components: {gmm.n_components}")
    st.write("### Cluster Visualization (first two features)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='rainbow', s=10)
    ax.set_xlabel(feature_set.columns[0])
    ax.set_ylabel(feature_set.columns[1])
    ax.set_title(f"{algo} Clustering")
    st.pyplot(fig)
    st.write("### Download Clustered Data")
    result = data.copy()
    result['cluster'] = labels
    st.download_button("Download CSV with clusters", result.to_csv(index=False), "clustered_wine.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin.")
