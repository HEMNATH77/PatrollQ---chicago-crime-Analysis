import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import gdown
import os
import numpy as np

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="Chicago Crime Analytics Dashboard",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🚔 Chicago Crime Pattern Analytics</h1>
    <p style='text-align:center; color:gray;'>
    Interactive dashboard for geographic & temporal crime clustering
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# GOOGLE DRIVE CSV DOWNLOAD
# -------------------------------
FILE_ID = "1AVf2d9g_6gySqjDsKjIEq0E1ChF04HJH"
CSV_FILE = "chicago_crime_cleaned.csv"

@st.cache_data(show_spinner=True)
def load_data():
    if not os.path.exists(CSV_FILE):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            CSV_FILE,
            quiet=False
        )
    return pd.read_csv(CSV_FILE)

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    with open("geo_scaler_v3.pkl", "rb") as f:
        geo_scaler = pickle.load(f)
    with open("kmeans_geo_model_v3.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    with open("dbscan_geo_model_v3.pkl", "rb") as f:
        dbscan_model = pickle.load(f)
    with open("hierarchical_geo_model_v3.pkl", "rb") as f:
        hierarchical_model = pickle.load(f)
    with open("temporal_scaler_v3.pkl", "rb") as f:
        temporal_scaler = pickle.load(f)
    with open("temporal_kmeans_model_v3.pkl", "rb") as f:
        temporal_model = pickle.load(f)
    return geo_scaler, kmeans_model, dbscan_model, hierarchical_model, temporal_scaler, temporal_model

df = load_data()
geo_scaler, kmeans_model, dbscan_model, hierarchical_model, temporal_scaler, temporal_model = load_models()

# -------------------------------
# TEMPORAL FEATURES
# -------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Hour"] = df["Date"].dt.hour
df["Day"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Filters & Navigation")

page = st.sidebar.radio(
    "📊 Select Analysis",
    [
        "Geographic Clustering (K-Means)",
        "DBSCAN Clustering",
        "Hierarchical Clustering",
        "Temporal Clustering - Time-based Patterns",
        "Temporal Clustering - Peak Times & Seasonal Trends",
        "Temporal Clustering - Crime Profiles",
        "Dimensionality Reduction - PCA Analysis",
        "Dimensionality Reduction - t-SNE / UMAP Visualization"
    ]
)

st.sidebar.markdown("---")

year = st.sidebar.selectbox("📅 Select Year", sorted(df["Year"].unique()))
crime_types = st.sidebar.multiselect(
    "🚨 Crime Type(s)",
    sorted(df["Primary Type"].unique())
)

filtered_df = df[df["Year"] == year]
if crime_types:
    filtered_df = filtered_df[filtered_df["Primary Type"].isin(crime_types)]

# -------------------------------
# KPI METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Crimes", f"{len(filtered_df):,}")

with col2:
    st.metric("Crime Types", filtered_df["Primary Type"].nunique())

with col3:
    st.metric("Year Selected", year)

st.markdown("---")

# -------------------------------
# GEOGRAPHIC SCALING
# -------------------------------
geo_features = ["Latitude", "Longitude", "X Coordinate", "Y Coordinate"]
X_geo_scaled = geo_scaler.transform(filtered_df[geo_features])

# -------------------------------
# K-MEANS
# -------------------------------
if page == "Geographic Clustering (K-Means)":
    st.subheader("📍 Geographic Crime Hotspots (K-Means)")
    filtered_df["cluster"] = kmeans_model.predict(X_geo_scaled)

    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="cluster",
        hover_data=["Primary Type", "Description"],
        zoom=9,
        color_continuous_scale="Turbo"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# DBSCAN
# -------------------------------
elif page == "DBSCAN Clustering":
    st.subheader("🚨 Density-Based Crime Detection (DBSCAN)")
    filtered_df["cluster"] = dbscan_model.fit_predict(X_geo_scaled)

    core = filtered_df[filtered_df["cluster"] != -1]
    noise = filtered_df[filtered_df["cluster"] == -1]

    fig = px.scatter_mapbox(
        core,
        lat="Latitude",
        lon="Longitude",
        color="cluster",
        zoom=9
    )

    fig.add_scattermapbox(
        lat=noise["Latitude"],
        lon=noise["Longitude"],
        mode="markers",
        marker=dict(size=4, color="gray", opacity=0.4),
        name="Noise"
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# HIERARCHICAL
# -------------------------------
elif page == "Hierarchical Clustering":
    st.subheader("🌳 Hierarchical Crime Structure")
    sample = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)
    X_sample = geo_scaler.transform(sample[geo_features])
    linkage = sch.linkage(X_sample, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(linkage, truncate_mode="level", p=5, ax=ax)
    st.pyplot(fig)

# -------------------------------
# TEMPORAL CLUSTERING
# -------------------------------
temporal_features = ["Hour", "Day", "Month"]
X_temp = temporal_scaler.transform(filtered_df[temporal_features])
filtered_df["temporal_cluster"] = temporal_model.predict(X_temp)

# -------------------------------
# TEMPORAL PAGES
# -------------------------------
if page == "Temporal Clustering - Time-based Patterns":
    st.subheader("🕒 Time-based Crime Behavior")
    summary = filtered_df.groupby("temporal_cluster")[temporal_features].mean().reset_index()
    st.dataframe(summary, use_container_width=True)

elif page == "Temporal Clustering - Peak Times & Seasonal Trends":
    st.subheader("📊 Peak Crime Activity")
    fig = px.bar(
        filtered_df.groupby("Hour").size().reset_index(name="Count"),
        x="Hour",
        y="Count"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Temporal Clustering - Crime Profiles":
    st.subheader("🧩 Crime Timing Profiles")
    fig = px.scatter(
        filtered_df.groupby("Primary Type")[["Hour", "Month"]].mean().reset_index(),
        x="Hour",
        y="Month",
        color="Primary Type"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# PCA
# -------------------------------
elif page == "Dimensionality Reduction - PCA Analysis":
    from sklearn.decomposition import PCA
    st.subheader("🧠 Principal Component Analysis")

    numeric = filtered_df.select_dtypes(include=["int64", "float64"]).dropna()
    X_scaled = StandardScaler().fit_transform(numeric)

    pca = PCA(n_components=3)
    pca.fit(X_scaled)

    st.success(f"Variance Explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    fig = px.bar(
        x=["PC1", "PC2", "PC3"],
        y=pca.explained_variance_ratio_ * 100
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# t-SNE / UMAP
# -------------------------------
elif page == "Dimensionality Reduction - t-SNE / UMAP Visualization":
    from sklearn.manifold import TSNE
    import umap

    st.subheader("🎨 High-Dimensional Visualization")
    method = st.radio("Select Method", ["t-SNE", "UMAP"], horizontal=True)

    numeric = filtered_df.select_dtypes(include=["int64", "float64"]).dropna()
    X_scaled = StandardScaler().fit_transform(numeric)

    embed = TSNE(n_components=2).fit_transform(X_scaled) if method == "t-SNE" \
        else umap.UMAP().fit_transform(X_scaled)

    embed_df = pd.DataFrame(embed, columns=["X", "Y"])
    embed_df["Primary Type"] = filtered_df["Primary Type"].values[:len(embed_df)]

    fig = px.scatter(embed_df, x="X", y="Y", color="Primary Type")
    st.plotly_chart(fig, use_container_width=True)
