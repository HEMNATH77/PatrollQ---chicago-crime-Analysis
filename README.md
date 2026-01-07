# 🚔 Chicago Crime Pattern Analytics Dashboard

An interactive **Streamlit dashboard** for analyzing and visualizing **crime patterns in Chicago** using **machine learning clustering techniques**. The project explores **geographic hotspots**, **temporal crime behaviors**, and **high-dimensional data patterns** through an intuitive, dashboard-style UI.

---

## 📌 Project Overview

This application leverages **unsupervised machine learning** to uncover hidden patterns in Chicago crime data. It combines:

* 📍 **Geographic clustering** (hotspot detection)
* 🕒 **Temporal clustering** (time, day, seasonal trends)
* 📊 **Dimensionality reduction** for pattern visualization

The dashboard is designed for:

* Data science projects
* Academic submissions
* Internship / interview demonstrations
* Exploratory crime data analysis

---

## 🧠 Techniques & Models Used

### 🔹 Clustering Algorithms

* **K-Means** – Geographic crime hotspots
* **DBSCAN** – Density-based crime detection (handles noise & outliers)
* **Hierarchical Clustering** – Crime zone hierarchy
* **Temporal K-Means** – Time-based crime behavior patterns

### 🔹 Dimensionality Reduction

* **PCA (Principal Component Analysis)** – Feature reduction & variance explanation
* **t-SNE** – Local neighborhood visualization
* **UMAP** – Global + local structure visualization

---

## 📊 Dashboard Features

* 🎛️ Interactive sidebar filters (Year & Crime Type)
* 📈 KPI metrics (Total crimes, crime types, selected year)
* 🗺️ Interactive crime maps (Mapbox)
* 🕒 Hourly, weekly & seasonal crime trends
* 🧩 Crime profile comparison by time
* 🎨 High-dimensional 2D projections (t-SNE / UMAP)

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **Streamlit** – Web app framework
* **Pandas & NumPy** – Data processing
* **Scikit-learn** – ML models & preprocessing
* **Plotly** – Interactive visualizations
* **Matplotlib & SciPy** – Hierarchical clustering
* **gdown** – Reliable Google Drive data loading

---

## 📁 Project Structure

```
chicago_crime_pattern/
│
├── app.py
├── README.md
├── chicago_crime_cleaned.csv   # Auto-downloaded
│
├── geo_scaler_v3.pkl
├── kmeans_geo_model_v3.pkl
├── dbscan_geo_model_v3.pkl
├── hierarchical_geo_model_v3.pkl
├── temporal_scaler_v3.pkl
├── temporal_kmeans_model_v3.pkl
```

> ⚠️ All `.pkl` files must be present in the root directory for the app to run.

---

## 🚀 How to Run the App

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/chicago-crime-pattern-analytics.git
cd chicago-crime-pattern-analytics
```

### 2️⃣ Install Dependencies

```bash
pip install streamlit pandas plotly scikit-learn matplotlib scipy gdown umap-learn
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

The app will automatically download the dataset from Google Drive on first run.

---

## 📂 Dataset

* Source: **Chicago Crime Data (public dataset)**
* Cleaned and preprocessed for clustering
* Loaded securely via **Google Drive + gdown** to avoid HTTP errors

---

## 💡 Key Insights Enabled

* Identification of high-crime geographic zones
* Understanding peak crime hours and seasonal patterns
* Differentiation of crime types based on temporal behavior
* Visualization of complex crime data in reduced dimensions

---
