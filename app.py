# ============================================================
# NEXUSAI STUDIO — INTERACTIVE MACHINE LEARNING PLATFORM
# Final Project Assignment — Artificial Intelligence
# Author: Khoirul Anwar (682024023)
# Universitas Kristen Satya Wacana (UKSW)
# ============================================================

import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(__file__), ".streamlit")
os.environ["SCIKIT_LEARN_DATA"] = os.path.join(os.path.dirname(__file__), ".streamlit", "scikit_learn_data")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
import json
import urllib.request
import ssl

# Scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score,
)

# MongoDB
try:
    import pymongo
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    import certifi
    from dotenv import load_dotenv
    load_dotenv()
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="NexusAI Studio — Artificial Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Memuat Custom CSS dari file eksternal ---
def load_css():
    """Membaca file CSS dari root repository dan menerapkannya."""
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as file:
            css = file.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Menjalankan fungsi CSS
load_css()


def style_dark_figure(fig, ax=None):
    """Menerapkan styling dark modern bertema cyber/glassmorphism pada kanvas Matplotlib."""
    fig.patch.set_facecolor("#111827")
    fig.patch.set_alpha(0.0)
    axes = ax if isinstance(ax, (list, np.ndarray)) else [ax] if ax else fig.axes
    for a in axes:
        a.set_facecolor("#161F30")
        a.tick_params(colors="#94A3B8", labelsize=8.5)
        a.xaxis.label.set_color("#CBD5E1")
        a.yaxis.label.set_color("#CBD5E1")
        a.xaxis.label.set_fontsize(9)
        a.yaxis.label.set_fontsize(9)
        a.title.set_color("#F8FAFC")
        a.title.set_fontsize(10)
        for spine in a.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.0)
        a.grid(True, color="#334155", linestyle="--", linewidth=0.6, alpha=0.5)
        leg = a.get_legend()
        if leg:
            leg.get_frame().set_facecolor("#1E293B")
            leg.get_frame().set_edgecolor("#475569")
            for text in leg.get_texts():
                text.set_color("#F1F5F9")
                text.set_fontsize(8.5)


# ============================================================
# KONEKSI MONGODB ATLAS
# ============================================================
@st.cache_resource(show_spinner=False)
def _init_mongo_client(uri: str):
    """Membuka koneksi MongoClient aman dengan timeout 10 detik dan auto-fallback TLS."""
    clean_uri = uri.strip().strip('"').strip("'")
    # Coba 1: Menggunakan CA bundle dari certifi
    try:
        c = MongoClient(clean_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=10000)
        c.admin.command("ping")
        return c
    except Exception:
        # Coba 2: Menggunakan CA sertifikat bawaan sistem OS
        c = MongoClient(clean_uri, serverSelectionTimeoutMS=10000)
        c.admin.command("ping")
        return c


def get_mongo_client():
    """Mengambil koneksi MongoDB dengan fallback st.secrets dan .env, serta status diagnostik."""
    if not MONGO_AVAILABLE:
        return None, "Library pymongo/certifi belum terinstal"

    uri = None
    # 1. Coba dari st.secrets (Streamlit Cloud)
    try:
        if "MONGODB_URI" in st.secrets:
            uri = st.secrets["MONGODB_URI"]
    except Exception as e_sec:
        return None, f"Format secrets keliru: {e_sec}"

    # 2. Coba dari environment variable .env (Lokal)
    if not uri:
        uri = os.getenv("MONGODB_URI")

    if not uri:
        return None, "Secret MONGODB_URI belum disetel di Cloud"

    try:
        client = _init_mongo_client(str(uri))
        return client, "Connected"
    except Exception as e_conn:
        err = str(e_conn)
        short_err = "Koneksi Timeout (Coba Sambungkan)" if "timed out" in err.lower() else "Gagal Autentikasi / Cluster"
        return None, short_err


# ============================================================
# DATASET LOADERS (SKLEARN, UCI REPOSITORY, REAL-TIME API)
# ============================================================
@st.cache_data
def load_supervised_dataset(nama_dataset, task):
    """Load dataset Supervised Learning dari Sklearn, UCI, atau Real-Time CoinGecko."""
    if task == "Regression":
        if nama_dataset == "Diabetes":
            data = datasets.load_diabetes()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = ["target"]
        elif nama_dataset == "California Housing":
            data = datasets.fetch_california_housing()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = ["median_house_value"]
        elif nama_dataset == "Wine Quality (UCI)":
            try:
                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = ssl._create_unverified_context()
            url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            req = urllib.request.Request(url_wine, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                df = pd.read_csv(resp, sep=";")
            X = df.drop(columns=["quality"]).values
            y = df["quality"].values.astype(float)
            feature_names = list(df.drop(columns=["quality"]).columns)
            target_names = ["quality_score"]
        elif nama_dataset == "Live Crypto BTC (Real-Time)":
            try:
                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = ssl._create_unverified_context()
            url_btc = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily"
            req = urllib.request.Request(url_btc, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                data_b = json.loads(resp.read().decode())
            prices = data_b.get("prices", [])
            mcaps = data_b.get("market_caps", [])
            vols = data_b.get("total_volumes", [])
            n = min(len(prices), len(mcaps), len(vols))
            rows = []
            for i in range(n):
                rows.append([float(i + 1), float(mcaps[i][1])/1e9, float(vols[i][1])/1e9, float(prices[i][1])])
            df_btc = pd.DataFrame(rows, columns=["Day_Index", "Market_Cap_Billion_USD", "Total_Volume_Billion_USD", "BTC_Price_USD"])
            X = df_btc[["Day_Index", "Market_Cap_Billion_USD", "Total_Volume_Billion_USD"]].values
            y = df_btc["BTC_Price_USD"].values
            feature_names = ["Day_Index", "Market_Cap_Billion_USD", "Total_Volume_Billion_USD"]
            target_names = ["BTC_Price_USD"]
        else:
            raise ValueError(f"Dataset '{nama_dataset}' tidak dikenali.")
    else:
        if nama_dataset == "Iris":
            data = datasets.load_iris()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = list(data.target_names)
        elif nama_dataset == "Wine":
            data = datasets.load_wine()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = list(data.target_names)
        elif nama_dataset == "Breast Cancer":
            data = datasets.load_breast_cancer()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = list(data.target_names)
        elif nama_dataset == "Digits":
            data = datasets.load_digits()
            X, y = data.data, data.target
            feature_names = list(data.feature_names)
            target_names = [str(i) for i in np.unique(y)]
        elif nama_dataset == "Wine Quality (UCI)":
            try:
                ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ctx = ssl._create_unverified_context()
            url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            req = urllib.request.Request(url_wine, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                df = pd.read_csv(resp, sep=";")
            X = df.drop(columns=["quality"]).values
            y = df["quality"].values.astype(int)
            feature_names = list(df.drop(columns=["quality"]).columns)
            target_names = [f"Grade {i}" for i in sorted(np.unique(y))]
        else:
            raise ValueError(f"Dataset '{nama_dataset}' tidak dikenali.")

    return X, y, feature_names, target_names


@st.cache_data
def load_clustering_dataset(nama_dataset):
    """Load dataset untuk Unsupervised K-Means Clustering."""
    if nama_dataset == "Iris":
        iris = datasets.load_iris()
        return iris.data, list(iris.feature_names)
    elif nama_dataset == "Wholesale Customers (UCI)":
        try:
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl._create_unverified_context()
        url_w = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
        req = urllib.request.Request(url_w, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            df_w = pd.read_csv(resp)
        cols_cluster = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
        return df_w[cols_cluster].values, cols_cluster
    else:
        raise ValueError(f"Dataset '{nama_dataset}' tidak dikenali.")


def get_model_instance(nama_model, task):
    """Inisialisasi estimator Scikit-Learn / Neural Network MLP."""
    if task == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
            "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
    return models.get(nama_model)


# ============================================================
# SESSION STATE MANAGEMENT
# ============================================================
# Menentukan halaman awal aplikasi
if "active_nav" not in st.session_state:
    st.session_state.active_nav = "🧭 Overview Hub"

# Menyimpan pilihan navigasi sidebar
if "workspace_navigation" not in st.session_state:
    st.session_state.workspace_navigation = st.session_state.active_nav

# Fungsi untuk berpindah halaman
def navigate_to(page):
    st.session_state.workspace_navigation = page
    st.session_state.active_nav = page

if "sup_trained" not in st.session_state:
    st.session_state.sup_trained = False
if "cluster_trained" not in st.session_state:
    st.session_state.cluster_trained = False

# MongoDB Connection Check
mongo_client, mongo_status_msg = get_mongo_client()
is_mongo_online = mongo_client is not None

komentar_count = 0
if is_mongo_online:
    try:
        komentar_count = mongo_client["test_db"]["komentar_db"].count_documents({})
    except Exception:
        komentar_count = 0


# ============================================================
# SIDEBAR CONTROL & NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown(
        '<div style="padding: 10px 0 14px 0; display: flex; align-items: center; gap: 12px;">'
        '<span style="font-size: 1.8rem; line-height: 1;">🤖</span>'
        '<div>'
        '<div style="font-weight: 800; font-size: 1.15rem; letter-spacing: -0.3px; color: #F1F5F9; line-height: 1.2;">NexusAI Studio</div>'
        '<div style="font-size: 0.74rem; color: #94A3B8; margin-top: 3px;">Artificial Intelligence Platform</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # Identitas Pengembang (Berjarak & Rapi)
    st.markdown(
        '<div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 9px; padding: 10px 13px; margin: 6px 0 14px 0;">'
        '<div style="font-size: 0.66rem; font-weight: 700; color: #94A3B8; letter-spacing: 0.5px;">👤 DEVELOPER IDENTITY</div>'
        '<div style="font-size: 0.88rem; font-weight: 700; color: #F1F5F9; margin-top: 3px;">Khoirul Anwar</div>'
        '<div style="font-size: 0.73rem; color: #64748B; margin-top: 2px;">NIM: <b>682024023</b> • TAS AI</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='font-size: 0.86rem; font-weight: 700; color: #F1F5F9; margin-bottom: 8px;'>🧭 Platform Navigation</div>", unsafe_allow_html=True)
    nav_options = [
        "🧭 Overview Hub",
        "📈 Supervised Learning Studio",
        "🔵 Unsupervised Clustering Lab",
        "🗄️ MongoDB Cloud & Retrain",
        "💬 Community Feedback & Groq AI",
    ]
    # Navigasi utama
    selected_nav = st.radio(
        "Pilih Modul Workspace:",
        options=nav_options,
        key="workspace_navigation",
        label_visibility="collapsed"
    )
    st.session_state.active_nav = selected_nav

    st.divider()

    # Cloud Database Telemetry Badge (Berjarak & Rapi)
    st.markdown("<div style='font-size: 0.86rem; font-weight: 700; color: #F1F5F9; margin-bottom: 8px;'>📡 System Telemetry</div>", unsafe_allow_html=True)
    if is_mongo_online:
        st.markdown(
            '<div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 9px 12px; margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">'
            '<span style="color: #10B981; font-size: 0.95rem;">●</span>'
            '<div style="line-height: 1.35;">'
            '<div style="font-size: 0.8rem; font-weight: 700; color: #34D399;">MongoDB: Connected</div>'
            '<div style="font-size: 0.72rem; color: #94A3B8;">Cluster: test_db (Active)</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 9px 12px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px;">'
            f'<span style="color: #EF4444; font-size: 0.95rem;">●</span>'
            f'<div style="line-height: 1.35;">'
            f'<div style="font-size: 0.8rem; font-weight: 700; color: #F87171;">MongoDB: Disconnected</div>'
            f'<div style="font-size: 0.69rem; color: #94A3B8;">{mongo_status_msg}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("🔄 Hubungkan Ulang", key="btn_reconnect_mongo", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

    c_sb1, c_sb2 = st.columns(2)
    with c_sb1:
        st.caption(f"💬 **{komentar_count}** Feedback")
    with c_sb2:
        st.caption("🧠 **12** AI Models")

    st.divider()
    st.caption("NexusAI Studio v2.4 • TAS AI")


# ============================================================
# 1. HALAMAN: 🧭 OVERVIEW HUB (DASHBOARD UTAMA)
# ============================================================
if st.session_state.active_nav == "🧭 Overview Hub":
    # Hero Banner - Penjelasan Singkat Web & Keterangan Project
    st.markdown(
        '<div class="nexus-hero">'
        '<div class="nexus-hero-eyebrow">FINAL PROJECT · ARTIFICIAL INTELLIGENCE</div>'
        '<div class="nexus-hero-title">Platform Pembelajaran & Eksperimen Artificial Intelligence</div>'
        '<p class="nexus-hero-description">'
        'Aplikasi web interaktif untuk eksplorasi dan simulasi pemodelan Machine Learning (Supervised Learning & Unsupervised Clustering), integrasi database cloud MongoDB Atlas untuk retraining model kontinu, serta analisis feedback publik cerdas dengan AI.'
        '</p>'
        '<div class="nexus-hero-meta">'
        '<span>👤 <b>Pengembang:</b> Khoirul Anwar (682024023)</span>'
        '<span>📚 <b>Mata Kuliah:</b> Artificial Intelligence</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # 4 Quick KPI Metrics (Hierarki visual konsisten & tanpa teks terpotong)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            label="AI Models",
            value="12",
            delta="11 Supervised + K-Means"
        )

    with m2:
        st.metric(
            label="Clustering Lab",
            value="K-Means",
            delta="Centroid & Silhouette"
        )

    with m3:
        st.metric(
            label="Datasets",
            value="8 Data",
            delta="Sklearn, UCI & API"
        )

    with m4:
        st.metric(
            label="Cloud Database",
            value="MongoDB",
            delta="Atlas M0 (test_db)" if is_mongo_online else "Status: Offline"
        )

    # 4 Interactive Workspace Modules
    st.subheader("🚀 Platform Workspace Modules")
    st.caption("Pilih modul analitik Artificial Intelligence di bawah ini atau klik info untuk melihat penjelasan:")

    c_m1, c_m2 = st.columns(2)

    with c_m1:
        with st.container(border=True):
            st.markdown(
                '<div style="height: 3px; background: linear-gradient(90deg, #3B82F6, #8B5CF6); margin: -12px -12px 10px -12px; border-radius: 6px 6px 0 0;"></div>'
                '<span class="nexus-badge badge-blue">Supervised Learning</span> '
                '<span class="nexus-badge badge-purple">Deep Learning MLP</span>',
                unsafe_allow_html=True
            )
            st.markdown("#### 📈 Supervised Learning Studio")
            with st.popover("ℹ️ Penjelasan Modul", use_container_width=True):
                st.markdown("##### 📈 Supervised Learning Studio")
                st.markdown(
                    "Modul pemodelan prediktif interaktif untuk tugas **Regression** dan **Classification**:\n"
                    "- **11 Algoritma**: Linear, Ridge, Lasso, Logistic, KNN, Decision Tree, Random Forest, SVM, dan Neural Network (MLP).\n"
                    "- **Visualisasi Diagnostik**: Actual vs Predicted, Residual Distribution, Confusion Matrix Heatmap, dan Feature Importance.\n"
                    "- **Evaluasi Performa**: Scorecard metrik (R², MAE, MSE, RMSE, Akurasi, F1) dan validasi 5-Fold Cross Validation.\n"
                    "- **Dukungan Dataset**: Scikit-Learn, UCI Machine Learning, dan Live Crypto Bitcoin API."
                )
            st.button(
                "Buka Supervised Studio ➔",
                key="btn_to_sup",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=("📈 Supervised Learning Studio",)
            )

    with c_m2:
        with st.container(border=True):
            st.markdown(
                '<div style="height: 3px; background: linear-gradient(90deg, #A855F7, #EC4899); margin: -12px -12px 10px -12px; border-radius: 6px 6px 0 0;"></div>'
                '<span class="nexus-badge badge-purple">Unsupervised Learning</span> '
                '<span class="nexus-badge badge-blue">Centroid Analytics</span>',
                unsafe_allow_html=True
            )
            st.markdown("#### 🔵 Unsupervised Clustering Lab")
            with st.popover("ℹ️ Penjelasan Modul", use_container_width=True):
                st.markdown("##### 🔵 Unsupervised Clustering Lab")
                st.markdown(
                    "Modul eksplorasi pola data tanpa label berbasis **K-Means Clustering**:\n"
                    "- **Parameter Interaktif**: Penyesuaian jumlah klaster ($K=2..8$) dan konfigurasi *random seed*.\n"
                    "- **Proyeksi Fitur 2D**: Pilihan pasangan fitur sumbu X dan Y pada scatter plot visual.\n"
                    "- **Centroid Analytics**: Penandaan titik pusat (centroid) klaster berwarna merah kontras di kanvas.\n"
                    "- **Validasi Klaster**: Evaluasi metrik **Silhouette Score** dan **Inertia** untuk menguji kepadatan kelompok.\n"
                    "- **Dataset**: Iris Flower dan UCI Wholesale Customers."
                )
            st.button(
                "Buka Clustering Lab ➔",
                key="btn_to_cluster",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=("🔵 Unsupervised Clustering Lab",)
            )

    c_m3, c_m4 = st.columns(2)

    with c_m3:
        with st.container(border=True):
            st.markdown(
                '<div style="height: 3px; background: linear-gradient(90deg, #10B981, #06B6D4); margin: -12px -12px 10px -12px; border-radius: 6px 6px 0 0;"></div>'
                '<span class="nexus-badge badge-green">Cloud Database</span> '
                '<span class="nexus-badge badge-amber">Continuous Retrain</span>',
                unsafe_allow_html=True
            )
            st.markdown("#### 🗄️ MongoDB Cloud Database & Retrain")
            with st.popover("ℹ️ Penjelasan Modul", use_container_width=True):
                st.markdown("##### 🗄️ MongoDB Cloud Database & Retrain")
                st.markdown(
                    "Modul integrasi database cloud **MongoDB Atlas** terhubung langsung ke cluster cloud (`test_db.iris_db`):\n"
                    "- **Operasi CRUD Real-Time**: Input dokumen data sampel Iris baru dan hapus record dari database cloud.\n"
                    "- **Cloud Explorer**: Menampilkan seluruh data tersimpan dalam tabel dinamis.\n"
                    "- **Continuous Retraining Pipeline**: Menggabungkan 150 data bawaan Scikit-Learn dengan data kustom MongoDB untuk melatih ulang model secara otomatis."
                )
            st.button(
                "Buka Database & Retrain ➔",
                key="btn_to_db",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=("🗄️ MongoDB Cloud & Retrain",)
            )

    with c_m4:
        with st.container(border=True):
            st.markdown(
                '<div style="height: 3px; background: linear-gradient(90deg, #F59E0B, #EF4444); margin: -12px -12px 10px -12px; border-radius: 6px 6px 0 0;"></div>'
                '<span class="nexus-badge badge-amber">NLP Sentiment</span> '
                '<span class="nexus-badge badge-green">Groq Llama 3.1</span>',
                unsafe_allow_html=True
            )
            st.markdown("#### 💬 Community Feedback & Groq AI")
            with st.popover("ℹ️ Penjelasan Modul", use_container_width=True):
                st.markdown("##### 💬 Community Feedback & Groq AI")
                st.markdown(
                    "Modul interaksi publik cerdas terintegrasi cloud database dan Large Language Model:\n"
                    "- **Form Ulasan & Rating**: Penilaian bintang 1–5 tersimpan di MongoDB Atlas (`komentar_db`).\n"
                    "- **Analisis Sentimen Instan**: Klasifikasi otomatis sentimen Positif, Netral, atau Negatif.\n"
                    "- **Groq LPU LLM Engine**: Ringkasan eksekutif wawasan publik otomatis didukung oleh model Llama 3.1 8B Instant."
                )
            st.button(
                "Buka Feedback & Groq AI ➔",
                key="btn_to_fb",
                type="primary",
                use_container_width=True,
                on_click=navigate_to,
                args=("💬 Community Feedback & Groq AI",)
            )


# ============================================================
# 2. HALAMAN: 📈 SUPERVISED LEARNING STUDIO
# ============================================================
elif st.session_state.active_nav == "📈 Supervised Learning Studio":
    st.markdown("## 📈 Supervised Learning Studio")
    st.caption("Eksplorasi pemodelan terpadu untuk Regression dan Classification menggunakan algoritma klasik dan Deep Learning (MLP).")

    # Command Bar
    with st.container(border=True):
        st.markdown("**⚙️ Experiment Configuration & Command Bar**")
        cb1, cb2, cb3 = st.columns([1.5, 2, 2.5])
        with cb1:
            task_type = st.radio("Task Mode", ["Regression", "Classification"], horizontal=True)
        with cb2:
            if task_type == "Regression":
                ds_list = ["Diabetes", "California Housing", "Wine Quality (UCI)", "Live Crypto BTC (Real-Time)"]
            else:
                ds_list = ["Iris", "Wine", "Breast Cancer", "Digits", "Wine Quality (UCI)"]
            selected_ds = st.selectbox("Select Dataset", ds_list)
        with cb3:
            if task_type == "Regression":
                md_list = [
                    "Linear Regression", "Ridge Regression", "Lasso Regression",
                    "Decision Tree", "Random Forest", "Neural Network (MLP)"
                ]
            else:
                md_list = [
                    "Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree",
                    "Random Forest", "Support Vector Machine (SVM)", "Neural Network (MLP)"
                ]
            selected_md = st.selectbox("Select Algorithm", md_list)

        cb4, cb5, cb6 = st.columns([2, 2, 2])
        with cb4:
            ts_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
        with cb5:
            r_seed = st.number_input("Random Seed", 0, 9999, 42, 1)
        with cb6:
            st.write("")
            btn_run_sup = st.button("🚀 Train Model", type="primary", use_container_width=True)

    # State data holder
    if "sup_data" not in st.session_state:
        st.session_state.sup_data = None

    if btn_run_sup:
        try:
            with st.spinner(f"Memuat dataset {selected_ds}..."):
                X, y, f_names, t_names = load_supervised_dataset(selected_ds, task_type)

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts_size, random_state=int(r_seed))
            model = get_model_instance(selected_md, task_type)

            with st.spinner(f"Melatih model {selected_md}..."):
                t_start = time.time()
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                t_duration = time.time() - t_start

            # Metrics
            met_dict = {}
            if task_type == "Regression":
                met_dict["MAE"] = mean_absolute_error(y_te, y_pred)
                met_dict["MSE"] = mean_squared_error(y_te, y_pred)
                met_dict["RMSE"] = np.sqrt(met_dict["MSE"])
                met_dict["R2"] = r2_score(y_te, y_pred)
                met_dict["MAPE"] = mean_absolute_percentage_error(y_te, y_pred)
            else:
                met_dict["Accuracy"] = accuracy_score(y_te, y_pred)
                met_dict["Precision"] = precision_score(y_te, y_pred, average="weighted", zero_division=0)
                met_dict["Recall"] = recall_score(y_te, y_pred, average="weighted", zero_division=0)
                met_dict["F1"] = f1_score(y_te, y_pred, average="weighted", zero_division=0)

            # 5-Fold Cross Validation
            with st.spinner("Menghitung 5-Fold Cross Validation..."):
                cv_res = cross_val_score(model, X, y, cv=5)
                cv_tuple = (cv_res.mean(), cv_res.std())

            st.session_state.sup_data = {
                "task": task_type,
                "dataset": selected_ds,
                "model_name": selected_md,
                "model": model,
                "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te, "y_pred": y_pred,
                "metrics": met_dict,
                "cv": cv_tuple,
                "features": f_names,
                "targets": t_names,
                "train_time": t_duration,
            }
            st.session_state.sup_trained = True
            st.success(f"✅ Model **{selected_md}** berhasil dilatih dalam {t_duration:.3f} detik!")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat training: {e}")

    # Results Display
    if st.session_state.sup_trained and st.session_state.sup_data:
        sd = st.session_state.sup_data
        st.divider()

        # Tabs: Diagnostics & Evaluation
        res_tab1, res_tab2 = st.tabs(["📊 Diagnostic Plots & Visualizations", "📈 KPI Evaluation Metrics & Scorecard"])

        with res_tab1:
            st.markdown(f"#### 🔍 Visual Analytics — {sd['model_name']} ({sd['dataset']})")
            if sd["task"] == "Regression":
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.markdown("**Scatter: Actual vs Predicted**")
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    ax.scatter(sd["y_te"], sd["y_pred"], alpha=0.85, color="#38BDF8", edgecolor="#0284C7", s=50)
                    min_v = min(np.min(sd["y_te"]), np.min(sd["y_pred"]))
                    max_v = max(np.max(sd["y_te"]), np.max(sd["y_pred"]))
                    ax.plot([min_v, max_v], [min_v, max_v], color="#F43F5E", linestyle="--", lw=2, label="Perfect Line")
                    ax.set_xlabel("Aktual")
                    ax.set_ylabel("Prediksi")
                    ax.legend()
                    style_dark_figure(fig, ax)
                    st.pyplot(fig, transparent=True)
                    plt.close(fig)

                with p_c2:
                    st.markdown("**Perbandingan Index Prediksi (50 Titik)**")
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    n_pts = min(50, len(sd["y_te"]))
                    ax.plot(range(n_pts), np.array(sd["y_te"])[:n_pts], marker="o", label="Actual", color="#38BDF8", linewidth=1.8, markersize=5)
                    ax.plot(range(n_pts), np.array(sd["y_pred"])[:n_pts], marker="x", label="Predicted", color="#F59E0B", linewidth=1.8, markersize=6)
                    ax.set_xlabel("Index Sampel")
                    ax.set_ylabel("Nilai")
                    ax.legend()
                    style_dark_figure(fig, ax)
                    st.pyplot(fig, transparent=True)
                    plt.close(fig)

                st.markdown("**Residual Distribution Plot**")
                fig, ax = plt.subplots(figsize=(10, 3.5))
                res = np.array(sd["y_te"]) - np.array(sd["y_pred"])
                ax.scatter(sd["y_pred"], res, alpha=0.85, color="#10B981", edgecolor="#047857", s=50)
                ax.axhline(0, color="#F43F5E", linestyle="--", lw=1.8)
                ax.set_xlabel("Fitted Values (Prediksi)")
                ax.set_ylabel("Residuals")
                style_dark_figure(fig, ax)
                st.pyplot(fig, transparent=True)
                plt.close(fig)

            else:
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    st.markdown("**Confusion Matrix Heatmap**")
                    cm = confusion_matrix(sd["y_te"], sd["y_pred"])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="mako",
                                xticklabels=sd["targets"], yticklabels=sd["targets"], ax=ax,
                                cbar=True, annot_kws={"size": 11, "weight": "bold", "color": "#FFFFFF"})
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    style_dark_figure(fig, ax)
                    ax.grid(False)
                    st.pyplot(fig, transparent=True)
                    plt.close(fig)

                with p_c2:
                    st.markdown("**Feature Importance Analysis**")
                    md = sd["model"]
                    if hasattr(md, "feature_importances_"):
                        imp = md.feature_importances_
                        idx = np.argsort(imp)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.barh(range(len(idx)), imp[idx], color="#818CF8", edgecolor="#6366F1")
                        ax.set_yticks(range(len(idx)))
                        ax.set_yticklabels([sd["features"][i] for i in idx])
                        ax.set_xlabel("Importance Value")
                        style_dark_figure(fig, ax)
                        st.pyplot(fig, transparent=True)
                        plt.close(fig)
                    elif hasattr(md, "coef_"):
                        coef = np.abs(md.coef_)
                        if coef.ndim > 1:
                            coef = coef.mean(axis=0)
                        idx = np.argsort(coef)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.barh(range(len(idx)), coef[idx], color="#818CF8", edgecolor="#6366F1")
                        ax.set_yticks(range(len(idx)))
                        ax.set_yticklabels([sd["features"][i] for i in idx])
                        ax.set_xlabel("Coefficient Magnitude")
                        style_dark_figure(fig, ax)
                        st.pyplot(fig, transparent=True)
                        plt.close(fig)
                    else:
                        st.info("ℹ️ Model ini (misal Neural Network MLP / KNN) tidak mengekspos fitur importance linier langsung.")

        with res_tab2:
            st.markdown(f"#### 🏆 Model Performance Scorecard — {sd['model_name']}")
            mets = sd["metrics"]
            if sd["task"] == "Regression":
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("📏 MAE", f"{mets['MAE']:.4f}")
                sc2.metric("📐 MSE", f"{mets['MSE']:.4f}")
                sc3.metric("📊 RMSE", f"{mets['RMSE']:.4f}")
                sc4.metric("🎯 R² Score", f"{mets['R2']:.4f}")
                sc5.metric("📉 MAPE", f"{mets['MAPE']*100:.2f}%")

                r2_val = mets["R2"]
                if r2_val >= 0.85:
                    st.success(f"🌟 **R² = {r2_val:.4f}** — Performa Model Sangat Baik ({r2_val*100:.1f}% variansi data terjelaskan).")
                elif r2_val >= 0.65:
                    st.info(f"✅ **R² = {r2_val:.4f}** — Performa Model Cukup Baik ({r2_val*100:.1f}% variansi).")
                else:
                    st.warning(f"⚠️ **R² = {r2_val:.4f}** — Performa Model Terbatas.")

            else:
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("🎯 Accuracy", f"{mets['Accuracy']*100:.2f}%")
                sc2.metric("🎪 Precision", f"{mets['Precision']*100:.2f}%")
                sc3.metric("🔍 Recall", f"{mets['Recall']*100:.2f}%")
                sc4.metric("⚖️ F1-Score", f"{mets['F1']*100:.2f}%")

                acc_val = mets["Accuracy"]
                if acc_val >= 0.90:
                    st.success(f"🌟 **Akurasi = {acc_val*100:.2f}%** — Model Sangat Presisi.")
                elif acc_val >= 0.75:
                    st.info(f"✅ **Akurasi = {acc_val*100:.2f}%** — Akurasi Cukup Baik.")
                else:
                    st.warning(f"⚠️ **Akurasi = {acc_val*100:.2f}%** — Performa Model Perlu Dioptimasi.")

                st.markdown("**Detailed Classification Report:**")
                clf_rep = classification_report(sd["y_te"], sd["y_pred"], target_names=sd["targets"], output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(clf_rep).transpose().style.format("{:.4f}"), use_container_width=True)

            st.divider()
            cv_m, cv_s = sd["cv"]
            cv_c1, cv_c2 = st.columns(2)
            cv_c1.metric("5-Fold Cross-Validation (Mean)", f"{cv_m:.4f}")
            cv_c2.metric("Cross-Validation (Std Dev)", f"±{cv_s:.4f}")

    else:
        st.info("💡 **Silakan pilih task, dataset, dan model di Command Bar di atas, lalu klik `🚀 Train Model`.**")


# ============================================================
# 3. HALAMAN: 🔵 UNSUPERVISED CLUSTERING LAB
# ============================================================
elif st.session_state.active_nav == "🔵 Unsupervised Clustering Lab":
    st.markdown("## 🔵 Unsupervised Clustering Lab (K-Means)")
    st.caption("Eksplorasi pengelompokan pola data berdimensi banyak menggunakan algoritma K-Means dan evaluasi Silhouette.")

    with st.container(border=True):
        st.markdown("**⚙️ Clustering Parameter Configuration**")
        cl_c1, cl_c2, cl_c3 = st.columns([2.5, 2.5, 2])
        with cl_c1:
            cl_dataset = st.selectbox("Select Clustering Dataset", ["Iris", "Wholesale Customers (UCI)"])
        with cl_c2:
            cl_k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=3)
        with cl_c3:
            cl_seed = st.number_input("Random State Seed", 0, 9999, 42, 1)

        btn_run_cluster = st.button("🚀 Run K-Means Clustering", type="primary", use_container_width=True)

    if "cluster_data_state" not in st.session_state:
        st.session_state.cluster_data_state = None

    if btn_run_cluster:
        try:
            with st.spinner(f"Memuat data {cl_dataset}..."):
                X_raw, feat_names = load_clustering_dataset(cl_dataset)

            with st.spinner(f"Menjalankan K-Means Clustering (K={cl_k})..."):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)
                km_model = KMeans(n_clusters=int(cl_k), random_state=int(cl_seed), n_init=10)
                cl_labels = km_model.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, cl_labels) if len(np.unique(cl_labels)) > 1 else float("nan")
                centers_orig = scaler.inverse_transform(km_model.cluster_centers_)

            st.session_state.cluster_data_state = {
                "dataset": cl_dataset,
                "K": cl_k,
                "X": X_raw,
                "features": feat_names,
                "labels": cl_labels,
                "centers": centers_orig,
                "silhouette": sil_score,
                "inertia": float(km_model.inertia_),
            }
            st.session_state.cluster_trained = True
            st.success(f"✅ K-Means Clustering selesai dengan Silhouette Score: **{sil_score:.4f}**!")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat clustering: {e}")

    if st.session_state.cluster_trained and st.session_state.cluster_data_state:
        cd = st.session_state.cluster_data_state
        st.divider()

        # Scorecard
        st.markdown(f"#### 📊 Cluster Quality Scorecard — {cd['dataset']}")
        q1, q2, q3 = st.columns(3)
        q1.metric("Selected Clusters (K)", str(cd["K"]))
        q2.metric("Silhouette Score", f"{cd['silhouette']:.4f}")
        q3.metric("Inertia (Within-Cluster Sum)", f"{cd['inertia']:.2f}")

        st.caption("ℹ️ Silhouette Score bernilai antara -1 hingga +1. Nilai yang semakin mendekati 1 menunjukkan struktur cluster yang terpisah secara optimal.")

        st.divider()

        # Visualisasi
        st.markdown(f"#### 🧩 2D Scatter Projection & Centroid Mapping — {cd['dataset']}")
        ax_c1, ax_c2 = st.columns(2)
        with ax_c1:
            axis_x = st.selectbox("Pilih Fitur Sumbu X", cd["features"], index=0, key="cl_axis_x")
        with ax_c2:
            default_y_idx = 1 if len(cd["features"]) > 1 else 0
            axis_y = st.selectbox("Pilih Fitur Sumbu Y", cd["features"], index=default_y_idx, key="cl_axis_y")

        ix = cd["features"].index(axis_x)
        iy = cd["features"].index(axis_y)

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(cd["X"][:, ix], cd["X"][:, iy], c=cd["labels"], cmap="tab10", s=65, alpha=0.85, edgecolor="#1E293B", linewidth=0.5)
        ax.scatter(cd["centers"][:, ix], cd["centers"][:, iy], marker="X", s=280, c="#EF4444", edgecolor="#FFFFFF", linewidth=2.2, label="Centroid (Pusat Cluster)", zorder=5)
        ax.set_xlabel(axis_x)
        ax.set_ylabel(axis_y)
        ax.set_title(f"K-Means Clustering Scatter Plot ({cd['dataset']})")
        ax.legend(loc="upper right")
        style_dark_figure(fig, ax)
        st.pyplot(fig, transparent=True)
        plt.close(fig)

        # Distribusi Data per Cluster
        st.markdown("#### 📦 Distribusi Jumlah Sampel per Cluster")
        counts = pd.Series(cd["labels"]).value_counts().sort_index()
        df_dist = pd.DataFrame({"Cluster ID": [f"Cluster {i}" for i in counts.index], "Total Sampel": counts.values})
        st.dataframe(df_dist, use_container_width=True, hide_index=True)

    else:
        st.info("💡 **Silakan pilih dataset dan jumlah cluster di atas, lalu klik `🚀 Run K-Means Clustering`.**")


# ============================================================
# 4. HALAMAN: 🗄️ MONGODB CLOUD & RETRAIN
# ============================================================
elif st.session_state.active_nav == "🗄️ MongoDB Cloud & Retrain":
    st.markdown("## 🗄️ MongoDB Cloud Database & Continuous Retrain")
    st.caption("Operasi database real-time dengan MongoDB Atlas (`test_db.iris_db`) dan Active Continuous Retraining.")

    if not is_mongo_online:
        st.error(
            "❌ **Koneksi MongoDB Atlas Tidak Aktif!**\n\n"
            "Pastikan:\n"
            "1. File `.streamlit/secrets.toml` atau `.env` berisi `MONGODB_URI` yang valid.\n"
            "2. IP Address `0.0.0.0/0` telah di-whitelist di Network Access dashboard MongoDB Atlas."
        )
        st.stop()

    client = mongo_client
    db = client["test_db"]
    iris_col = db["iris_db"]

    # Statistics Telemetry
    total_docs = iris_col.count_documents({})
    st.markdown("### 📊 Database Collection Statistics")
    db_c1, db_c2, db_c3 = st.columns(3)
    db_c1.metric("Total Dokumen Cloud", total_docs)
    db_c2.metric("Active Database", "test_db")
    db_c3.metric("Target Collection", "iris_db")

    st.divider()

    # Form Insert Data
    st.markdown("### ➕ Tambah Sampel Bunga Iris ke MongoDB Atlas")
    with st.form("form_add_iris"):
        fi_1, fi_2 = st.columns(2)
        with fi_1:
            sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1, 0.1)
            sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5, 0.1)
        with fi_2:
            pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4, 0.1)
            pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2, 0.1)
        species = st.selectbox("Species Label", ["setosa", "versicolor", "virginica"])

        btn_insert_iris = st.form_submit_button("💾 Simpan Data ke MongoDB Atlas", type="primary")
        if btn_insert_iris:
            iris_col.insert_one({
                "sepal_length": float(sl),
                "sepal_width": float(sw),
                "petal_length": float(pl),
                "petal_width": float(pw),
                "species": species,
                "timestamp": datetime.now(),
            })
            st.success("🎉 Data baru berhasil disimpan ke MongoDB Atlas!")
            st.rerun()

    st.divider()

    # Data Explorer & Delete
    st.markdown("### 📋 Data Explorer di MongoDB Atlas")
    docs = list(iris_col.find().sort("timestamp", -1))
    if not docs:
        st.info("📭 Belum ada sampel data di koleksi `iris_db`. Tambahkan sampel baru melalui formulir di atas.")
    else:
        df_docs = pd.DataFrame(docs)
        df_display = df_docs.copy()
        df_display["_id"] = df_display["_id"].astype(str)
        st.dataframe(df_display, use_container_width=True)

        st.markdown("#### 🗑️ Hapus Dokumen Berdasarkan ID")
        del_c1, del_c2 = st.columns([3, 1])
        with del_c1:
            del_id = st.selectbox("Pilih Document ID", df_docs["_id"].astype(str).tolist())
        with del_c2:
            st.write("")
            if st.button("🗑️ Hapus Dokumen", type="secondary"):
                iris_col.delete_one({"_id": ObjectId(del_id)})
                st.success("Dokumen berhasil dihapus!")
                st.rerun()

    st.divider()

    # Continuous Retrain Engine
    st.markdown("### 🧠 Continuous Active Model Retraining")
    st.caption("Latih ulang model **Random Forest Classifier** dengan menggabungkan 150 data bawaan scikit-learn dengan seluruh data tambahan dari MongoDB Atlas.")

    if st.button("🚀 Jalankan Retrain Model Sekarang", type="primary"):
        try:
            iris_orig = datasets.load_iris()
            X_orig = iris_orig.data
            y_orig = iris_orig.target

            df_mongo = pd.DataFrame(list(iris_col.find({}, {"_id": 0})))
            if df_mongo.empty:
                st.warning("⚠️ Tidak ada data MongoDB tambahan. Model akan dilatih dengan 150 data scikit-learn asli.")
                X_comb, y_comb = X_orig, y_orig
            else:
                sp_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
                df_mongo["target"] = df_mongo["species"].map(sp_map)
                X_new = df_mongo[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
                y_new = df_mongo["target"].values
                X_comb = np.vstack([X_orig, X_new])
                y_comb = np.hstack([y_orig, y_new])

            X_tr, X_te, y_tr, y_te = train_test_split(X_comb, y_comb, test_size=0.2, random_state=42)
            rf_retrained = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_retrained.fit(X_tr, y_tr)
            y_rt_pred = rf_retrained.predict(X_te)
            acc_rt = accuracy_score(y_te, y_rt_pred)

            st.success(f"✅ Model Berhasil Dilatih Ulang! Akurasi Uji: **{acc_rt*100:.2f}%**")

            rt_c1, rt_c2, rt_c3 = st.columns(3)
            rt_c1.metric("Data Bawaan Sklearn", len(X_orig))
            rt_c2.metric("Data Tambahan MongoDB", len(df_mongo) if not df_mongo.empty else 0)
            rt_c3.metric("Total Dataset Training", len(X_comb))

            st.markdown("**Classification Report Setelah Retraining:**")
            st.text(classification_report(y_te, y_rt_pred, target_names=iris_orig.target_names, zero_division=0))

            imp_df = pd.DataFrame({
                "Fitur Bunga": iris_orig.feature_names,
                "Feature Importance": rf_retrained.feature_importances_
            }).sort_values("Feature Importance", ascending=False)
            st.dataframe(imp_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat retraining: {e}")


# ============================================================
# 5. HALAMAN: 💬 COMMUNITY FEEDBACK & GROQ AI
# ============================================================
elif st.session_state.active_nav == "💬 Community Feedback & Groq AI":
    st.markdown("## 💬 Public Community Feedback & Groq AI Analytics")
    st.caption("Pusat interaksi masukan publik dengan analisis sentimen real-time dan ringkasan eksekutif berbantuan LLM Groq Llama 3.1.")

    if not is_mongo_online:
        st.error("❌ MongoDB Atlas tidak aktif. Aktifkan koneksi terlebih dahulu untuk mengakses fitur komentar publik.")
        st.stop()

    client = mongo_client
    db = client["test_db"]
    komentar_col = db["komentar_db"]

    # Fungsi Sentimen Analisis
    def analisa_sentimen_kata(teks):
        pos_words = [
            "bagus", "keren", "hebat", "mantap", "suka", "mudah", "cepat", "bermanfaat",
            "membantu", "sempurna", "luar biasa", "puas", "good", "great", "excellent",
            "awesome", "helpful", "easy", "fast", "love", "nice", "best", "terbaik", "menarik", "interaktif"
        ]
        neg_words = [
            "jelek", "buruk", "kurang", "sulit", "lambat", "error", "rusak", "bingung",
            "kecewa", "gagal", "bug", "lemot", "bad", "terrible", "poor", "slow",
            "confusing", "hard", "worst", "fail", "hate", "susah"
        ]
        t = teks.lower()
        p = sum(1 for w in pos_words if w in t)
        n = sum(1 for w in neg_words if w in t)
        if p > n:
            return "Positif 😊"
        elif n > p:
            return "Negatif 🙁"
        else:
            return "Netral 😐"

    # Form Komentar
    st.markdown("### ✍️ Kirim Masukan / Ulasan Pengunjung")
    with st.form("form_submit_feedback"):
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            u_nama = st.text_input("Nama Pengunjung *", placeholder="Misal: Leassa Mishima")
        with fb_c2:
            u_inst = st.text_input("Afiliasi / Universitas (Opsional)", placeholder="Misal: UKSW / Umum")

        u_rating = st.slider("Rating Kepuasan Aplikasi ⭐", 1, 5, 5, help="1 = Sangat Kurang, 5 = Sangat Bagus")
        u_pesan = st.text_area("Pesan Ulasan / Rekomendasi Fitur *", placeholder="Bagikan tanggapan Anda mengenai performa model, antarmuka, dan fitur AI...")

        btn_send_fb = st.form_submit_button("🚀 Kirim Masukan Publik", type="primary")
        if btn_send_fb:
            if not u_nama.strip():
                st.warning("⚠️ Mohon lengkapi nama Anda.")
            elif not u_pesan.strip():
                st.warning("⚠️ Pesan ulasan tidak boleh kosong.")
            else:
                s_label = analisa_sentimen_kata(u_pesan)
                komentar_col.insert_one({
                    "nama": u_nama.strip(),
                    "institusi": u_inst.strip() if u_inst else "Umum",
                    "rating": int(u_rating),
                    "komentar": u_pesan.strip(),
                    "sentimen": s_label,
                    "timestamp": datetime.now(),
                })
                st.success("🎉 Terima kasih! Masukan Anda telah tersimpan di MongoDB Atlas.")
                st.rerun()

    st.divider()

    # Statistik & AI Summary
    all_comments = list(komentar_col.find().sort("timestamp", -1))
    n_comm = len(all_comments)

    st.markdown("### 📊 Ringkasan Metrik Kepuasan Pengunjung")
    if n_comm > 0:
        r_list = [c.get("rating", 5) for c in all_comments]
        avg_r = sum(r_list) / n_comm
        pos_n = sum(1 for c in all_comments if "Positif" in str(c.get("sentimen", "")))
        neg_n = sum(1 for c in all_comments if "Negatif" in str(c.get("sentimen", "")))
        net_n = n_comm - pos_n - neg_n

        f_s1, f_s2, f_s3, f_s4 = st.columns(4)
        f_s1.metric("Total Ulasan Publik", n_comm)
        f_s2.metric("Rata-Rata Rating", f"{avg_r:.2f} / 5.0 ⭐")
        f_s3.metric("Sentimen Positif", f"{pos_n} ({pos_n/n_comm*100:.0f}%)")
        f_s4.metric("Sentimen Negatif/Netral", f"{neg_n + net_n}")

        # Groq LLM Auto-Summary Expander
        with st.expander("🤖 AI Auto-Summary & Executive Insights (Groq LPU LLM)", expanded=False):
            st.caption("Otomatisasi analisis rangkuman feedback dari seluruh pengunjung menggunakan model Llama 3.1 via Groq API.")
            user_groq_key = st.text_input("Groq API Key (Opsional - Kosongkan untuk ringkasan analitik bawaan):", type="password")

            if st.button("✨ Buat Ringkasan AI Sekarang"):
                with st.spinner("Menganalisis data ulasan pengunjung..."):
                    if user_groq_key.strip():
                        try:
                            sample_txt = "\n".join([
                                f"- [{c.get('rating', 5)}★] {c.get('nama')}: {c.get('komentar')}"
                                for c in all_comments[:15]
                            ])
                            payload = {
                                "model": "llama-3.1-8b-instant",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": (
                                            "Berikut kumpulan feedback pengunjung aplikasi web NexusAI Studio (Machine Learning & MongoDB):\n\n"
                                            f"{sample_txt}\n\n"
                                            "Buatlah ringkasan eksekutif singkat (1-2 paragraf) dalam bahasa Indonesia yang merangkum tingkat kepuasan, hal yang disukai, dan rekomendasi pengguna."
                                        )
                                    }
                                ],
                                "temperature": 0.5,
                                "max_tokens": 300,
                            }
                            req = urllib.request.Request(
                                "https://api.groq.com/openai/v1/chat/completions",
                                data=json.dumps(payload).encode("utf-8"),
                                headers={
                                    "Authorization": f"Bearer {user_groq_key.strip()}",
                                    "Content-Type": "application/json"
                                },
                                method="POST"
                            )
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                res_json = json.loads(resp.read().decode("utf-8"))
                                ai_text = res_json["choices"][0]["message"]["content"]
                                st.success("✅ **Ringkasan Eksekutif AI (Groq Llama 3.1):**")
                                st.markdown(ai_text)
                        except Exception as e_ai:
                            st.warning(f"⚠️ Gagal menghubungi Groq API ({e_ai}). Menampilkan ringkasan analitik internal:")
                            st.info(
                                f"Berdasarkan **{n_comm} ulasan**, tingkat kepuasan rata-rata adalah **{avg_r:.2f} / 5.0 ⭐** "
                                f"dengan dominasi sentimen positif sebesar **{pos_n/n_comm*100:.1f}%**. Pengunjung sangat mengapresiasi "
                                f"kelengkapan pemodelan ML (Regresi, Klasifikasi, Clustering) dan integrasi database MongoDB Atlas."
                            )
                    else:
                        st.info(
                            f"📌 **Ringkasan Analitik Komunitas:**\n\n"
                            f"- Dari total **{n_comm} ulasan**, kepuasan pengguna mencapai **{avg_r:.2f} / 5.0 ⭐**.\n"
                            f"- Proporsi respon: **{pos_n} Positif** ({pos_n/n_comm*100:.1f}%), **{net_n} Netral**, dan **{neg_n} Negatif**.\n"
                            f"- Kesimpulan: Aplikasi berfungsi dengan sangat stabil dan interaktif dalam menyajikan materi Artificial Intelligence."
                        )

        st.divider()

        # Feed Ulasan
        st.markdown("### 💬 Feed Ulasan & Diskusi Pengunjung")
        for c in all_comments:
            with st.container(border=True):
                k_c1, k_c2 = st.columns([4, 1])
                with k_c1:
                    stars = "⭐" * int(c.get("rating", 5))
                    s_tag = c.get("sentimen", "Netral 😐")
                    st.markdown(f"**👤 {c.get('nama', 'Anonim')}** — *{c.get('institusi', 'Umum')}* &nbsp;&nbsp; {stars}")
                    st.write(c.get("komentar", ""))
                    t_val = c.get("timestamp")
                    t_str = t_val.strftime("%d %b %Y, %H:%M") if isinstance(t_val, datetime) else str(t_val)
                    st.caption(f"🕒 {t_str} &nbsp;|&nbsp; Sentimen: **{s_tag}**")
                with k_c2:
                    cid = str(c.get("_id"))
                    if st.button("🗑️ Hapus", key=f"del_fb_{cid}", type="secondary"):
                        komentar_col.delete_one({"_id": ObjectId(cid)})
                        st.success("Ulasan dihapus!")
                        st.rerun()

    else:
        st.info("📭 Belum ada ulasan publik di database. Jadilah yang pertama memberikan masukan di atas!")


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    '<div style="text-align: center; opacity: 0.75; font-size: 0.78rem; padding: 8px 0; color: #94A3B8;">'
    '🤖 <b>NexusAI Studio</b> • Tugas Akhir Semester <b>Artificial Intelligence</b><br>'
    'Pengembang: <b>Khoirul Anwar (682024023)</b>'
    '</div>',
    unsafe_allow_html=True
)
