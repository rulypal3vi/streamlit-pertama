# 🤖 NexusAI Studio — Artificial Intelligence & Cloud Analytics Platform

**NexusAI Studio** adalah platform web Artificial Intelligence & Machine Learning interaktif berbasis **Streamlit** dan **MongoDB Atlas**. Platform ini dirancang untuk pemodelan Supervised Learning, Deep Learning (Multi-Layer Perceptron), Unsupervised Clustering, integrasi cloud database untuk continuous retraining, serta analisis sentimen feedback publik yang diperkuat oleh Groq LPU LLM (Llama 3.1).

Dibuat sebagai **Tugas Akhir Semester (TAS) — Artificial Intelligence**  
* **Nama**: Khoirul Anwar  
* **NIM**: 682024023  

---

## 🚀 Fitur & Modul Workspace NexusAI Studio

Aplikasi ini dirancang dengan arsitektur workspace modern yang terbagi ke dalam **5 Modul Interaktif**:

| Modul | Nama Modul | Fitur & Deskripsi Utama |
|:---:|---|---|
| **1** | **🧭 Overview Hub** | Executive summary platform, developer identity, 4 kartu KPI telemetri sistem real-time, navigasi modul cepat, dan diagram Mermaid arsitektur end-to-end pipeline Artificial Intelligence. |
| **2** | **📈 Supervised Learning Studio** | Command Bar konfigurasi untuk **Regression** dan **Classification** (11 Model termasuk **Deep Learning MLP**). Dilengkapi visualisasi diagnostik (Actual vs Predicted, Residual Plot, Confusion Matrix, Feature Importance) serta validasi model **5-Fold Cross-Validation**. |
| **3** | **🔵 Unsupervised Clustering Lab** | Pemodelan **K-Means Clustering** ($K=2 .. 8$) dengan pemilihan sumbu X & Y interaktif, visualisasi 2D Scatter Plot dengan penanda Centroid merah mencolok, serta metrik evaluasi **Silhouette Score** dan **Inertia**. |
| **4** | **🗄️ MongoDB Cloud & Retrain** | Integrasi database cloud **MongoDB Atlas** (`test_db`): Formulir input data Iris baru, tabel data explorer real-time, fitur hapus data, serta engine **Continuous Retraining** yang menggabungkan 150 data bawaan scikit-learn dengan data kustom MongoDB. |
| **5** | **💬 Community Feedback & Groq AI** | Interaksi publik dengan rating bintang ⭐ (1–5), analisis sentimen instan (*Positif, Netral, Negatif*), penyimpanan ke MongoDB (`komentar_db`), serta ringkasan eksekutif berbasis **Groq LPU LLM (Llama 3.1)**. |

---

## 📊 Ragam Algoritma & Sumber Dataset

### 1. Model Artificial Intelligence & Machine Learning
* **Regression**: Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, Random Forest Regressor, dan **Deep Learning: Neural Network (MLP Regressor)**.
* **Classification**: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree Classifier, Random Forest Classifier, Support Vector Machine (SVM), dan **Deep Learning: Neural Network (MLP Classifier)**.
* **Clustering (Unsupervised)**: K-Means Clustering dengan evaluasi Silhouette Score & Inertia.

### 2. Sumber Dataset
* **Scikit-Learn Toy Datasets**: Diabetes, California Housing, Iris, Wine, Breast Cancer, Digits.
* **UCI Machine Learning Repository**: 
  * *Wine Quality Red (UCI)* — Digunakan untuk pemodelan regresi maupun klasifikasi mutu minuman anggur.
  * *Wholesale Customers (UCI)* — Digunakan untuk clustering pengeluaran retail multi-fitur.
* **Real-Time Live Dataset**:
  * *Live Crypto Bitcoin (Real-Time)* — Mengambil data 30 hari pasar cryptocurrency Bitcoin secara real-time via REST API CoinGecko.

---

## 🗄️ Konfigurasi MongoDB Atlas

1. Masuk ke [MongoDB Atlas Dashboard](https://cloud.mongodb.com).
2. Buat **Database User** dan atur hak akses *read/write*.
3. Pada menu **Network Access**, tambahkan IP Address `0.0.0.0/0` (Allow from Anywhere).
4. Buat file `.streamlit/secrets.toml` untuk pengujian lokal:
   ```toml
   MONGODB_URI = "mongodb+srv://<db_username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
   ```
   *(File ini dilindungi oleh `.gitignore` agar kredensial database tidak ter-commit ke publik).*

---

## 📦 Panduan Instalasi & Menjalankan Lokal

```bash
# 1. Clone repository
git clone https://github.com/rulypal3vi/streamlit-pertama.git
cd streamlit-pertama

# 2. Buat & aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows (PowerShell/CMD)
# source .venv/bin/activate # Linux/macOS

# 3. Install seluruh dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit
streamlit run app.py
```

Buka browser pada alamat: `http://localhost:8501`.

---

## 🚀 Deployment ke Streamlit Cloud

1. Push repository ke GitHub:
   ```bash
   git add .
   git commit -m "feat: complete NexusAI Studio platform for TAS Artificial Intelligence"
   git push origin main
   ```
2. Buka [share.streamlit.io](https://share.streamlit.io) dan hubungkan akun GitHub Anda.
3. Pilih repository `streamlit-pertama` dan arahkan Main file path ke `app.py`.
4. Masuk ke **Advanced Settings → Secrets**, lalu tambahkan:
   ```toml
   MONGODB_URI = "mongodb+srv://<db_username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"
   ```
5. Klik **Deploy**! Aplikasi Anda siap diakses secara publik dan didaftarkan pada portal pengumpulan tugas.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.12
* **Web Framework**: Streamlit
* **Artificial Intelligence & Machine Learning**: Scikit-Learn, NumPy, Pandas
* **Deep Learning**: Scikit-Learn Multi-Layer Perceptron (`MLPClassifier`, `MLPRegressor`)
* **Visualization**: Matplotlib, Seaborn
* **Cloud Database**: MongoDB Atlas (`pymongo`, `certifi`, `dnspython`)
* **Generative AI & API**: Groq Cloud API (Llama 3.1 8B Instant), CoinGecko REST API

---

## 📄 Lisensi
Proyek ini dilisensikan di bawah [MIT License](LICENSE).
