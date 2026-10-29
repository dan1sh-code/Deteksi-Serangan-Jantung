<<<<<<< HEAD
# 🫀 CardioGuard AI - Multi-Model Heart Risk Prediction System

Sistem Machine Learning berkinerja tinggi untuk memprediksi risiko penyakit jantung secara real-time dan batch CSV, ditenagai oleh **Polars Engine** (DataFrame berbasis Rust), **LightGBM**, dan **SuperEnsemble Machine Learning**.

---

## 📁 Struktur Folder Project Modern

```text
Prediksi_Riwayat_Jantung/
├── data/                       # Tempat file dataset (Data_Penyakit_Jantung.csv & heart.csv)
├── models/                     # Tempat menyimpan file model terlatih (joblib)
├── static/
│   └── plots/                  # Visualisasi evaluasi (ROC Curve, Confusion Matrix, dsb)
├── templates/
│   └── index.html              # Interface Dashboard Web Glassmorphism
├── src/                        # Core Python Source Code Modules
│   ├── eda.py                  # Module Eksplorasi Data Polars
│   ├── train_model.py          # Module Penggabungan Dataset & Training 5 Model ML
│   └── predict.py              # Inference Engine & Kalkulator Risiko
├── app.py                      # Flask Web Application & REST API Entrypoint
├── eda.py                      # CLI Runner EDA
├── train_model.py              # CLI Runner Training
├── predict.py                  # CLI Runner Prediction
└── README.md
```

---

## ⚡ Hasil Penggabungan Dataset & Performa Model

Dataset digabungkan dari dua sumber data medis (`Data_Penyakit_Jantung.csv` + `heart.csv`) menghasilkan total **79,886 baris data rekam medis** yang diproses oleh **Polars** hanya dalam **0.241 detik**.

| Algoritma Machine Learning | ROC-AUC Score | F1-Score | Status |
| :--- | :-: | :-: | :--- |
| 🏆 **1. Logistic Regression Scaled** | **84.72%** | **0.3456** | Model Utama Terpilih |
| 🌟 **2. SuperEnsemble (LGB + HGB + LR)** | **84.48%** | **0.3420** | Triple Soft Voting |
| 🌲 **3. HistGradientBoosting** | **84.05%** | **0.3380** | Gradient Boosted Trees |
| ⚡ **4. LightGBM Classifier** | **83.75%** | **0.3350** | Fast Boosting Engine |
| 🌳 **5. ExtraTrees Classifier** | **81.36%** | **0.3120** | Randomized Trees |

---

## 🚀 Cara Menjalankan Aplikasi

1. **Jalankan Web Server**:
   ```bash
   python app.py
   ```
   Buka di browser: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

2. **Jalankan Pelatihan via CLI**:
   ```bash
   python train_model.py
   ```

3. **Jalankan Eksplorasi Data via CLI**:
   ```bash
   python eda.py
   ```
=======
# Deteksi-Serangan-Jantung
>>>>>>> dcdae703c6b7206062ca8988f4814408ea04cf06
