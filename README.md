# Sistem Peringatan Dini Kualitas Udara (Air Quality Early Warning System)

**Kelompok 5 - PPD | Modul Peringatan Dini Cerdas**

Aplikasi web berbasis Flask ini dirancang untuk memberikan prakiraan kualitas udara (PM2.5) di Jakarta menggunakan model Machine Learning (XGBoost). Sistem ini menyediakan wawasan prediktif untuk mendukung pengambilan keputusan terkait kesehatan dan aktivitas luar ruangan.

## Fitur Utama

1.  **Prakiraan Kualitas Udara (H+1, H+3, H+7)**
    *   Memprediksi konsentrasi PM2.5 untuk 1, 3, dan 7 hari ke depan.
    *   Menampilkan interval kepercayaan (Confidence Interval 95%) untuk setiap prediksi.
    *   Kategorisasi otomatis (Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat).

2.  **Sistem Pendukung Keputusan (Decision Support)**
    *   **Penilaian Risiko**: Menentukan tingkat risiko (Rendah hingga Kritis) berdasarkan prediksi dan tren terkini.
    *   **Rekomendasi**: Memberikan saran tindakan statis yang sesuai dengan tingkat risiko (misalnya: "Gunakan masker").
    *   **Indikator Keyakinan**: Menampilkan seberapa yakin model terhadap prediksinya.

3.  **Simulator Skenario (Scenario Simulator)**
    *   **Simulasi Interaktif**: Pengguna dapat memilih skenario (Normal, +10% Polusi, -10% Polusi) untuk melihat dampaknya terhadap prakiraan.
    *   **Analisis Sensitivitas**: Menyajikan perbandingan antara prakiraan baseline dan hasil simulasi untuk H+1, H+3, dan H+7.
    *   **Kontrol Aman**: Input simulasi dibatasi pada skenario yang telah ditentukan untuk menjaga validitas model.


4.  **Wawasan Model (Model Insights)**
    *   Halaman khusus untuk melihat performa model (MAE, R2).
    *   Visualisasi validasi aktual vs prediksi dan distribusi error.

5.  **Desain Akademis & Responsif**
    *   Antarmuka yang bersih, minimalis, dan profesional menggunakan Bootstrap 5.
    *   Visualisasi data interaktif (Matplotlib statis) yang informatif.

## Struktur Folder

```
air-quality-ml-app/
├── app.py                 # File utama aplikasi Flask (Backend Logic)
├── requirements.txt       # Daftar dependensi Python
├── data/
│   └── historical_data.csv # Dataset historis (sumber input)
├── models/                # Artefak Machine Learning
│   ├── best_model.pkl     # Model XGBoost terlatih
│   ├── feature_pipeline.pkl # Pipeline pemrosesan fitur
│   ├── metadata.pkl       # Metrik evaluasi model
│   └── residual_std.pkl   # Standar deviasi residu (untuk CI)
├── static/
│   ├── css/style.css      # Kustomisasi tampilan (CSS)
│   └── plots/             # Folder penyimpanan plot yang dihasilkan
├── templates/
│   ├── base.html          # Layout dasar HTML
│   ├── index.html         # Halaman Dashboard Utama
│   └── insight.html       # Halaman Wawasan Model
└── ...
```

## Persyaratan Sistem

*   Python 3.8 atau lebih baru.
*   Sistem Operasi: Windows, macOS, atau Linux.

## Cara Menjalankan Aplikasi

Ikuti langkah-langkah berikut untuk menjalankan sistem di komputer lokal Anda:

### 1. Persiapan Lingkungan (Opsional tapi Disarankan)
Disarankan menggunakan virtual environment agar dependensi tidak tercampur.

```bash
# Membuat virtual environment
python -m venv venv

# Mengaktifkan (Windows)
venv\Scripts\activate

# Mengaktifkan (Mac/Linux)
source venv/bin/activate
```

### 2. Instalasi Dependensi
Instal semua pustaka yang diperlukan menggunakan `pip`.

```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Jalankan skrip utama `app.py`.

```bash
python app.py
```

### 4. Akses di Browser
Setelah aplikasi berjalan, buka browser (Chrome, Firefox, Edge) dan akses alamat berikut:

[http://localhost:5000](http://localhost:5000)

## Catatan Teknis

*   **Offline Mode**: Aplikasi ini dirancang untuk berjalan sepenuhnya offline setelah dependensi terinstal. Tidak ada panggilan API eksternal saat runtime.
*   **Inferensi**: Model melakukan inferensi secara *real-time* berdasarkan data historis yang dimuat di folder `data/`.
*   **Feature Engineering**: Logika pemrosesan fitur (lags, rolling stats) direplikasi di `app.py` untuk menjamin konsistensi input saat forecasting.

---
**Dibuat oleh Kelompok 5 untuk Demonstrasi Akademis.**
