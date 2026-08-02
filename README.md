# ChickVoice

**Early Warning System berbasis analisis bioakustik, machine learning, dan Internet of Things untuk mendeteksi indikasi awal gangguan pernapasan pada ayam petelur.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Platform](https://img.shields.io/badge/Platform-Fedora%20Linux-lightgrey)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

---

## Daftar Isi

- [Tentang ChickVoice](#tentang-chickvoice)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Repository](#struktur-repository)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi pada Fedora](#instalasi-pada-fedora)
- [Persiapan Dataset](#persiapan-dataset)
- [Menjalankan Simulasi](#menjalankan-simulasi)
- [Menjalankan Prediksi Satu Audio](#menjalankan-prediksi-satu-audio)
- [Menjalankan Training Model](#menjalankan-training-model)
- [Konversi TensorFlow Lite](#konversi-tensorflow-lite)
- [Preprocessing Audio](#preprocessing-audio)
- [API dan Database](#api-dan-database)
- [Zona Waktu WIB](#zona-waktu-wib)
- [File yang Tidak Disimpan di Git](#file-yang-tidak-disimpan-di-git)
- [Troubleshooting](#troubleshooting)
- [Batasan Sistem](#batasan-sistem)
- [Tim Pengembang](#tim-pengembang)
- [Lisensi](#lisensi)
- [Referensi](#referensi)

---

## Tentang ChickVoice

ChickVoice merupakan purwarupa sistem pemantauan kesehatan ayam petelur berbasis suara.

Sistem menganalisis rekaman audio untuk mengklasifikasikan tiga kondisi:

1. `Healthy`, yaitu suara ayam sehat.
2. `Noise`, yaitu kebisingan lingkungan kandang.
3. `Unhealthy`, yaitu suara yang memiliki indikasi gangguan kesehatan atau pernapasan.

ChickVoice menggunakan:

- **Mel-Frequency Cepstral Coefficients atau MFCC** untuk mengekstrak karakteristik audio.
- **1D Convolutional Neural Network atau 1D-CNN** untuk melakukan klasifikasi.
- **TensorFlow Lite** untuk menyiapkan model agar dapat dikembangkan menuju perangkat edge.
- **FastAPI** sebagai REST API.
- **SQLite** sebagai basis data simulasi.
- **Streamlit** sebagai dashboard pemantauan zona kandang.

ChickVoice merupakan sistem peringatan dini. Sistem ini tidak menggantikan diagnosis dokter hewan atau pemeriksaan laboratorium.

---

## Fitur Utama

- Klasifikasi audio tiga kelas: `Healthy`, `Noise`, dan `Unhealthy`.
- Ekstraksi 40 koefisien MFCC.
- Input model berukuran `(200, 40)`.
- Model klasifikasi berbasis 1D-CNN.
- Model Keras dalam format `.h5`.
- Model edge dalam format `.tflite`.
- Simulasi pemantauan Zona A sampai Zona D.
- REST API untuk menerima hasil inferensi.
- Basis data SQLite untuk menyimpan event.
- Dashboard heatmap kondisi kandang.
- Penyimpanan confidence score setiap prediksi.
- Penyimpanan label sumber dan hasil prediksi model.
- Skenario pengujian `normal`, `outbreak`, dan `noisy`.
- Dukungan simulasi otomatis melalui shell script.

---

## Arsitektur Sistem

```text
File audio WAV
      │
      ▼
librosa.load(sr=None)
      │
      ▼
Ekstraksi 40 MFCC
      │
      ▼
Padding atau pemotongan menjadi 200 frame
      │
      ▼
Transpose menjadi bentuk (200, 40)
      │
      ▼
Model 1D-CNN
      │
      ├── Healthy
      ├── Noise
      └── Unhealthy
      │
      ▼
HTTP POST
      │
      ▼
FastAPI REST API
      │
      ▼
SQLite Database
      │
      ▼
Streamlit Dashboard
```

Model menerima bentuk input:

```text
(None, 200, 40)
```

Model menghasilkan tiga probabilitas:

```text
(None, 3)
```

Urutan kelas model:

```text
0 = Healthy
1 = Noise
2 = Unhealthy
```

---

## Struktur Repository

```text
lkti_dl_03-07/
├── chickvoice_mfcc_cnn.h5
├── chickvoice_model.tflite
├── convert_tflite.py
├── main.py
├── README.md
├── README_SIMULATION.md
├── requirements-simulation.txt
│
├── evaluation_plots/
│   ├── plot_accuracy.png
│   ├── plot_confusion_matrix.png
│   └── plot_loss.png
│
├── Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/
│   ├── Chicken_Audio_Dataset/
│   │   ├── Healthy/
│   │   ├── Noise/
│   │   └── Unhealthy/
│   └── Chicken_Audio_Dataset.tar.gz
│
└── simulation/
    ├── __init__.py
    ├── api.py
    ├── audio_simulator.py
    ├── config.py
    ├── dashboard.py
    ├── model_utils.py
    ├── run_all.sh
    ├── chickvoice.db
    └── logs/
```

File database, log, virtual environment, cache Python, dan dataset dapat dikecualikan dari Git melalui `.gitignore`.

---

## Persyaratan Sistem

### Perangkat lunak

- Fedora Linux atau distribusi Linux lain.
- Python 3.11.
- Git.
- pip.
- Python virtual environment.
- libsndfile.
- Browser modern.

### Perangkat minimum

| Komponen | Minimum | Rekomendasi |
|---|---:|---:|
| RAM | 4 GB | 8 GB atau lebih |
| Penyimpanan kosong | 3 GB | 5 GB atau lebih |
| CPU | Dual-core | Quad-core atau lebih |
| GPU | Tidak wajib | GPU kompatibel TensorFlow untuk training |

GPU tidak diperlukan untuk menjalankan simulasi dan prediksi satu file.

---

## Instalasi pada Fedora

### 1. Instal dependency sistem

```bash
sudo dnf install -y \
  python3.11 \
  python3.11-devel \
  python3-pip \
  gcc \
  gcc-c++ \
  git \
  curl \
  unzip \
  libsndfile \
  libsndfile-devel
```

Periksa versi Python:

```bash
python3.11 --version
```

Output yang diharapkan:

```text
Python 3.11.x
```

### 2. Clone repository

```bash
git clone <URL_REPOSITORY>
cd lkti_dl_03-07
```

Ganti `<URL_REPOSITORY>` dengan URL repository Git yang digunakan oleh tim.

### 3. Buat virtual environment

```bash
python3.11 -m venv venv
```

### 4. Aktifkan virtual environment

```bash
source venv/bin/activate
```

Prompt terminal akan memiliki awalan:

```text
(venv)
```

### 5. Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 6. Instal dependency simulasi

```bash
python -m pip install -r requirements-simulation.txt
```

Jika dependency untuk training belum tersedia, jalankan:

```bash
python -m pip install \
  tensorflow \
  numpy \
  librosa \
  soundfile \
  scikit-learn \
  matplotlib \
  pandas \
  fastapi \
  "uvicorn[standard]" \
  requests \
  streamlit \
  streamlit-autorefresh
```

### 7. Beri izin eksekusi pada script

```bash
chmod +x simulation/run_all.sh
```

### 8. Verifikasi library

```bash
python - <<'PY'
import tensorflow as tf
import numpy as np
import librosa
import sklearn
import fastapi
import streamlit

print("TensorFlow :", tf.__version__)
print("NumPy      :", np.__version__)
print("Librosa    :", librosa.__version__)
print("Scikit-learn berhasil dimuat")
print("FastAPI berhasil dimuat")
print("Streamlit berhasil dimuat")
PY
```

---

## Persiapan Dataset

Dataset tidak disarankan untuk disimpan langsung di Git karena terdiri dari banyak file audio.

Dataset dapat diperoleh dari:

[Enhancing Poultry Health Management Through Machine Learning-Based Analysis of Vocalization Signals Dataset](https://data.mendeley.com/datasets/zp4nf2dxbh/1)

Setelah diunduh, letakkan dataset dengan struktur berikut:

```text
Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/
└── Chicken_Audio_Dataset/
    ├── Healthy/
    │   ├── 1.wav
    │   ├── 2.wav
    │   └── ...
    │
    ├── Noise/
    │   ├── 1.wav
    │   ├── 2.wav
    │   └── ...
    │
    └── Unhealthy/
        ├── 1.wav
        ├── 2.wav
        └── ...
```

Periksa jumlah file:

```bash
find \
  "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset" \
  -type f \
  -name "*.wav" \
  | wc -l
```

Periksa setiap kelas:

```bash
for class_name in Healthy Noise Unhealthy
do
  count=$(find \
    "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/$class_name" \
    -type f \
    -name "*.wav" \
    | wc -l)

  echo "$class_name: $count file"
done
```

---

## Menjalankan Simulasi

Simulasi dapat dijalankan dengan satu perintah atau melalui tiga terminal terpisah.

### Cara cepat

Aktifkan virtual environment:

```bash
source venv/bin/activate
```

Jalankan skenario outbreak:

```bash
./simulation/run_all.sh outbreak 2
```

Parameter:

```text
outbreak = nama skenario
2        = interval antarevent dalam detik
```

Alamat layanan:

| Layanan | Alamat |
|---|---|
| REST API | http://127.0.0.1:8000 |
| Dokumentasi API | http://127.0.0.1:8000/docs |
| Dashboard | http://localhost:8501 |

Tekan `Ctrl+C` untuk menghentikan simulasi.

### Skenario normal

```bash
./simulation/run_all.sh normal 2
```

Skenario ini mengambil mayoritas file audio dari kelas `Healthy`.

### Skenario outbreak

```bash
./simulation/run_all.sh outbreak 2
```

Skenario ini meningkatkan pemilihan audio `Unhealthy`, terutama pada zona tertentu.

### Skenario noisy

```bash
./simulation/run_all.sh noisy 2
```

Skenario ini meningkatkan pemilihan file dari kelas `Noise`.

### Catatan tentang skenario

Skenario hanya mengatur distribusi file audio yang dipilih dari dataset.

Hasil yang dikirim ke dashboard tetap berasal dari prediksi model CNN. Model dapat menghasilkan prediksi yang berbeda dari label folder audio.

Contoh:

```text
Label sumber   : Unhealthy
Prediksi model : Healthy
```

Kondisi tersebut menunjukkan kesalahan klasifikasi model dan tidak boleh disembunyikan.

---

## Menjalankan Simulasi Secara Manual

### Terminal 1: FastAPI

```bash
cd lkti_dl_03-07
source venv/bin/activate

python -m uvicorn simulation.api:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

### Terminal 2: Dashboard

```bash
cd lkti_dl_03-07
source venv/bin/activate

python -m streamlit run simulation/dashboard.py
```

### Terminal 3: Simulator audio

```bash
cd lkti_dl_03-07
source venv/bin/activate

python -m simulation.audio_simulator \
  --scenario outbreak \
  --interval 2
```

### Menjalankan sejumlah event tertentu

Contoh 20 event:

```bash
python -m simulation.audio_simulator \
  --scenario outbreak \
  --interval 1 \
  --iterations 20
```

---

## Menjalankan Prediksi Satu Audio

### Audio Healthy

```bash
python -m simulation.audio_simulator \
  --audio "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/Healthy/1.wav" \
  --zone A
```

### Audio Unhealthy

```bash
python -m simulation.audio_simulator \
  --audio "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/Unhealthy/1.wav" \
  --zone B
```

### Audio Noise

```bash
python -m simulation.audio_simulator \
  --audio "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/Noise/1.wav" \
  --zone C
```

### Prediksi tanpa mengirim ke API

```bash
python -m simulation.audio_simulator \
  --audio "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/Healthy/1.wav" \
  --zone A \
  --dry-run
```

Contoh output:

```text
Model berhasil dimuat
Input shape : (None, 200, 40)
Output shape: (None, 3)
Urutan kelas: ['Healthy', 'Noise', 'Unhealthy']

Zona            : A
Label sumber    : Healthy
Prediksi model  : Healthy
Confidence      : 98.41%
Sample rate     : 22050 Hz
Feature shape   : (1, 200, 40)
```

Nilai confidence dan sample rate dapat berbeda pada setiap file.

---

## Menjalankan Training Model

File `main.py` digunakan untuk:

1. Membaca seluruh audio dataset.
2. Mengekstrak fitur MFCC.
3. Menyamakan panjang matriks fitur.
4. Mengubah label menjadi data numerik.
5. Membagi data training dan testing.
6. Melatih model 1D-CNN.
7. Mengevaluasi model.
8. Menyimpan model dan grafik evaluasi.

Jalankan:

```bash
source venv/bin/activate
python main.py
```

Model disimpan sebagai:

```text
chickvoice_mfcc_cnn.h5
```

Grafik evaluasi disimpan pada:

```text
evaluation_plots/
├── plot_accuracy.png
├── plot_confusion_matrix.png
└── plot_loss.png
```

Training dapat menimpa model dan grafik sebelumnya. Simpan salinan model jika hasil lama masih diperlukan.

---

## Konversi TensorFlow Lite

Gunakan:

```bash
python convert_tflite.py
```

Hasil konversi:

```text
chickvoice_model.tflite
```

Model TensorFlow Lite disiapkan untuk pengembangan menuju edge device.

File `.tflite` belum membuktikan bahwa model sudah berjalan langsung pada ESP32. Implementasi pada mikrokontroler tetap memerlukan:

- TensorFlow Lite Micro.
- Firmware untuk akuisisi audio.
- Implementasi MFCC yang konsisten dengan training.
- Pengaturan memori perangkat.
- Pengujian latensi.
- Pengujian pada lingkungan kandang.

---

## Preprocessing Audio

Parameter preprocessing model saat ini:

| Parameter | Nilai |
|---|---|
| Sample rate | Mempertahankan sample rate asli, `sr=None` |
| Jumlah MFCC | 40 |
| Panjang maksimum | 200 frame |
| Input satu file | `(1, 200, 40)` |
| Format audio | WAV |
| Padding | Zero padding |
| Pemotongan | Maksimal 200 frame |
| Urutan kelas | Healthy, Noise, Unhealthy |

Alur preprocessing:

```text
Audio WAV
    │
    ▼
librosa.load(file, sr=None)
    │
    ▼
librosa.feature.mfcc(n_mfcc=40)
    │
    ▼
Matriks awal: (40, jumlah_frame)
    │
    ▼
Padding atau crop: (40, 200)
    │
    ▼
Transpose: (200, 40)
    │
    ▼
Tambah dimensi batch: (1, 200, 40)
```

Model hanya dapat memberikan hasil yang konsisten jika preprocessing inference sama dengan preprocessing training.

---

## API dan Database

FastAPI menyediakan endpoint berikut:

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/health` | Memeriksa status API |
| POST | `/events` | Menyimpan hasil inferensi |
| GET | `/events` | Mengambil riwayat event |
| GET | `/latest` | Mengambil event terbaru setiap zona |
| DELETE | `/events` | Menghapus seluruh event |

### Memeriksa API

```bash
curl http://127.0.0.1:8000/health
```

### Melihat event terbaru

```bash
curl http://127.0.0.1:8000/latest
```

### Melihat 20 event terakhir

```bash
curl "http://127.0.0.1:8000/events?limit=20"
```

### Menghapus seluruh event

```bash
curl -X DELETE http://127.0.0.1:8000/events
```

Database lokal berada pada:

```text
simulation/chickvoice.db
```

Database tersebut tidak perlu disimpan di Git.

---

## Zona Waktu WIB

Untuk menggunakan waktu Indonesia Barat, gunakan zona waktu:

```text
Asia/Jakarta
```

Contoh implementasi Python:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

WIB = ZoneInfo("Asia/Jakarta")

timestamp = datetime.now(WIB).isoformat()
```

Contoh output:

```text
2026-08-02T13:45:20+07:00
```

Hindari timestamp tanpa informasi zona waktu.

Periksa penggunaan timezone pada repository:

```bash
grep -R "timezone.utc\|datetime.now" -n simulation
```

Jika simulator masih menggunakan UTC, ubah:

```python
datetime.now(timezone.utc).isoformat()
```

menjadi:

```python
datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()
```

Tambahkan import:

```python
from zoneinfo import ZoneInfo
```

---

## File yang Tidak Disimpan di Git

File berikut sebaiknya masuk `.gitignore`:

```gitignore
# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
.venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Python tools
.pytest_cache/
.mypy_cache/
.pyright/
.ruff_cache/
.ipynb_checkpoints/

# Build
build/
dist/
*.egg-info/

# Environment variables
.env
.env.*
!.env.example

# Runtime database and logs
simulation/chickvoice.db
simulation/logs/
*.db
*.sqlite
*.sqlite3
*.log

# Generated folders
recordings/
uploads/
outputs/
tensorboard/
checkpoint/
checkpoints/

# Dataset lokal
Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/
```

Jika file sudah pernah masuk tracking Git, `.gitignore` tidak langsung menghapusnya.

Gunakan:

```bash
git rm -r --cached simulation/__pycache__
git rm --cached simulation/chickvoice.db
git rm -r --cached simulation/logs
git rm -r --cached \
  "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection"
```

Kemudian:

```bash
git add .
git status
```

Pastikan database, log, cache, virtual environment, dan dataset tidak muncul dalam daftar file yang akan di-commit.

---

## Troubleshooting

### `python3.11: command not found`

Periksa paket yang tersedia:

```bash
dnf search python3.11
```

Instal:

```bash
sudo dnf install python3.11 python3.11-devel
```

### Virtual environment belum aktif

```bash
source venv/bin/activate
```

### `No module named tensorflow`

```bash
python -m pip install tensorflow
```

### `No module named librosa`

```bash
python -m pip install librosa soundfile
```

### `No module named uvicorn`

```bash
python -m pip install "uvicorn[standard]"
```

### `No module named streamlit`

```bash
python -m pip install streamlit streamlit-autorefresh
```

### Error `libsndfile`

```bash
sudo dnf install libsndfile libsndfile-devel
```

Kemudian:

```bash
python -m pip install --force-reinstall soundfile
```

### Port 8000 sudah digunakan

```bash
ss -ltnp | grep 8000
```

Cari proses Uvicorn:

```bash
ps aux | grep uvicorn
```

Hentikan proses lama atau terminal sebelumnya.

### Port 8501 sudah digunakan

```bash
ss -ltnp | grep 8501
```

Jalankan dashboard pada port lain:

```bash
python -m streamlit run \
  simulation/dashboard.py \
  --server.port 8502
```

### API tidak dapat diakses

```bash
curl http://127.0.0.1:8000/health
```

Jika gagal, jalankan kembali:

```bash
python -m uvicorn simulation.api:app --reload
```

### Dataset tidak ditemukan

Periksa path:

```bash
ls \
  "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset"
```

Output harus memiliki:

```text
Healthy
Noise
Unhealthy
```

### Bentuk input model tidak sesuai

Periksa model:

```bash
python - <<'PY'
import tensorflow as tf

model = tf.keras.models.load_model(
    "chickvoice_mfcc_cnn.h5",
    compile=False
)

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)
PY
```

Output yang diharapkan:

```text
Input shape : (None, 200, 40)
Output shape: (None, 3)
```

---

## Batasan Sistem

ChickVoice masih berada pada tahap purwarupa penelitian.

Batasan saat ini:

- Simulasi menggunakan file audio dataset, bukan mikrofon kandang secara langsung.
- Skenario `normal`, `outbreak`, dan `noisy` mengatur distribusi input simulasi.
- Model belum diuji secara luas pada kandang ayam petelur komersial di Indonesia.
- Akurasi pada dataset tidak otomatis sama dengan akurasi lapangan.
- Model belum membedakan jenis penyakit secara spesifik.
- Label `Unhealthy` menunjukkan indikasi suara abnormal, bukan diagnosis penyakit tertentu.
- TensorFlow Lite belum berarti implementasi ESP32 telah selesai.
- Sistem belum menggantikan pemeriksaan dokter hewan.
- Sistem belum dapat disebut production ready.

Status yang tepat:

```text
Research Prototype
```

---

## Tim Pengembang

ChickVoice dikembangkan oleh:

1. Hafidz Januar Faturahman
2. Muhammad Naufal Syifau Rahman
3. Farhan Esha Putra Kusuma Atmaja
4. Lalita Bela

Fakultas Teknik, Universitas Siliwangi.

---

## Lisensi

Proyek ini dikembangkan untuk keperluan akademik dan penelitian.

Kode program, model, dan dokumentasi proyek mengikuti ketentuan yang ditetapkan oleh tim pengembang.

Dataset memiliki sumber dan ketentuan lisensi tersendiri. Pengguna wajib mengikuti lisensi dari penyedia dataset asli.

Repository ini tidak memberikan jaminan bahwa hasil klasifikasi dapat digunakan sebagai diagnosis medis atau veteriner.

---

## Referensi

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Librosa Documentation](https://librosa.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Mendeley Poultry Vocalization Dataset](https://data.mendeley.com/datasets/zp4nf2dxbh/1)

---

## Status Proyek

```text
Status              : Research Prototype
Model input         : (None, 200, 40)
Model output        : (None, 3)
Jumlah kelas        : 3
Kelas               : Healthy, Noise, Unhealthy
Dashboard           : Streamlit
API                 : FastAPI
Database            : SQLite
Edge model          : TensorFlow Lite
Validasi lapangan   : Belum dilakukan secara penuh
```

**Terakhir diperbarui: Agustus 2026**
