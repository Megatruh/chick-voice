# ChickVoice 🎙️

**Sistem Deteksi Dini Penyakit Pernapasan Ayam Broiler Berbasis Analisis Akustik dan IoT**

---

## 📋 Daftar Isi

- [Pengenalan Proyek](#pengenalan-proyek)
- [Fitur Utama](#fitur-utama)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi dan Setup](#instalasi-dan-setup)
- [Struktur Proyek](#struktur-proyek)
- [Penggunaan](#penggunaan)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Tim Pengembang](#tim-pengembang)
- [Lisensi](#lisensi)

---

## 📖 Pengenalan Proyek

**ChickVoice** adalah sistem otomatis berbasis Artificial Intelligence yang dirancang untuk mendeteksi dini penyakit pernapasan pada ayam broiler melalui analisis sinyal akustik (suara) di kandang. Sistem ini memanfaatkan:

- **Ekstraksi Fitur Suara (MFCC)**: Mel-Frequency Cepstral Coefficients untuk menganalisis karakteristik akustik rekaman audio
- **Deep Learning (CNN 1D)**: Convolutional Neural Network untuk klasifikasi pola suara
- **Edge Computing**: Konversi model ke TensorFlow Lite untuk eksekusi on-device di mikrokontroler (ESP32/Raspberry Pi)

Dengan sistem ini, peternak dapat melakukan monitoring kesehatan ayam secara real-time dan responsif terhadap deteksi dini penyakit.

---

## ✨ Fitur Utama

- ✅ **Klasifikasi 3-Kelas**: Membedakan suara normal (Healthy), noise, dan suara ayam sakit (Unhealthy)
- ✅ **Ekstraksi MFCC**: Menganalisis karakteristik spektral suara dengan akurat
- ✅ **CNN 1D Architecture**: Model deep learning yang efisien untuk data audio time-series
- ✅ **Model Optimization**: Konversi ke TensorFlow Lite dengan quantization untuk perangkat edge
- ✅ **On-Device Inference**: Dapat berjalan langsung di mikrokontroler dengan resource terbatas
- ✅ **Dataset Terstruktur**: Dataset audio Poultry Vocalization dengan label yang jelas

---

## ⚠️ Persyaratan Sistem

### Python Version (PENTING!)

```
🔴 WAJIB MENGGUNAKAN PYTHON 3.11
❌ JANGAN gunakan Python 3.12 atau 3.13
❌ JANGAN gunakan Python 3.10 atau lebih rendah
```

**Alasan**: TensorFlow memiliki dukungan terbatas untuk Python versi 3.12+ pada Windows. Python 3.11 adalah versi paling stabil untuk proyek ini.

### Persyaratan Perangkat

| Komponen    | Minimal    | Rekomendasi                                        |
| ----------- | ---------- | -------------------------------------------------- |
| **OS**      | Windows 10 | Windows 10/11                                      |
| **Python**  | 3.11.x     | 3.11.x (latest)                                    |
| **RAM**     | 4 GB       | 8 GB+                                              |
| **Storage** | 5 GB       | 10 GB+                                             |
| **GPU**     | -          | NVIDIA CUDA 11.8+ (opsional, mempercepat training) |

### Dependency Library

- **TensorFlow** 2.13+ - Framework deep learning
- **NumPy** - Komputasi numerik
- **Librosa** - Analisis audio dan ekstraksi fitur MFCC
- **Scikit-learn** - Preprocessing dan utility ML

---

## 🔧 Instalasi dan Setup

### ✅ Step 1: Install Python 3.11

1. Download Python 3.11 dari [python.org](https://www.python.org/downloads/)
2. Jalankan installer
3. **PENTING**: Centang opsi `Add Python 3.11 to PATH`
4. Klik `Install Now`

Verifikasi instalasi dengan membuka **Command Prompt (CMD)** atau **PowerShell** dan jalankan:

```cmd
python --version
```

Output yang diharapkan: `Python 3.11.x`

---

### ✅ Step 2: Clone atau Download Proyek

Jika menggunakan Git:

```cmd
git clone <repository-url>
cd lkti_dl_03-07
```

Atau download ZIP dan ekstrak folder proyek.

---

### ✅ Step 3: Buat Virtual Environment

Buka **Command Prompt (CMD)** atau **PowerShell** di folder proyek dan jalankan:

```cmd
python -m venv venv
```

Perintah ini akan membuat folder `venv` (Virtual Environment) di proyek Anda.

---

### ✅ Step 4: Aktifkan Virtual Environment

#### Untuk CMD (Command Prompt):

```cmd
venv\Scripts\activate
```

#### Untuk PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

**Catatan**: Jika PowerShell menampilkan error tentang execution policy, jalankan terlebih dahulu:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Setelah aktivasi, prompt Anda akan berubah menjadi:

```
(venv) C:\path\to\project>
```

Prefix `(venv)` menandakan virtual environment sudah aktif.

---

### ✅ Step 5: Upgrade pip

Pastikan pip (package manager) sudah versi terbaru:

```cmd
python -m pip install --upgrade pip
```

---

### ✅ Step 6: Instalasi Library Dependencies

Jalankan perintah berikut untuk menginstal semua dependencies:

```cmd
pip install tensorflow numpy librosa scikit-learn
```

**Penjelasan**:

- `tensorflow` - Framework deep learning (versi untuk Windows)
- `numpy` - Numerical computing
- `librosa` - Audio processing dan MFCC extraction
- `scikit-learn` - Machine learning utilities

Tunggu proses instalasi selesai (bisa memakan waktu 5-15 menit tergantung kecepatan internet).

Verifikasi instalasi dengan:

```cmd
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__} installed successfully')"
```

---

### ✅ Step 7: Verifikasi Setup Lengkap

Jalankan skrip verifikasi untuk memastikan semua library terinstal:

```cmd
python -c "
import tensorflow as tf
import numpy as np
import librosa
from sklearn import preprocessing
print('✅ Semua library berhasil diinstal!')
print(f'   TensorFlow: {tf.__version__}')
print(f'   NumPy: {np.__version__}')
print(f'   Librosa: {librosa.__version__}')
"
```

---

## 📁 Struktur Proyek

```
lkti_dl_03-07/
├── main.py                                          # Script training model
├── convert_tflite.py                               # Script konversi ke TensorFlow Lite
├── chickvoice_model.h5                             # Model terlatih (format Keras)
├── chickvoice_mfcc_cnn.h5                          # Model alternatif
├── chickvoice_model.tflite                         # Model optimized untuk edge device
├── README.md                                        # Dokumentasi ini
└── Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/
    └── Chicken_Audio_Dataset/
        ├── Healthy/                                 # Audio suara ayam sehat (~300+ files)
        ├── Noise/                                   # Audio noise kandang (~100+ files)
        └── Unhealthy/                              # Audio suara ayam sakit (~300+ files)
```

---

## 🚀 Penggunaan

### Alur Kerja Lengkap

```
┌─────────────────────────────────────────────────┐
│ 1. Aktifkan Virtual Environment                 │
│    venv\Scripts\activate (CMD)                  │
│    atau venv\Scripts\Activate.ps1 (PowerShell)  │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│ 2. Jalankan Training Model                      │
│    python main.py                               │
│                                                  │
│    Output: chickvoice_model.h5 & metrics        │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│ 3. Konversi ke TensorFlow Lite                  │
│    python convert_tflite.py                     │
│                                                  │
│    Output: chickvoice_model.tflite (optimized)  │
└─────────────────────────────────────────────────┘
```

### Step 1: Aktifkan Virtual Environment

**CMD**:

```cmd
venv\Scripts\activate
```

**PowerShell**:

```powershell
venv\Scripts\Activate.ps1
```

---

### Step 2: Jalankan Training Model

Skrip `main.py` akan:

1. Memuat dataset audio dari folder `Chicken_Audio_Dataset/`
2. Mengekstrak fitur MFCC dari setiap audio
3. Membangun arsitektur CNN 1D
4. Melatih model dengan dataset
5. Menyimpan model ke file `chickvoice_model.h5`

Jalankan dengan:

```cmd
python main.py
```

**Output yang diharapkan**:

```
Loading audio files from dataset...
✅ Loaded 700+ audio samples

Extracting MFCC features...
⏳ Processing... 100%

Building CNN model...
Model Summary:
  Total parameters: ~50K

Training model...
Epoch 1/20: loss=0.85, accuracy=0.78
Epoch 2/20: loss=0.62, accuracy=0.85
...
Epoch 20/20: loss=0.12, accuracy=0.95

✅ Model training complete!
Model saved to: chickvoice_model.h5
Accuracy: 95.2%
```

---

### Step 3: Konversi Model ke TensorFlow Lite

Skrip `convert_tflite.py` akan mengoptimalkan model `.h5` agar dapat berjalan di perangkat edge dengan resource terbatas.

Jalankan dengan:

```cmd
python convert_tflite.py
```

**Output yang diharapkan**:

```
Loading model from: chickvoice_model.h5
Model loaded successfully!

Converting to TensorFlow Lite...
Applying quantization optimization...

✅ Conversion successful!
Model saved to: chickvoice_model.tflite
File size: 450 KB (original: 2.1 MB)
Compression ratio: 78.6%
```

Model `chickvoice_model.tflite` siap digunakan di mikrokontroler (ESP32/Raspberry Pi).

---

### Step 4: Deaktifkan Virtual Environment (Opsional)

Ketika selesai bekerja, deaktifkan venv dengan:

```cmd
deactivate
```

---

## 📊 Dataset

### Struktur Dataset

```
Chicken_Audio_Dataset/
├── Healthy/          (Suara Ayam Sehat)
│   ├── healthy_001.wav
│   ├── healthy_002.wav
│   └── ... (~300+ files)
│
├── Unhealthy/        (Suara Ayam Sakit)
│   ├── sick_001.wav
│   ├── sick_002.wav
│   └── ... (~300+ files)
│
└── Noise/            (Noise Kandang)
    ├── noise_001.wav
    ├── noise_002.wav
    └── ... (~100+ files)
```

### Karakteristik Dataset

| Kelas         | Jumlah File | Durasi   | Deskripsi                             |
| ------------- | ----------- | -------- | ------------------------------------- |
| **Healthy**   | ~300+       | 5-30 sec | Suara normal ayam sehat di kandang    |
| **Unhealthy** | ~300+       | 5-30 sec | Suara ayam dengan gangguan pernapasan |
| **Noise**     | ~100+       | 5-30 sec | Noise latar kandang (kipas, air, dll) |

### Preproses Data

Model menggunakan fitur **MFCC (Mel-Frequency Cepstral Coefficients)**:

- **Jumlah MFCC**: 13 coefficients
- **Hop Length**: 512
- **FFT Size**: 2048
- **Sample Rate**: 22050 Hz

---

## 🧠 Arsitektur Model

### CNN 1D Architecture

```
Input Layer (MFCC Features)
    ↓
Conv1D (32 filters, kernel=3)
    ↓
BatchNormalization
    ↓
ReLU Activation
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
Conv1D (64 filters, kernel=3)
    ↓
BatchNormalization
    ↓
ReLU Activation
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
Flatten Layer
    ↓
Dense (128 units) + ReLU + Dropout(0.5)
    ↓
Dense (3 units) + Softmax
    ↓
Output (Healthy | Unhealthy | Noise)
```

### Hyperparameter Training

| Parameter            | Nilai                    |
| -------------------- | ------------------------ |
| **Optimizer**        | Adam                     |
| **Loss Function**    | Categorical Crossentropy |
| **Batch Size**       | 32                       |
| **Epochs**           | 20                       |
| **Validation Split** | 20%                      |
| **Learning Rate**    | 0.001                    |

---


## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik dan penelitian. Silakan hubungi tim pengembang untuk informasi lisensi lebih lanjut.

---

## 🆘 Troubleshooting

### Masalah: "Python command not found" atau "python is not recognized"

**Solusi**:

- Pastikan Python 3.11 sudah diinstall dan PATH sudah dikonfigurasi
- Restart Command Prompt/PowerShell
- Verifikasi dengan: `python --version`

---

### Masalah: Virtual Environment tidak aktif

**Solusi CMD**:

```cmd
venv\Scripts\activate
```

**Solusi PowerShell**:

```powershell
venv\Scripts\Activate.ps1
```

Jika error pada PowerShell, jalankan:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Masalah: Instalasi TensorFlow lambat atau error

**Solusi**:

1. Upgrade pip: `python -m pip install --upgrade pip`
2. Upgrade setuptools: `pip install --upgrade setuptools wheel`
3. Install TensorFlow dengan versi spesifik untuk Windows 3.11:
   ```cmd
   pip install tensorflow==2.13.0
   ```

---

### Masalah: CUDA/GPU tidak terdeteksi (tidak ada GPU di system adalah normal)

Jika ingin menggunakan GPU (opsional, hanya untuk mempercepat training):

```cmd
pip install tensorflow[and-cuda]==2.13.0
```

Namun untuk kebanyakan laptop, menggunakan CPU sudah cukup.

---

### Masalah: Dataset tidak ditemukan

**Solusi**:

- Pastikan folder `Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/` sudah ada
- Pastikan subfolder `Healthy/`, `Unhealthy/`, dan `Noise/` berisi file audio
- Verifikasi path pada script `main.py`

---

## 📞 Support & Contact

Untuk pertanyaan atau issue teknis, silakan hubungi tim pengembang melalui repositori ini.

---

## 🔗 Referensi

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Librosa Documentation](https://librosa.org/)
- [MFCC Feature Extraction](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- [CNN for Time Series Classification](https://arxiv.org/abs/1611.06455)
- [Dataset](https://data.mendeley.com/datasets/zp4nf2dxbh/1)

---

**Terakhir diupdate**: 2026-06-23  
**Status**: ✅ Production Ready
