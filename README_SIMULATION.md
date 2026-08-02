# Simulasi ChickVoice Terintegrasi

Paket ini ditempatkan langsung di root repository `lkti_dl_03-07`.

## Struktur akhir

```text
lkti_dl_03-07/
├── chickvoice_mfcc_cnn.h5
├── chickvoice_model.tflite
├── main.py
├── requirements-simulation.txt
├── README_SIMULATION.md
├── Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/
└── simulation/
    ├── __init__.py
    ├── api.py
    ├── audio_simulator.py
    ├── config.py
    ├── dashboard.py
    ├── model_utils.py
    └── run_all.sh
```

## Alur

```text
WAV dataset
→ librosa.load(sr=None)
→ 40 MFCC
→ padding/crop 200 frame
→ transpose menjadi (1, 200, 40)
→ chickvoice_mfcc_cnn.h5
→ HTTP POST
→ FastAPI
→ SQLite
→ Streamlit
```

## Instalasi Fedora

Dari root repository:

```bash
source venv/bin/activate
python -m pip install -r requirements-simulation.txt
chmod +x simulation/run_all.sh
```

## Jalankan satu perintah

```bash
./simulation/run_all.sh outbreak 2
```

Alamat:

- API: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

Tekan `Ctrl+C` untuk menghentikan seluruh proses.

## Jalankan manual

Terminal 1:

```bash
source venv/bin/activate
python -m uvicorn simulation.api:app --reload
```

Terminal 2:

```bash
source venv/bin/activate
python -m streamlit run simulation/dashboard.py
```

Terminal 3:

```bash
source venv/bin/activate
python -m simulation.audio_simulator --scenario outbreak --interval 2
```

## Skenario

```bash
python -m simulation.audio_simulator --scenario normal --interval 2
python -m simulation.audio_simulator --scenario outbreak --interval 2
python -m simulation.audio_simulator --scenario noisy --interval 2
```

Batasi menjadi 20 event:

```bash
python -m simulation.audio_simulator \
  --scenario outbreak \
  --interval 1 \
  --iterations 20
```

## Prediksi satu file

```bash
python -m simulation.audio_simulator \
  --audio "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/Unhealthy/1.wav" \
  --zone B
```

Prediksi tanpa mengirim ke API:

```bash
python -m simulation.audio_simulator \
  --audio "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset/Healthy/1.wav" \
  --zone A \
  --dry-run
```

## Hapus riwayat

```bash
curl -X DELETE http://127.0.0.1:8000/events
```

## Urutan kelas

```text
0 = Healthy
1 = Noise
2 = Unhealthy
```

Urutan ini mengikuti `LabelEncoder` untuk tiga nama folder tersebut.
