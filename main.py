import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================
DATASET_PATH = "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset"
CLASSES = ["Healthy", "Noise", "Unhealthy"]
# Panjang maksimum frame MFCC (Sesuaikan dengan rata-rata durasi audio, misal 5 detik)
MAX_PAD_LEN = 200 

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR (MFCC)
# ==========================================
def extract_features(file_path):
    """
    Mengekstrak fitur MFCC dari file audio.
    Sesuai dengan spesifikasi dokumen perancangan ChickVoice.
    """
    try:
        # Load audio (sr=None agar menggunakan sample rate asli audio)
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        # Ekstraksi MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Padding atau Truncating agar dimensi seragam untuk input CNN
        pad_width = MAX_PAD_LEN - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]
            
        return mfccs
    except Exception as e:
        print(f"Error memproses {file_path}: {e}")
        return None

# ==========================================
# 3. PROSES PEMUATAN DATA
# ==========================================
def load_data():
    features = []
    labels = []
    
    print("Mulai mengekstrak fitur audio...")
    for label in CLASSES:
        class_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_dir):
            print(f"Peringatan: Folder {class_dir} tidak ditemukan!")
            continue
            
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                data = extract_features(file_path)
                
                if data is not None:
                    features.append(data)
                    labels.append(label)
                    
    return np.array(features), np.array(labels)

# Jalankan ekstraksi
X, y = load_data()
print(f"Total data diproses: {len(X)}")

# ==========================================
# 4. PREPROCESSING & SPLITTING
# ==========================================
# Encode label teks ('Healthy', 'Unhealthy', 'Noise') menjadi angka (0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape X untuk input Conv1D (samples, time_steps, features)
# Transpose MFCC dari (n_mfcc, time_steps) menjadi (time_steps, n_mfcc)
X = np.transpose(X, (0, 2, 1))

# Bagi data menjadi Training (80%) dan Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

print(f"Dimensi X_train: {X_train.shape}")
print(f"Dimensi y_train: {y_train.shape}")

# ==========================================
# 5. ARSITEKTUR MODEL CNN
# ==========================================
print("Membangun model CNN MFCC...")
model = Sequential()

# Layer Konvolusi 1
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Layer Konvolusi 2
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Flatten & Dense Layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output Layer (3 Kelas)
model.add(Dense(len(CLASSES), activation='softmax'))

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ==========================================
# 6. TRAINING & EVALUASI
# ==========================================
print("Memulai proses training...")
history = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test))

# Evaluasi
score = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAkurasi Testing: {score[1]*100:.2f}%")

# ==========================================
# 7. SIMPAN MODEL
# ==========================================
# Simpan dalam format Keras untuk nantinya dikonversi ke TFLite (untuk Edge Device)
model_name = "chickvoice_mfcc_cnn.h5"
model.save(model_name)
print(f"Model berhasil disimpan dengan nama: {model_name}")
print("Label classes mapping:", dict(zip(le.classes_, le.transform(le.classes_))))