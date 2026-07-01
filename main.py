import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
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

# ==========================================
# 8. VISUALISASI EVALUASI MODEL
# ==========================================
print("\nMembuat grafik evaluasi model...")

# --- Konfigurasi Umum ---
PLOT_OUTPUT_DIR = "evaluation_plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# ------------------------------------------
# 8a. Grafik Akurasi (Training vs Validation)
# ------------------------------------------
fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
ax_acc.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, marker='o', markersize=4)
ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s', markersize=4, linestyle='--')
ax_acc.set_title('Model Accuracy: Training vs Validation', fontsize=16, fontweight='bold', pad=15)
ax_acc.set_xlabel('Epoch', fontsize=13)
ax_acc.set_ylabel('Accuracy', fontsize=13)
ax_acc.legend(fontsize=11, loc='lower right')
ax_acc.set_ylim([0, 1.05])
ax_acc.tick_params(axis='both', labelsize=11)
fig_acc.tight_layout()

acc_path = os.path.join(PLOT_OUTPUT_DIR, "plot_accuracy.png")
fig_acc.savefig(acc_path, dpi=300, bbox_inches='tight')
plt.close(fig_acc)
print(f"✔ Grafik Akurasi disimpan: {acc_path}")

# ------------------------------------------
# 8b. Grafik Loss (Training vs Validation)
# ------------------------------------------
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
ax_loss.plot(history.history['loss'], label='Training Loss', linewidth=2, marker='o', markersize=4, color='#e05c5c')
ax_loss.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=4, linestyle='--', color='#f5a623')
ax_loss.set_title('Model Loss: Training vs Validation', fontsize=16, fontweight='bold', pad=15)
ax_loss.set_xlabel('Epoch', fontsize=13)
ax_loss.set_ylabel('Loss', fontsize=13)
ax_loss.legend(fontsize=11, loc='upper right')
ax_loss.tick_params(axis='both', labelsize=11)
fig_loss.tight_layout()

loss_path = os.path.join(PLOT_OUTPUT_DIR, "plot_loss.png")
fig_loss.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.close(fig_loss)
print(f"✔ Grafik Loss disimpan: {loss_path}")

# ------------------------------------------
# 8c. Confusion Matrix
# ------------------------------------------
# Prediksi kelas dari X_test
y_pred_proba = model.predict(X_test, verbose=0)
y_pred_indices = np.argmax(y_pred_proba, axis=1)
y_true_indices = np.argmax(y_test, axis=1)

# Decode indeks numerik kembali ke nama kelas asli
class_labels = le.classes_   # ['Healthy', 'Noise', 'Unhealthy'] (urutan sesuai LabelEncoder)
cm = confusion_matrix(y_true_indices, y_pred_indices)

fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_labels,
    yticklabels=class_labels,
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 14, "weight": "bold"},
    ax=ax_cm
)
ax_cm.set_title('Confusion Matrix – ChickVoice CNN', fontsize=16, fontweight='bold', pad=15)
ax_cm.set_xlabel('Predicted Label', fontsize=13, labelpad=10)
ax_cm.set_ylabel('True Label', fontsize=13, labelpad=10)
ax_cm.tick_params(axis='both', labelsize=12)
fig_cm.tight_layout()

cm_path = os.path.join(PLOT_OUTPUT_DIR, "plot_confusion_matrix.png")
fig_cm.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close(fig_cm)
print(f"✔ Confusion Matrix disimpan: {cm_path}")

print(f"\n✅ Semua grafik evaluasi berhasil disimpan di folder: '{PLOT_OUTPUT_DIR}/'")