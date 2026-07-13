import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================
DATASET_PATH = "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection/Chicken_Audio_Dataset"
CLASSES = ["Healthy", "Noise", "Unhealthy"]
MAX_PAD_LEN = 200 

# ==========================================
# 2. FUNGSI EKSTRAKSI FITUR & AUGMENTASI
# ==========================================
def extract_features(file_path, augment=False):
    """
    Mengekstrak MFCC. Jika augment=True, tambahkan sedikit noise buatan
    untuk memperbanyak variasi data dan melatih model agar lebih kebal noise.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        # --- TEKNIK AUGMENTASI ---
        if augment:
            # Menambahkan white noise ringan
            noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
            audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
            
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
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
# 3. PROSES PEMUATAN DATA (DOUBLE DATASET)
# ==========================================
def load_data():
    features = []
    labels = []
    
    print("Mulai mengekstrak fitur audio (proses ini akan memakan waktu lebih lama karena augmentasi)...")
    for label in CLASSES:
        class_dir = os.path.join(DATASET_PATH, label)
        if not os.path.exists(class_dir):
            continue
            
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                
                # 1. Ekstrak audio asli
                data_asli = extract_features(file_path, augment=False)
                if data_asli is not None:
                    features.append(data_asli)
                    labels.append(label)
                
                # 2. Ekstrak audio dengan noise buatan (Duplikasi Data)
                data_aug = extract_features(file_path, augment=True)
                if data_aug is not None:
                    features.append(data_aug)
                    labels.append(label)
                    
    return np.array(features), np.array(labels)

X, y = load_data()
print(f"Total data diproses (Asli + Augmentasi): {len(X)}")

# ==========================================
# 4. PREPROCESSING & SPLITTING
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X = np.transpose(X, (0, 2, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# ==========================================
# 5. ARSITEKTUR MODEL 1D-CNN OPTIMASI
# ==========================================
print("Membangun model CNN MFCC dengan Batch Normalization...")
model = Sequential()

# Layer Konvolusi 1
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Layer Konvolusi 2
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Layer Konvolusi 3 (Ekstraksi pola yang lebih dalam)
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(CLASSES), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ==========================================
# 6. TRAINING DENGAN CALLBACKS
# ==========================================
# Kurangi learning rate jika akurasi validasi stuck selama 3 epoch
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# Hentikan training jika tidak ada perbaikan selama 10 epoch untuk mencegah overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

print("Memulai proses training yang telah dioptimasi...")
history = model.fit(
    X_train, y_train, 
    batch_size=32, 
    epochs=50, 
    validation_data=(X_test, y_test),
    callbacks=[lr_reduction, early_stopping]
)

score = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAkurasi Testing Akhir: {score[1]*100:.2f}%")

# ==========================================
# 7. SIMPAN MODEL
# ==========================================
model_name = "chickvoice_mfcc_cnn.h5"
model.save(model_name)
print(f"Model berhasil disimpan: {model_name}")

# ==========================================
# 8. VISUALISASI HASIL (Otomatis Tersimpan)
# ==========================================
PLOT_OUTPUT_DIR = "evaluation_plots" # Menyimpan di direktori evaluation_plots

# 8a. Plot Akurasi
fig_acc, ax_acc = plt.subplots(figsize=(8, 6))
ax_acc.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax_acc.set_title('Grafik Akurasi Model', fontsize=14, fontweight='bold')
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.legend()
ax_acc.grid(True, linestyle='--', alpha=0.7)
acc_path = os.path.join(PLOT_OUTPUT_DIR, "plot_accuracy.png")
fig_acc.savefig(acc_path, dpi=300, bbox_inches='tight')
plt.close(fig_acc)
print(f"✔ Grafik Akurasi disimpan: {acc_path}")

# 8b. Plot Loss
fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
ax_loss.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax_loss.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax_loss.set_title('Grafik Loss Model', fontsize=14, fontweight='bold')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()
ax_loss.grid(True, linestyle='--', alpha=0.7)
loss_path = os.path.join(PLOT_OUTPUT_DIR, "plot_loss.png")
fig_loss.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.close(fig_loss)
print(f"✔ Grafik Loss disimpan: {loss_path}")

# 8c. Confusion Matrix
y_pred_proba = model.predict(X_test, verbose=0)
y_pred_indices = np.argmax(y_pred_proba, axis=1)
y_true_indices = np.argmax(y_test, axis=1)

class_labels = le.classes_
cm = confusion_matrix(y_true_indices, y_pred_indices)

fig_cm, ax_cm = plt.subplots(figsize=(8, 7))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_labels, yticklabels=class_labels,
    linewidths=0.5, linecolor='white',
    annot_kws={"size": 14, "weight": "bold"}, ax=ax_cm
)
ax_cm.set_title('Confusion Matrix – ChickVoice CNN', fontsize=16, fontweight='bold', pad=15)
ax_cm.set_xlabel('Predicted Label', fontsize=13, labelpad=10)
ax_cm.set_ylabel('True Label', fontsize=13, labelpad=10)
cm_path = os.path.join(PLOT_OUTPUT_DIR, "plot_confusion_matrix.png")
fig_cm.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close(fig_cm)
print(f"✔ Confusion Matrix disimpan: {cm_path}")