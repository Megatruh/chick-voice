import tensorflow as tf

# ==========================================
# KONFIGURASI NAMA FILE
# ==========================================
h5_model_path = "chickvoice_mfcc_cnn.h5"
tflite_model_path = "chickvoice_model.tflite"

def convert_model():
    try:
        print(f"1. Memuat model dari {h5_model_path}...")
        # Load model .h5 yang sudah di-training
        model = tf.keras.models.load_model(h5_model_path)
        
        print("2. Menyiapkan TFLite Converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # OPTIMASI EDGE DEVICE (Quantization)
        # Ini akan mengubah bobot model dari float32 ke int8/float16
        # Sangat direkomendasikan untuk mikrokontroler (ESP32)
        print("3. Menerapkan optimasi ukuran model...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        print("4. Memulai proses konversi (mungkin memakan waktu beberapa detik)...")
        tflite_model = converter.convert()
        
        print(f"5. Menyimpan model hasil konversi ke {tflite_model_path}...")
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
            
        print("\n✅ Konversi BERHASIL!")
        print(f"File {tflite_model_path} sekarang siap di-deploy ke ESP32 atau Raspberry Pi.")
        
    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat konversi: {e}")

if __name__ == "__main__":
    convert_model()