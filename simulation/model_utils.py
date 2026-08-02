from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

from simulation.config import CLASS_LABELS, MAX_PAD_LEN, MODEL_PATH, N_MFCC


@dataclass(frozen=True)
class PredictionResult:
    label: str
    classification: str
    confidence: float
    probabilities: dict[str, float]
    sample_rate: int
    feature_shape: tuple[int, ...]


def load_chickvoice_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Pastikan chickvoice_mfcc_cnn.h5 berada di root repository."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)

    if input_shape != (None, MAX_PAD_LEN, N_MFCC):
        raise ValueError(
            f"Input model {input_shape} tidak sesuai "
            f"(None, {MAX_PAD_LEN}, {N_MFCC})."
        )
    if output_shape[-1] != len(CLASS_LABELS):
        raise ValueError(
            f"Output model {output_shape} tidak cocok dengan "
            f"{len(CLASS_LABELS)} kelas."
        )
    return model


def extract_features(audio_path: Path) -> tuple[np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio tidak ditemukan: {audio_path}")

    audio, sample_rate = librosa.load(audio_path, sr=None)
    if audio.size == 0:
        raise ValueError(f"File audio kosong: {audio_path}")

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )

    pad_width = MAX_PAD_LEN - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(
            mfccs,
            ((0, 0), (0, pad_width)),
            mode="constant",
        )
    else:
        mfccs = mfccs[:, :MAX_PAD_LEN]

    # Sama dengan training: np.transpose(X, (0, 2, 1)).
    features = mfccs.T[np.newaxis, ...].astype(np.float32)

    expected = (1, MAX_PAD_LEN, N_MFCC)
    if features.shape != expected:
        raise ValueError(
            f"Bentuk fitur {features.shape} tidak sesuai {expected}."
        )

    return features, int(sample_rate)


def predict_audio(
    model: tf.keras.Model,
    audio_path: Path,
) -> PredictionResult:
    features, sample_rate = extract_features(audio_path)
    output = model.predict(features, verbose=0)[0]

    predicted_index = int(np.argmax(output))
    label = CLASS_LABELS[predicted_index]
    mapping = {
        "Healthy": "healthy",
        "Noise": "noise",
        "Unhealthy": "unhealthy",
    }

    return PredictionResult(
        label=label,
        classification=mapping[label],
        confidence=float(output[predicted_index]),
        probabilities={
            class_name: float(probability)
            for class_name, probability in zip(CLASS_LABELS, output)
        },
        sample_rate=sample_rate,
        feature_shape=tuple(features.shape),
    )
