from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import requests

from simulation.config import (
    API_BASE_URL,
    CLASS_LABELS,
    DATASET_DIR,
    ROOT_DIR,
    VALID_ZONES,
)
from simulation.model_utils import load_chickvoice_model, predict_audio


SCENARIO_WEIGHTS: dict[str, dict[str, tuple[int, int, int]]] = {
    "normal": {
        "A": (88, 10, 2),
        "B": (88, 10, 2),
        "C": (88, 10, 2),
        "D": (88, 10, 2),
    },
    "outbreak": {
        "A": (85, 12, 3),
        "B": (15, 5, 80),
        "C": (85, 12, 3),
        "D": (85, 12, 3),
    },
    "noisy": {
        "A": (27, 70, 3),
        "B": (27, 70, 3),
        "C": (27, 70, 3),
        "D": (27, 70, 3),
    },
}


def validate_dataset() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_DIR}")

    for label in CLASS_LABELS:
        class_dir = DATASET_DIR / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")
        if not list(class_dir.glob("*.wav")):
            raise FileNotFoundError(f"Tidak ada WAV dalam: {class_dir}")


def pick_audio(scenario: str, zone: str) -> tuple[Path, str]:
    source_label = random.choices(
        CLASS_LABELS,
        weights=SCENARIO_WEIGHTS[scenario][zone],
        k=1,
    )[0]
    audio_path = random.choice(list((DATASET_DIR / source_label).glob("*.wav")))
    return audio_path, source_label


def send_prediction(
    *,
    api_url: str,
    node_id: str,
    zone: str,
    audio_path: Path,
    source_label: str | None,
    classification: str,
    confidence: float,
) -> dict:
    try:
        relative_audio = str(audio_path.relative_to(ROOT_DIR))
    except ValueError:
        relative_audio = str(audio_path)

    payload = {
        "node_id": node_id,
        "zone": zone,
        "classification": classification,
        "confidence": confidence,
        "timestamp" : datetime.now(ZoneInfo("Asia/Jakarta")).isoformat(),
        "audio_file": relative_audio,
        "source_label": source_label,
    }

    response = requests.post(api_url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def show_result(zone: str, audio_path: Path, source_label: str | None, result) -> None:
    print()
    print("=" * 72)
    print(f"Zona            : {zone}")
    print(f"Audio           : {audio_path}")
    print(f"Label sumber    : {source_label or '-'}")
    print(f"Prediksi model  : {result.label}")
    print(f"Confidence      : {result.confidence:.2%}")
    print(f"Sample rate     : {result.sample_rate} Hz")
    print(f"Feature shape   : {result.feature_shape}")
    print("Probabilitas:")
    for label, probability in result.probabilities.items():
        print(f"  {label:<10}: {probability:.2%}")


def run_single(args, model) -> None:
    audio_path = Path(args.audio).expanduser().resolve()
    source_label = audio_path.parent.name if audio_path.parent.name in CLASS_LABELS else None
    result = predict_audio(model, audio_path)
    show_result(args.zone, audio_path, source_label, result)

    if args.dry_run:
        print("Dry run aktif. Data tidak dikirim ke API.")
        return

    api_result = send_prediction(
        api_url=args.api_url,
        node_id=args.node_id,
        zone=args.zone,
        audio_path=audio_path,
        source_label=source_label,
        classification=result.classification,
        confidence=result.confidence,
    )
    print(f"API berhasil. Event ID: {api_result['id']}")


def run_scenario(args, model) -> None:
    validate_dataset()
    iteration = 0
    zone_index = 0

    print(
        f"Skenario={args.scenario}, interval={args.interval} detik, "
        f"iterations={args.iterations or 'tanpa batas'}"
    )
    print("Tekan Ctrl+C untuk berhenti.")

    try:
        while args.iterations == 0 or iteration < args.iterations:
            zone = VALID_ZONES[zone_index % len(VALID_ZONES)]
            audio_path, source_label = pick_audio(args.scenario, zone)
            result = predict_audio(model, audio_path)
            show_result(zone, audio_path, source_label, result)

            if not args.dry_run:
                api_result = send_prediction(
                    api_url=args.api_url,
                    node_id=args.node_id,
                    zone=zone,
                    audio_path=audio_path,
                    source_label=source_label,
                    classification=result.classification,
                    confidence=result.confidence,
                )
                print(f"API berhasil. Event ID: {api_result['id']}")

            iteration += 1
            zone_index += 1
            if args.iterations == 0 or iteration < args.iterations:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nSimulasi dihentikan.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulasi ChickVoice memakai file WAV dan model CNN asli."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Prediksi satu file WAV.")
    source.add_argument(
        "--scenario",
        choices=tuple(SCENARIO_WEIGHTS),
        help="Jalankan skenario normal, outbreak, atau noisy.",
    )
    parser.add_argument("--zone", choices=VALID_ZONES, default="A")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="0 berarti berjalan terus.",
    )
    parser.add_argument("--node-id", default="CV-AUDIO-01")
    parser.add_argument("--api-url", default=f"{API_BASE_URL}/events")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.interval < 0.5:
        parser.error("--interval minimum 0.5 detik.")
    if args.iterations < 0:
        parser.error("--iterations tidak boleh negatif.")

    model = load_chickvoice_model()
    print("Model berhasil dimuat")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Urutan kelas: {list(CLASS_LABELS)}")

    if args.audio:
        run_single(args, model)
    else:
        run_scenario(args, model)


if __name__ == "__main__":
    main()
