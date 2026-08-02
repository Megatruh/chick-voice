from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "chickvoice_mfcc_cnn.h5"
DATASET_DIR = (
    ROOT_DIR
    / "Poultry_Vocalization Signal_Dataset_for_Early_Disease_Detection"
    / "Chicken_Audio_Dataset"
)
DB_PATH = Path(__file__).resolve().with_name("chickvoice.db")
API_BASE_URL = "http://127.0.0.1:8000"
VALID_ZONES = ("A", "B", "C", "D")
CLASS_LABELS = ("Healthy", "Noise", "Unhealthy")
N_MFCC = 40
MAX_PAD_LEN = 200
