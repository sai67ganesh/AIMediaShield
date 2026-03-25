import json
import math
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "calibrator_weights.json"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calibrate(raw_score: float) -> float:
    """
    Maps raw model score → calibrated probability
    """
    if not WEIGHTS_PATH.exists():
        return raw_score  # safe fallback

    with open(WEIGHTS_PATH, "r") as f:
        w = json.load(f)

    a = w["a"]
    b = w["b"]

    calibrated = sigmoid(a * raw_score + b)
    return round(float(calibrated), 4)
