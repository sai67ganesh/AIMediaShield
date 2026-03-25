import json
import numpy as np
from sklearn.linear_model import LogisticRegression

# 🔧 MANUALLY fill these after computing raw scores
# Format: (raw_score, label)
# label: 0 = real, 1 = fake
DATA = [
    # REAL videos
    (0.18, 0), (0.21, 0), (0.24, 0), (0.26, 0),
    (0.22, 0), (0.19, 0), (0.25, 0),

    # FAKE videos
    (0.28, 1), (0.31, 1), (0.34, 1), (0.29, 1),
    (0.36, 1), (0.32, 1), (0.30, 1),
]

X = np.array([[d[0]] for d in DATA])
y = np.array([d[1] for d in DATA])

model = LogisticRegression(solver="lbfgs")
model.fit(X, y)

a = float(model.coef_[0][0])
b = float(model.intercept_[0])

weights = {"a": a, "b": b}

with open("calibration/calibrator_weights.json", "w") as f:
    json.dump(weights, f, indent=2)

print("[OK] Calibration trained")
print(weights)
