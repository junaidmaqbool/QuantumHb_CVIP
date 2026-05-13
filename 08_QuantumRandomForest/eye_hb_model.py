"""
eye_hb_model.py -- Kernel-Based Quantum Random Forest (QRF)
Paper : Srikumar et al., Quantum Machine Intelligence, Springer (2023)
Task  : classification OR regression
Arch  : Ensemble of QSVM-leaf nodes forming a quantum decision forest

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
Uses parameter-shift diff_method (avoids autoray). Classical RandomForest
fallback when quantum backend unavailable.
Note: quantum kernel computation is slow on CPU; warned in logs.
"""

import sys
import os
import logging
import traceback
import numpy as np

logger = logging.getLogger("QuantumHb.QRandomForest")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from quantum_compat import (
        PENNYLANE_AVAILABLE, BACKEND_INFO,
    )
except ImportError:
    PENNYLANE_AVAILABLE = False
    BACKEND_INFO = {"fallback_reason": "quantum_compat.py not found"}


def build_model(n_qubits=4, n_estimators=10, task="regression"):
    """
    Returns (None, QuantumRandomForest) or (None, ClassicalRF).
    The first element is always None (no separate kernel object needed).
    """
    logger.info(
        "QRF build_model | task=%s n_qubits=%d n_estimators=%d pennylane=%s",
        task, n_qubits, n_estimators, PENNYLANE_AVAILABLE,
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml
            from sklearn.svm import SVC, SVR
            from sklearn.base import BaseEstimator

            dev = qml.device("default.qubit", wires=n_qubits)

            # Always use parameter-shift for kernel circuits -- avoids autoray
            @qml.qnode(dev, diff_method="parameter-shift")
            def _kernel_circuit(x1, x2):
                for i in range(n_qubits):
                    qml.RY(x1[i], wires=i)
                qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            def _qkernel(X_a, X_b):
                n, m = len(X_a), len(X_b)
                if n * m > 2000:
                    logger.warning(
                        "QRF kernel: %d x %d = %d evaluations -- may be slow on CPU",
                        n, m, n * m,
                    )
                K = np.zeros((n, m))
                for i, a in enumerate(X_a):
                    for j, b in enumerate(X_b):
                        probs = _kernel_circuit(a * 3.14159265, b * 3.14159265)
                        K[i, j] = float(probs[0])
                return K

            class QuantumRandomForest(BaseEstimator):
                """Ensemble of QSVMs -- each tree uses a random feature subset."""

                def __init__(self):
                    self.estimators_ = []
                    self.feat_idx_   = []
                    self.X_train_    = None

                def fit(self, X, y):
                    rng    = np.random.RandomState(42)
                    n_feat = X.shape[1]
                    sub    = max(1, n_feat // 2)
                    self.estimators_, self.feat_idx_ = [], []
                    for _ in range(n_estimators):
                        idx = rng.choice(n_feat, sub, replace=False)
                        Xs  = np.clip(X[:, idx], -1, 1)
                        Ktr = _qkernel(Xs, Xs)
                        if task == "classification":
                            clf = SVC(kernel="precomputed", probability=True, C=1.0)
                        else:
                            clf = SVR(kernel="precomputed", C=1.0, epsilon=0.1)
                        clf.fit(Ktr, y)
                        self.estimators_.append(clf)
                        self.feat_idx_.append(idx)
                    self.X_train_ = X
                    return self

                def predict(self, X):
                    preds = []
                    for clf, idx in zip(self.estimators_, self.feat_idx_):
                        Xs  = np.clip(X[:, idx], -1, 1)
                        Xtr = np.clip(self.X_train_[:, idx], -1, 1)
                        Kte = _qkernel(Xs, Xtr)
                        preds.append(clf.predict(Kte))
                    arr = np.array(preds)
                    if task == "classification":
                        return (arr.mean(0) > 0.5).astype(int)
                    # Clip ensemble mean to [0,1] normalised range to prevent
                    # out-of-range kernel predictions from inflating MAE.
                    return np.clip(arr.mean(0), 0.0, 1.0)

                def predict_proba(self, X):
                    probs = []
                    for clf, idx in zip(self.estimators_, self.feat_idx_):
                        Xs  = np.clip(X[:, idx], -1, 1)
                        Xtr = np.clip(self.X_train_[:, idx], -1, 1)
                        Kte = _qkernel(Xs, Xtr)
                        probs.append(clf.predict_proba(Kte)[:, 1])
                    return np.array(probs).mean(0)

            logger.info("QRF quantum model built")
            return None, QuantumRandomForest()

        except Exception as exc:
            logger.warning("QRF quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback --------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("QRF -> classical RandomForest fallback  (reason: %s)", reason)
    if task == "classification":
        return None, RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1)
    return None, RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1)
