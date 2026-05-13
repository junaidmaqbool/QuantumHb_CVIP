"""
eye_hb_model.py — QSVM Medical Classifier (Bonus)
Source: GlazeDonuts/QSVM  (trained on medical breast-cancer data)
Task  : classification OR regression
Note  : Same quantum kernel as Model 1 but with optimised hyperparams
        tuned for medical binary classification tasks.
"""
import numpy as np

def build_model(n_qubits: int = 4, task: str = "regression"):
    try:
        from qiskit.circuit.library import PauliFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from sklearn.svm import SVC, SVR

        # PauliFeatureMap is richer than ZZFeatureMap — better for medical data
        feature_map = PauliFeatureMap(
            feature_dimension=n_qubits, reps=2,
            paulis=["Z", "ZZ"])
        qkernel = FidelityQuantumKernel(feature_map=feature_map)

        if task == "classification":
            estimator = SVC(kernel=qkernel.evaluate, probability=True, C=5.0)
        else:
            estimator = SVR(kernel=qkernel.evaluate, C=5.0, epsilon=0.05)
        return qkernel, estimator

    except (ImportError, Exception) as e:
        print(f"  [QSVM-Medical] Fallback: {e}")
        from sklearn.svm import SVC, SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        clf    = SVC(kernel="rbf", C=5.0, probability=True) if task == "classification" \
                 else SVR(kernel="rbf", C=5.0, epsilon=0.05)
        return None, Pipeline([("scaler", scaler), ("svm", clf)])
