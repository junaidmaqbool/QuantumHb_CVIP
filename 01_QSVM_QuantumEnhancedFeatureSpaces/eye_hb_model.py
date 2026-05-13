"""
eye_hb_model.py — QSVM (Quantum-Enhanced Feature Spaces)
Paper : Havlíček et al., Nature 567 (2019) | IBM Quantum
GitHub: qiskit-community/qiskit-machine-learning
Task  : classification OR regression
Note  : Uses Qiskit ZZFeatureMap + quantum kernel + scikit-learn SVM/SVR
"""
import numpy as np

def build_model(n_qubits: int = 4, task: str = "regression"):
    """
    Returns (kernel_fn, estimator) where:
      - kernel_fn(X_a, X_b) computes the quantum kernel matrix
      - estimator is an sklearn SVC/SVR that uses this kernel
    Call estimator.fit(X_train, y_train) and estimator.predict(X_test).
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from sklearn.svm import SVC, SVR

        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement="linear")
        qkernel     = FidelityQuantumKernel(feature_map=feature_map)

        if task == "classification":
            estimator = SVC(kernel=qkernel.evaluate, probability=True, C=1.0)
        else:
            estimator = SVR(kernel=qkernel.evaluate, C=1.0, epsilon=0.1)

        return qkernel, estimator

    except ImportError:
        # ── Fallback: RBF SVM (classical approximation) ──────────────────
        from sklearn.svm import SVC, SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        print("  [QSVM] qiskit-machine-learning not found → using RBF-SVM fallback")
        scaler = StandardScaler()
        if task == "classification":
            clf = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale")
        else:
            clf = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)
        model = Pipeline([("scaler", scaler), ("svm", clf)])
        return None, model
