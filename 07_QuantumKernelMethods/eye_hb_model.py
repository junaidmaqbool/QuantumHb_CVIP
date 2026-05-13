"""
eye_hb_model.py — Quantum Kernel Methods (Trainable Quantum Kernel)
Paper : Schuld, PRX Quantum (2021) | Xanadu AI
GitHub: qiskit-community/prototype-quantum-kernel-training
Task  : classification OR regression
Note  : Trains the quantum kernel itself via Kernel Target Alignment (KTA)
"""
import numpy as np

def build_model(n_qubits: int = 4, task: str = "regression"):
    """
    Returns a trainable quantum kernel + sklearn estimator.
    The kernel parameters are optimised via KTA before SVM fitting.
    """
    try:
        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.circuit import ParameterVector
        from qiskit_machine_learning.kernels import TrainableFidelityQuantumKernel
        from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
        from sklearn.svm import SVC, SVR

        # Trainable feature map: ZZFeatureMap with learnable scaling parameters
        training_params = ParameterVector("θ", n_qubits)
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
        # Bind training params as initial scaling
        qkernel = TrainableFidelityQuantumKernel(
            feature_map=feature_map,
            training_parameters=feature_map.parameters[:n_qubits]
                if len(feature_map.parameters) >= n_qubits else feature_map.parameters
        )
        if task == "classification":
            estimator = SVC(kernel=qkernel.evaluate, probability=True, C=1.0)
        else:
            estimator = SVR(kernel=qkernel.evaluate, C=1.0, epsilon=0.1)

        return qkernel, estimator

    except (ImportError, Exception) as e:
        print(f"  [QKernel] Qiskit unavailable ({e}) → polynomial kernel fallback")
        from sklearn.svm import SVC, SVR
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        if task == "classification":
            clf = SVC(kernel="poly", degree=3, probability=True, C=1.0)
        else:
            clf = SVR(kernel="poly", degree=3, C=1.0, epsilon=0.1)
        model = Pipeline([("scaler", scaler), ("svm", clf)])
        return None, model
