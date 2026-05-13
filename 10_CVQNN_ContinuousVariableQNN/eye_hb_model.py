"""
eye_hb_model.py -- Continuous-Variable Quantum Neural Network (CV-QNN)
Paper : Killoran et al., Physical Review Research 1 (2019) | Xanadu
Task  : classification OR regression
Note  : CV-QNN naturally suits REGRESSION; uses PennyLane gaussian device
        or default.qubit as fallback.

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
autoray >= 0.6 removed NumpyMimic; quantum_compat patches it before
pennylane is imported. Full classical MLP fallback added.
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.CVQNN")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from quantum_compat import (
        PENNYLANE_AVAILABLE, BACKEND_INFO,
        get_diff_method, build_classical_mlp,
    )
except ImportError:
    PENNYLANE_AVAILABLE = False
    BACKEND_INFO = {"fallback_reason": "quantum_compat.py not found"}
    get_diff_method = lambda: "none"

    def build_classical_mlp(n_features, n_hidden=64, task="regression"):
        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_features, n_hidden),
                    nn.GELU(),
                    nn.Linear(n_hidden, n_hidden // 2),
                    nn.GELU(),
                    nn.Linear(n_hidden // 2, 1),
                )
            def forward(self, x):
                o = self.net(x).squeeze(-1)
                # sigmoid for both tasks — keeps regression output in [0,1]
                return torch.sigmoid(o)
        return _MLP()


def build_model(n_features=4, n_qubits=4, n_layers=2, task="regression"):
    """
    Returns CVQNNNet (quantum) or ClassicalMLP (fallback).
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    logger.info(
        "CVQNN build_model | task=%s n_qubits=%d n_layers=%d pennylane=%s",
        task, n_qubits, n_layers, PENNYLANE_AVAILABLE,
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            diff_method = get_diff_method()

            # Try true CV gaussian device; fall back to default.qubit
            dev = None
            try:
                dev = qml.device("strawberryfields.fock", wires=n_qubits, cutoff_dim=5)
                logger.info("CVQNN: using strawberryfields.fock device")
            except Exception as sf_err:
                logger.info("CVQNN: strawberryfields unavailable (%s) -> default.qubit", sf_err)
                dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _cv_circuit(inputs, weights):
                # CV-QNN layer: Interferometer -> Squeezing -> Interferometer -> Displacement
                # Approximated with gate-based rotations for CPU compatibility.
                for i in range(n_qubits):
                    qml.RX(inputs[..., i], wires=i)
                for l in range(n_layers):
                    # Interferometer-like: alternating CNOT pairs
                    for i in range(0, n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(1, n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    # Squeezing-like: parameterized single-qubit rotations
                    for i in range(n_qubits):
                        qml.RX(weights[l, i, 0], wires=i)
                        qml.RY(weights[l, i, 1], wires=i)
                    # Displacement-like: phase shifts
                    for i in range(n_qubits):
                        qml.PhaseShift(weights[l, i, 2], wires=i)
                return qml.expval(qml.PauliZ(0))

            qlayer = qml.qnn.TorchLayer(
                _cv_circuit, {"weights": (n_layers, n_qubits, 3)})

            class CVQNNNet(nn.Module):
                _backend = "CVQNN-quantum"

                def __init__(self):
                    super().__init__()
                    self.reducer = nn.Linear(n_features, n_qubits)
                    self.bn      = nn.BatchNorm1d(n_qubits)
                    self.qlayer  = qlayer
                    # NOTE: removed learnable hb_scale — it caused predictions to
                    # grow far outside the normalised [0,1] Hb range (MAE ~45 g/dL).
                    # sigmoid maps the PauliZ expval [-1,1] cleanly to [0,1].

                def forward(self, x):
                    x   = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                    out = self.qlayer(x)
                    # sigmoid bounds both classification and regression to [0,1],
                    # matching normalised Hb labels and preventing out-of-range preds.
                    return torch.sigmoid(out)

            logger.info("CVQNN quantum model ready  (diff_method=%s)", diff_method)
            return CVQNNNet()

        except Exception as exc:
            logger.warning("CVQNN quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback --------------------------------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("CVQNN -> classical MLP fallback  (reason: %s)", reason)
    model = build_classical_mlp(n_features=n_features, n_hidden=64, task=task)
    model._backend = "CVQNN-classical-fallback"
    return model
