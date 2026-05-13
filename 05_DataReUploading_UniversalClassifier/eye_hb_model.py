"""
eye_hb_model.py -- Data Re-Uploading Universal Classifier
Paper : Perez-Salinas et al., Quantum 4, 226 (2020) | U. Barcelona
Task  : classification OR regression
Key   : Data re-uploaded at every layer -- single qubit is universal classifier

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
autoray >= 0.6 removed NumpyMimic; quantum_compat patches it before pennylane
is imported. diff_method auto-selected. Full classical fallback added.
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.DataReUploading")

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
                # sigmoid bounds both tasks to [0,1]: prevents out-of-range regression predictions.
                return torch.sigmoid(o)
        return _MLP()


def build_model(n_features=4, n_qubits=4, n_layers=4, task="regression"):
    """
    Returns DataReUploadNet (quantum) or ClassicalMLP (fallback).
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    logger.info(
        "DataReUploading build_model | task=%s n_qubits=%d n_layers=%d pennylane=%s",
        task, n_qubits, n_layers, PENNYLANE_AVAILABLE,
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            diff_method = get_diff_method()
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _circuit(inputs, weights, bias):
                # Data re-uploading: at each layer apply U(w*x + b) to every qubit
                # Core idea of Perez-Salinas et al. -- same data uploaded multiple
                # times with different learned scalings.
                for l in range(n_layers):
                    for q in range(n_qubits):
                        phi = weights[l, q] * inputs + bias[l, q]
                        qml.Rot(
                            phi[..., 0 % n_features],
                            phi[..., 1 % n_features],
                            phi[..., 2 % n_features],
                            wires=q,
                        )
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                return qml.expval(qml.PauliZ(0))

            weight_shapes = {
                "weights": (n_layers, n_qubits, n_features),
                "bias"   : (n_layers, n_qubits, n_features),
            }
            qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

            class DataReUploadNet(nn.Module):
                _backend = "DataReUploading-quantum"

                def __init__(self):
                    super().__init__()
                    self.reducer = nn.Linear(n_features, n_features)
                    self.bn      = nn.BatchNorm1d(n_features)
                    self.qlayer  = qlayer

                def forward(self, x):
                    x   = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                    out = self.qlayer(x)
                    # sigmoid maps expval [-1,1] to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

            logger.info("DataReUploading quantum model ready  (diff_method=%s)", diff_method)
            return DataReUploadNet()

        except Exception as exc:
            logger.warning("DataReUploading quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback --------------------------------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("DataReUploading -> classical MLP fallback  (reason: %s)", reason)
    model = build_classical_mlp(n_features=n_features, n_hidden=64, task=task)
    model._backend = "DataReUploading-classical-fallback"
    return model
