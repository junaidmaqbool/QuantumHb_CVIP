"""
eye_hb_model.py -- VQC (Variational Quantum Classifier)
Paper : Farhi & Neven, arXiv 1802.06002 (Google 2018)
Task  : classification (anemic/normal) OR regression (Hb g/dL)
Input : (batch, n_features) float tensor  <- from ResNet-18 + Linear reducer

ROOT CAUSE OF NumpyMimic ERROR
-------------------------------
PennyLane <= 0.35 calls autoray.autoray.NumpyMimic internally for numpy
dispatch. autoray >= 0.6 removed that private class, raising:
  AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'
This fires the instant pennylane is imported.

FIXES APPLIED
-------------
1. quantum_compat.py (root-level) is imported FIRST -- it injects a
   NumpyMimic stub into autoray before pennylane ever loads.
2. diff_method is health-checked: tries backprop, falls back to
   parameter-shift (avoids the broken autoray code path entirely).
3. Full classical MLP fallback when quantum backend is unavailable.
4. All quantum code is wrapped in try/except -- no hard crashes.
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.VQC")

# -- compatibility shim MUST be imported before pennylane ----------------------
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


def build_model(n_features=4, n_qubits=4, n_layers=2, task="regression"):
    """
    Returns VQCNet (quantum) or ClassicalMLP (fallback).
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    diff = get_diff_method()
    logger.info(
        "VQC build_model | task=%s n_qubits=%d n_layers=%d "
        "pennylane=%s diff_method=%s",
        task, n_qubits, n_layers, PENNYLANE_AVAILABLE, diff,
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            diff_method = get_diff_method()
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _circuit(inputs, weights):
                # Angle encoding
                for i in range(n_qubits):
                    qml.RY(inputs[..., i], wires=i)
                # Strongly-entangling variational layers
                for l in range(n_layers):
                    for i in range(n_qubits):
                        qml.RX(weights[l, i, 0], wires=i)
                        qml.RY(weights[l, i, 1], wires=i)
                        qml.RZ(weights[l, i, 2], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])  # ring entanglement
                return qml.expval(qml.PauliZ(0))

            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            qlayer = qml.qnn.TorchLayer(_circuit, weight_shapes)

            class VQCNet(nn.Module):
                _backend = "VQC-quantum"

                def __init__(self):
                    super().__init__()
                    self.reducer = nn.Linear(n_features, n_qubits)
                    self.bn = nn.BatchNorm1d(n_qubits)
                    self.qlayer = qlayer

                def forward(self, x):
                    x = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                    out = self.qlayer(x)  # (batch,)
                    # sigmoid bounds both tasks to [0,1]: matches normalised
                    # Hb labels and prevents out-of-range regression predictions.
                    return torch.sigmoid(out)

            logger.info("VQC quantum model ready  (diff_method=%s)", diff_method)
            return VQCNet()

        except Exception as exc:
            logger.warning("VQC quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback --------------------------------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("VQC -> classical MLP fallback  (reason: %s)", reason)
    model = build_classical_mlp(n_features=n_features, n_hidden=64, task=task)
    model._backend = "VQC-classical-fallback"
    return model
