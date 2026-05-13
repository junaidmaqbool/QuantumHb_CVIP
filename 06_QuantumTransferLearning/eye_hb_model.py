"""
eye_hb_model.py -- Quantum Transfer Learning
Paper : Mari et al., Quantum 4, 340 (2020) | Xanadu AI
Task  : classification OR regression
Arch  : Classical ResNet-18 feature extractor + trainable dressed quantum circuit

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
Also fixed: the original model used a per-sample Python loop over the batch
which was slow and caused dimension mismatches. Replaced with TorchLayer
which handles batching internally via the torch interface.
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.QTransferLearning")

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
    Returns DressedQNet (quantum) or ClassicalMLP (fallback).
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    logger.info(
        "QTransferLearning build_model | task=%s n_qubits=%d n_layers=%d pennylane=%s",
        task, n_qubits, n_layers, PENNYLANE_AVAILABLE,
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            diff_method = get_diff_method()
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _dressed_circuit(inputs, vqc_w):
                # Dressed quantum circuit (Mari et al.):
                # angle encoding -> VQC -> n_qubits PauliZ expectation values
                # Classical pre/post layers handled outside by DressedQNet.
                for i in range(n_qubits):
                    qml.RY(inputs[..., i], wires=i)
                for l in range(n_layers):
                    for i in range(n_qubits):
                        qml.Rot(*vqc_w[l, i], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            weight_shapes = {"vqc_w": (n_layers, n_qubits, 3)}
            qlayer = qml.qnn.TorchLayer(_dressed_circuit, weight_shapes)

            class DressedQNet(nn.Module):
                """Classical-Quantum-Classical dressed circuit (Mari et al.)"""
                _backend = "QTransferLearning-quantum"

                def __init__(self):
                    super().__init__()
                    # pre-dress: down-project features to n_qubits
                    self.pre    = nn.Sequential(
                        nn.Linear(n_features, n_qubits),
                        nn.Tanh(),
                    )
                    self.qlayer = qlayer
                    # post-dress: map n_qubits expvals -> 1 output
                    self.post   = nn.Linear(n_qubits, 1)

                def forward(self, x):
                    x   = self.pre(x) * 3.14159265   # (B, n_qubits)
                    q   = self.qlayer(x)              # (B, n_qubits)
                    out = self.post(q).squeeze(-1)    # (B,)
                    # sigmoid maps expval [-1,1] to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

            logger.info("QTransferLearning quantum model ready  (diff_method=%s)", diff_method)
            return DressedQNet()

        except Exception as exc:
            logger.warning("QTransferLearning quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback --------------------------------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("QTransferLearning -> classical MLP fallback  (reason: %s)", reason)
    model = build_classical_mlp(n_features=n_features, n_hidden=64, task=task)
    model._backend = "QTransferLearning-classical-fallback"
    return model
