"""
eye_hb_model.py -- QCNN (Quantum Convolutional Neural Network)
Paper : Cong, Choi & Lukin, Nature Physics 15 (2019)
Task  : classification OR regression
Input : (batch, n_features)
Note  : n_qubits must be a power of 2 (4, 8, 16)

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
autoray >= 0.6 removed NumpyMimic; quantum_compat patches it before
pennylane is imported. diff_method auto-selected. Classical fallback added.
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.QCNN")

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


def build_model(n_features=4, n_qubits=4, n_layers=1, task="regression"):
    """
    Returns QCNNNet (quantum) or ClassicalMLP (fallback).
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    logger.info(
        "QCNN build_model | task=%s n_qubits=%d pennylane=%s diff_method=%s",
        task, n_qubits, PENNYLANE_AVAILABLE, get_diff_method(),
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            # Clamp to supported sizes
            if n_qubits not in (4, 8, 16):
                logger.warning("QCNN: n_qubits=%d not in {4,8,16}, clamping to 4", n_qubits)
                n_qubits = 4

            diff_method = get_diff_method()
            dev = qml.device("default.qubit", wires=n_qubits)

            def _conv_layer(params, wires):
                qml.RX(params[0], wires=wires[0])
                qml.RX(params[1], wires=wires[1])
                qml.RZ(params[2], wires=wires[0])
                qml.RZ(params[3], wires=wires[1])
                qml.CNOT(wires=[wires[0], wires[1]])
                qml.RY(params[4], wires=wires[0])
                qml.RY(params[5], wires=wires[1])
                qml.CNOT(wires=[wires[1], wires[0]])

            def _pool_layer(params, wires):
                qml.CRZ(params[0], wires=[wires[0], wires[1]])
                qml.PauliX(wires=wires[0])
                qml.CRX(params[1], wires=[wires[0], wires[1]])

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _circuit(inputs, conv_w, pool_w):
                for i in range(n_qubits):
                    qml.RY(inputs[..., i % n_qubits], wires=i)
                active = list(range(n_qubits))
                c_idx, p_idx = 0, 0
                while len(active) > 1:
                    for j in range(0, len(active) - 1, 2):
                        _conv_layer(conv_w[c_idx], [active[j], active[j + 1]])
                        c_idx += 1
                    new_active = []
                    for j in range(0, len(active) - 1, 2):
                        _pool_layer(pool_w[p_idx], [active[j], active[j + 1]])
                        new_active.append(active[j + 1])
                        p_idx += 1
                    active = new_active
                return qml.expval(qml.PauliZ(active[0]))

            n_stages   = (n_qubits - 1).bit_length()
            n_conv_ops = sum(n_qubits >> (s + 1) for s in range(n_stages))
            n_pool_ops = n_conv_ops
            wshapes = {
                "conv_w": (n_conv_ops, 8),
                "pool_w": (n_pool_ops, 2),
            }
            qlayer = qml.qnn.TorchLayer(_circuit, wshapes)

            class QCNNNet(nn.Module):
                _backend = "QCNN-quantum"

                def __init__(self):
                    super().__init__()
                    self.reducer = nn.Linear(n_features, n_qubits)
                    self.bn = nn.BatchNorm1d(n_qubits)
                    self.qlayer = qlayer

                def forward(self, x):
                    x = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                    out = self.qlayer(x)
                    # sigmoid maps expval [-1,1] to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

            logger.info("QCNN quantum model ready  (diff_method=%s)", diff_method)
            return QCNNNet()

        except Exception as exc:
            logger.warning("QCNN quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback --------------------------------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("QCNN -> classical MLP fallback  (reason: %s)", reason)
    model = build_classical_mlp(n_features=n_features, n_hidden=64, task=task)
    model._backend = "QCNN-classical-fallback"
    return model
