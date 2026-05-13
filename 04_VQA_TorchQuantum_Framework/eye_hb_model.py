"""
eye_hb_model.py -- TorchQuantum VQA (Variational Quantum Algorithm)
Paper : Cerezo et al., Nature Reviews Physics 3 (2021) | MIT Han Lab
Task  : classification OR regression
Input : (batch, n_features)

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
Fallback chain:
  1. TorchQuantum (tq)   -- if installed
  2. PennyLane VQA       -- via quantum_compat (autoray patched)
  3. Classical MLP       -- always available
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.VQA_TQ")

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
    Returns TQVQANet / PLVQANet / ClassicalMLP depending on availability.
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    logger.info(
        "VQA_TQ build_model | task=%s n_qubits=%d n_layers=%d pennylane=%s",
        task, n_qubits, n_layers, PENNYLANE_AVAILABLE,
    )

    # ---- Level 1: TorchQuantum -----------------------------------------------
    try:
        import torchquantum as tq
        import torchquantum.functional as tqf
        logger.info("TorchQuantum version: %s", tq.__version__)

        class TQVQANet(nn.Module):
            _backend = "VQA-TorchQuantum"

            def __init__(self):
                super().__init__()
                self.reducer   = nn.Linear(n_features, n_qubits)
                self.bn        = nn.BatchNorm1d(n_qubits)
                self.rx_layers = nn.ModuleList([
                    tq.RX(has_params=True, trainable=True)
                    for _ in range(n_layers * n_qubits)])
                self.ry_layers = nn.ModuleList([
                    tq.RY(has_params=True, trainable=True)
                    for _ in range(n_layers * n_qubits)])
                self.rz_layers = nn.ModuleList([
                    tq.RZ(has_params=True, trainable=True)
                    for _ in range(n_layers * n_qubits)])
                self.measure   = tq.MeasureAll(tq.PauliZ)

            def forward(self, x):
                bsz  = x.shape[0]
                x    = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                qdev = tq.QuantumDevice(n_wires=n_qubits, bsz=bsz, device=x.device)
                qdev.reset_states(bsz)
                for i in range(n_qubits):
                    tqf.ry(qdev, wires=i, params=x[:, i].unsqueeze(-1))
                idx = 0
                for _l in range(n_layers):
                    for i in range(n_qubits):
                        self.rx_layers[idx](qdev, wires=i)
                        self.ry_layers[idx](qdev, wires=i)
                        self.rz_layers[idx](qdev, wires=i)
                        idx += 1
                    for i in range(n_qubits - 1):
                        tqf.cnot(qdev, wires=[i, i + 1])
                out = self.measure(qdev)[:, 0]
                # sigmoid maps model output to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

        logger.info("VQA: TorchQuantum model built")
        return TQVQANet()

    except ImportError:
        logger.info("TorchQuantum not installed -> trying PennyLane VQA")
    except Exception as tq_err:
        logger.warning("TorchQuantum build FAILED: %s -> trying PennyLane", tq_err)

    # ---- Level 2: PennyLane --------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            diff_method = get_diff_method()
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _circ(inputs, w):
                for i in range(n_qubits):
                    qml.RY(inputs[..., i], wires=i)
                for l in range(n_layers):
                    for i in range(n_qubits):
                        qml.Rot(*w[l, i], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(0))

            qlayer = qml.qnn.TorchLayer(_circ, {"w": (n_layers, n_qubits, 3)})

            class PLVQANet(nn.Module):
                _backend = "VQA-PennyLane"

                def __init__(self):
                    super().__init__()
                    self.reducer = nn.Linear(n_features, n_qubits)
                    self.bn      = nn.BatchNorm1d(n_qubits)
                    self.qlayer  = qlayer

                def forward(self, x):
                    x   = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                    out = self.qlayer(x)
                    # sigmoid maps model output to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

            logger.info("VQA PennyLane model built  (diff_method=%s)", diff_method)
            return PLVQANet()

        except Exception as pl_err:
            logger.warning("VQA PennyLane build FAILED: %s", pl_err)
            logger.debug(traceback.format_exc())

    # ---- Level 3: classical fallback -----------------------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "all quantum backends failed"
    logger.warning("VQA_TQ -> classical MLP fallback  (reason: %s)", reason)
    model = build_classical_mlp(n_features=n_features, n_hidden=64, task=task)
    model._backend = "VQA-classical-fallback"
    return model
