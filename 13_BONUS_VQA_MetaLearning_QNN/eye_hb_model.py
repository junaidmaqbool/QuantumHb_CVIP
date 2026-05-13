"""
eye_hb_model.py -- VQA Meta-Learning QNN (Bonus)
Paper : Verdon et al., "Learning to learn with QNNs" (2019)
Task  : classification OR regression
Arch  : Classical LSTM meta-learner initialises VQC parameters for fast convergence.

ROOT CAUSE / FIXES  -- see quantum_compat.py for full explanation.
Also fixed: the original code accessed qlayer.qnode_weights -- this key was
renamed in PennyLane 0.36+. Now uses a safe accessor that tries multiple
attribute names before falling back to named_parameters().
Classical LSTM+MLP fallback added.
"""

import sys
import os
import logging
import traceback
import torch
import torch.nn as nn

logger = logging.getLogger("QuantumHb.MetaLearning")

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


def _get_qlayer_weight(qlayer, key="weights"):
    """
    Safely fetch a trainable tensor from a TorchLayer.
    Handles PennyLane < 0.36 (qnode_weights dict) and
    PennyLane >= 0.36 (_qnode_weights / named_parameters).
    """
    for attr in ("qnode_weights", "_qnode_weights"):
        d = getattr(qlayer, attr, None)
        if isinstance(d, dict) and key in d:
            return d[key]
    for name, param in qlayer.named_parameters():
        if key in name:
            return param
    raise AttributeError("Cannot locate '%s' in TorchLayer parameters" % key)


def build_model(n_features=4, n_qubits=4, n_layers=2, task="regression"):
    """
    Returns MetaLearnQNN (quantum) or MetaLearnClassical (fallback).
    Interface: model(x: Tensor[B, n_features]) -> Tensor[B]
    """
    logger.info(
        "MetaLearning build_model | task=%s n_qubits=%d n_layers=%d pennylane=%s",
        task, n_qubits, n_layers, PENNYLANE_AVAILABLE,
    )

    # ---- quantum path --------------------------------------------------------
    if PENNYLANE_AVAILABLE:
        try:
            import pennylane as qml

            diff_method = get_diff_method()
            dev = qml.device("default.qubit", wires=n_qubits)

            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def _vqc(inputs, weights):
                for i in range(n_qubits):
                    qml.RY(inputs[..., i], wires=i)
                for l in range(n_layers):
                    for i in range(n_qubits):
                        qml.Rot(*weights[l, i], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(0))

            qlayer = qml.qnn.TorchLayer(_vqc, {"weights": (n_layers, n_qubits, 3)})

            class MetaLearnQNN(nn.Module):
                """LSTM meta-learner generates warm-start VQC weights."""
                _backend = "MetaLearning-quantum"

                def __init__(self):
                    super().__init__()
                    self.reducer   = nn.Linear(n_features, n_qubits)
                    self.bn        = nn.BatchNorm1d(n_qubits)
                    param_dim      = n_layers * n_qubits * 3
                    self.meta_lstm = nn.LSTM(n_qubits, param_dim, batch_first=True)
                    self.qlayer    = qlayer
                    self._warmed   = False

                def _warmstart(self, x_sample):
                    with torch.no_grad():
                        seq = x_sample[:min(8, len(x_sample))].unsqueeze(0)
                        out, _ = self.meta_lstm(seq)
                        init = out[0, -1]
                        try:
                            w_param = _get_qlayer_weight(self.qlayer, "weights")
                            w_param.data.copy_(init.view(w_param.shape))
                        except AttributeError as attr_err:
                            logger.warning("MetaLearning warmstart skipped: %s", attr_err)

                def forward(self, x):
                    x = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
                    if not self._warmed and self.training:
                        self._warmstart(x.detach())
                        self._warmed = True
                    out = self.qlayer(x)
                    # sigmoid maps model output to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

            logger.info("MetaLearning quantum model ready  (diff_method=%s)", diff_method)
            return MetaLearnQNN()

        except Exception as exc:
            logger.warning("MetaLearning quantum build FAILED: %s", exc)
            logger.debug(traceback.format_exc())

    # ---- classical fallback: LSTM meta-learner + MLP -------------------------
    reason = BACKEND_INFO.get("fallback_reason") or "quantum build error"
    logger.warning("MetaLearning -> classical LSTM+MLP fallback  (reason: %s)", reason)

    class MetaLearnClassical(nn.Module):
        """Classical analogue: LSTM warm-starts an MLP instead of a VQC."""
        _backend = "MetaLearning-classical-fallback"

        def __init__(self):
            super().__init__()
            self.reducer   = nn.Linear(n_features, 32)
            self.bn        = nn.BatchNorm1d(32)
            self.meta_lstm = nn.LSTM(32, 64, batch_first=True)
            self.mlp       = nn.Sequential(
                nn.Linear(32, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            self._warmed   = False

        def _warmstart(self, x_sample):
            with torch.no_grad():
                seq = x_sample[:min(8, len(x_sample))].unsqueeze(0)
                out, _ = self.meta_lstm(seq)
                init = out[0, -1]
                w = self.mlp[0].weight
                cols = min(32, init.shape[0])
                w.data[:, :cols] += 0.01 * init[:cols].unsqueeze(0)

        def forward(self, x):
            x = torch.tanh(self.bn(self.reducer(x))) * 3.14159265
            if not self._warmed and self.training:
                self._warmstart(x.detach())
                self._warmed = True
            out = self.mlp(x).squeeze(-1)
            # sigmoid maps model output to [0,1], matching normalised Hb labels.
                    return torch.sigmoid(out)

    logger.info("MetaLearning classical LSTM+MLP fallback built")
    return MetaLearnClassical()
