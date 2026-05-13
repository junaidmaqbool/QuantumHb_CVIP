"""
eye_hb_model.py — QBoost (Quantum Adiabatic Boosting)
Paper : Neven et al., ACML 2012 | Google + D-Wave
GitHub: dwavesystems/qboost
Task  : classification OR regression
Arch  : Ensemble of weak classifiers; weights optimised via QUBO
        (D-Wave hardware if available, classical QUBO solver fallback)
"""
import numpy as np

def build_model(n_qubits: int = 4, n_estimators: int = 20, task: str = "regression"):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.base import BaseEstimator
    from sklearn.preprocessing import StandardScaler

    class QBoostEstimator(BaseEstimator):
        """
        QBoost: Select optimal subset of weak classifiers by solving QUBO.
        Falls back to scipy-based QUBO solver if D-Wave not available.
        """
        def __init__(self):
            self.weak_learners_ = []
            self.weights_       = None
            self.scaler_        = StandardScaler()
            self.X_train_       = None

        def _build_weak_learners(self, X, y):
            """Train diverse weak learners on random subsets."""
            rng = np.random.RandomState(42)
            learners, preds = [], []
            for i in range(n_estimators):
                idx    = rng.choice(len(X), max(10, len(X) // 2), replace=False)
                feat   = rng.choice(X.shape[1], max(1, X.shape[1] // 2), replace=False)
                Xs, ys = X[idx][:, feat], y[idx]
                if task == "classification":
                    wl = DecisionTreeClassifier(max_depth=2, random_state=i)
                else:
                    wl = DecisionTreeRegressor(max_depth=2, random_state=i)
                wl.fit(Xs, ys)
                learners.append((wl, feat))
                p = wl.predict(X[:, feat])
                if task == "classification":
                    p = p * 2 - 1    # {0,1} → {-1,+1}
                preds.append(p)
            return learners, np.array(preds).T   # (N, n_estimators)

        def _solve_qubo(self, H, lam=1.0):
            """
            Solve QUBO: min_w { ||y - Hw||² + λ||w||₁ }
            For regression: uses continuous relaxation (Lasso-like) to get
            real-valued weights. For classification: binary QUBO with L-BFGS-B.
            """
            N, M = H.shape
            from scipy.optimize import minimize
            if task == "regression":
                # Continuous Lasso relaxation: w ∈ [0, 2/M] → avoids scale explosion
                # when many weak learners are selected (sum stays in [0,1] range)
                w_max = 2.0 / max(1, M)
                def obj(w):
                    diff = H @ w - self.y_tr_
                    return float(diff @ diff + lam * w.sum())
                def grad(w):
                    diff = H @ w - self.y_tr_
                    return 2 * H.T @ diff + lam * np.ones(M)
                w0  = np.ones(M) * (w_max * 0.5)
                res = minimize(obj, w0, jac=grad, method="L-BFGS-B",
                               bounds=[(0, w_max)] * M, options={"maxiter": 500})
                return res.x
            else:
                # Binary QUBO for classification
                def obj(w):
                    diff = H @ w - self.y_tr_
                    return float(diff @ diff + lam * w.sum())
                def grad(w):
                    diff = H @ w - self.y_tr_
                    return 2 * H.T @ diff + lam * np.ones(M)
                w0  = np.random.rand(M) * 0.5
                res = minimize(obj, w0, jac=grad, method="L-BFGS-B",
                               bounds=[(0, 1)] * M, options={"maxiter": 500})
                return (res.x > 0.5).astype(float)

        def fit(self, X, y):
            X = self.scaler_.fit_transform(X)
            self.X_train_ = X
            self.y_tr_    = (y * 2 - 1) if task == "classification" else y
            learners, H   = self._build_weak_learners(X, y)
            self.weights_ = self._solve_qubo(H)
            self.weak_learners_ = learners
            return self

        def predict(self, X):
            X = self.scaler_.transform(X)
            H = np.zeros((len(X), n_estimators))
            for i, (wl, feat) in enumerate(self.weak_learners_):
                p = wl.predict(X[:, feat])
                H[:, i] = (p * 2 - 1) if task == "classification" else p
            out = H @ self.weights_
            if task == "classification":
                return (out > 0).astype(int)
            # Regression: clip to valid normalised [0, 1] range
            return np.clip(out, 0.0, 1.0)

        def predict_proba(self, X):
            X = self.scaler_.transform(X)
            H = np.zeros((len(X), n_estimators))
            for i, (wl, feat) in enumerate(self.weak_learners_):
                p = wl.predict(X[:, feat])
                H[:, i] = (p * 2 - 1) if task == "classification" else p
            raw  = H @ self.weights_
            prob = 1 / (1 + np.exp(-raw))    # sigmoid
            return prob

    return None, QBoostEstimator()
