import numpy as np
from scipy import stats
from typing import Dict, Any, List


class CorrelationMatrix:
    """Calcule et analyse une matrice de corrélation robuste.

    Améliorations v2 :
    - filtrage pairwise avec np.isfinite, donc NaN et Inf exclus ;
    - validation explicite de la méthode ;
    - gestion des colonnes constantes et paires insuffisantes via un statut explicite ;
    - matrices numériques finies pour éviter les problèmes d'affichage dans l'IHM.
    """

    VALID_METHODS = {"pearson", "spearman", "kendall"}

    def __init__(self, data: np.ndarray, labels: List[str] = None, method: str = "pearson"):
        raw = np.asarray(data, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        self.data = raw
        self.labels = labels or [f"Var {i+1}" for i in range(self.data.shape[1])]
        self.method = str(method).lower() if method is not None else "pearson"
        self._results: Dict[str, Any] = {}
        self._compute()

    def _invalid(self, errors, warnings=None):
        n_vars = self.data.shape[1] if self.data.ndim == 2 else 0
        self._results = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings or [],
            "correlation_matrix": np.eye(n_vars),
            "p_matrix": np.zeros((n_vars, n_vars)),
            "pair_status": {},
            "labels": self.labels,
            "method": self.method,
            "n_obs": self.data.shape[0] if self.data.ndim == 2 else 0,
            "n_vars": n_vars,
            "strong_correlations": [],
        }

    def _compute_pair(self, x, y):
        if len(x) < 3:
            return 0.0, 1.0, "non_applicable", "moins de 3 paires finies"
        if np.std(x, ddof=1) <= 0 or np.std(y, ddof=1) <= 0:
            return 0.0, 1.0, "non_applicable", "variable constante"
        if self.method == "pearson":
            r, p = stats.pearsonr(x, y)
        elif self.method == "spearman":
            r, p = stats.spearmanr(x, y)
        else:
            r, p = stats.kendalltau(x, y)
        if not np.isfinite(r) or not np.isfinite(p):
            return 0.0, 1.0, "non_applicable", "corrélation non définie"
        return float(r), float(p), "ok", ""

    def _compute(self):
        errors, warnings = [], []
        if self.data.ndim != 2 or self.data.shape[1] < 1:
            errors.append("Les données doivent être un tableau 2D avec au moins une variable.")
        if self.method not in self.VALID_METHODS:
            errors.append(
                f"Méthode de corrélation non supportée : {self.method}. "
                f"Valeurs autorisées : {sorted(self.VALID_METHODS)}"
            )
        n_vars = self.data.shape[1] if self.data.ndim == 2 else 0
        if len(self.labels) != n_vars:
            warnings.append("Nombre de libellés différent du nombre de variables ; libellés automatiques utilisés.")
            self.labels = [f"Var {i+1}" for i in range(n_vars)]
        if errors:
            self._invalid(errors, warnings)
            return

        corr_matrix = np.eye(n_vars, dtype=float)
        p_matrix = np.zeros((n_vars, n_vars), dtype=float)
        n_pairwise = np.zeros((n_vars, n_vars), dtype=int)
        pair_status = {}

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                mask = np.isfinite(self.data[:, i]) & np.isfinite(self.data[:, j])
                x = self.data[mask, i]
                y = self.data[mask, j]
                n_pairwise[i, j] = n_pairwise[j, i] = int(len(x))
                r, p, status, reason = self._compute_pair(x, y)
                corr_matrix[i, j] = corr_matrix[j, i] = r
                p_matrix[i, j] = p_matrix[j, i] = p
                pair_status[f"{self.labels[i]}|{self.labels[j]}"] = {
                    "status": status,
                    "reason": reason,
                    "n_pairwise": int(len(x)),
                    "r": r,
                    "p": p,
                }
                if status != "ok":
                    warnings.append(f"{self.labels[i]} vs {self.labels[j]} : {reason}")

        strong = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr_matrix[i, j]) > 0.7 and pair_status[f"{self.labels[i]}|{self.labels[j]}"]["status"] == "ok":
                    strong.append({"var1": self.labels[i], "var2": self.labels[j], "r": corr_matrix[i, j], "p": p_matrix[i, j]})

        self._results = {
            "is_valid": True,
            "errors": [],
            "warnings": warnings,
            "correlation_matrix": corr_matrix,
            "p_matrix": p_matrix,
            "n_pairwise": n_pairwise,
            "pair_status": pair_status,
            "labels": self.labels,
            "method": self.method,
            "n_obs": self.data.shape[0],
            "n_vars": n_vars,
            "strong_correlations": strong,
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, f"MATRICE DE CORRÉLATION ({r['method'].upper()})", "=" * 60]
        if not r.get("is_valid", False):
            lines.append("\nÉtude non valide :")
            lines.extend([f"- {e}" for e in r.get("errors", [])])
            return "\n".join(lines)
        lines.append(f"\nNombre d'observations : {r['n_obs']}")
        lines.append(f"Nombre de variables : {r['n_vars']}")
        if r.get("warnings"):
            lines.append("\nAvertissements :")
            for w in r["warnings"][:12]:
                lines.append(f" - {w}")
        header = " " + " ".join([f"{l:>10s}" for l in r['labels']])
        lines.append("\nMatrice de corrélation :\n")
        lines.append(header)
        for i in range(r['n_vars']):
            row = f"{r['labels'][i]:<10s}"
            for j in range(r['n_vars']):
                row += f"{r['correlation_matrix'][i, j]:10.4f} "
            lines.append(row)
        lines.append("\nMatrice des p-values :\n")
        lines.append(header)
        for i in range(r['n_vars']):
            row = f"{r['labels'][i]:<10s}"
            for j in range(r['n_vars']):
                val = r['p_matrix'][i, j]
                row += "   <0.001 " if val < 0.001 else f"{val:10.4f} "
            lines.append(row)
        if r["strong_correlations"]:
            lines.append("\n⚠ Corrélations fortes (|r| > 0.7) :")
            for c in r["strong_correlations"]:
                lines.append(f" {c['var1']} ↔ {c['var2']} : r = {c['r']:.4f} (p = {c['p']:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)
