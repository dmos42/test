import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


class CorrelationMatrix:
    """Calcule et analyse une matrice de corrélation"""

    def __init__(self, data: np.ndarray, labels: List[str] = None, method: str = "pearson"):
        self.data = np.array(data, dtype=float)
        self.labels = labels or [f"Var {i+1}" for i in range(self.data.shape[1])]
        self.method = method
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        n_vars = self.data.shape[1]
        corr_matrix = np.zeros((n_vars, n_vars))
        p_matrix = np.ones((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                elif i < j:
                    mask = ~(np.isnan(self.data[:, i]) | np.isnan(self.data[:, j]))
                    x = self.data[mask, i]
                    y = self.data[mask, j]
                    if len(x) > 2:
                        if self.method == "pearson":
                            r, p = stats.pearsonr(x, y)
                        elif self.method == "spearman":
                            r, p = stats.spearmanr(x, y)
                        else:
                            r, p = stats.kendalltau(x, y)
                        corr_matrix[i, j] = corr_matrix[j, i] = r
                        p_matrix[i, j] = p_matrix[j, i] = p

        self._results = {
            "correlation_matrix": corr_matrix,
            "p_matrix": p_matrix,
            "labels": self.labels,
            "method": self.method,
            "n_obs": self.data.shape[0],
            "n_vars": n_vars,
        }

        # Déterminer les corrélations fortes
        self._results["strong_correlations"] = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if abs(corr_matrix[i, j]) > 0.7:
                    self._results["strong_correlations"].append({
                        "var1": self.labels[i],
                        "var2": self.labels[j],
                        "r": corr_matrix[i, j],
                        "p": p_matrix[i, j],
                    })

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, f"MATRICE DE CORRÉLATION ({r['method'].upper()})", "=" * 60]
        lines.append(f"\nNombre d'observations : {r['n_obs']}")
        lines.append(f"Nombre de variables : {r['n_vars']}")
        lines.append(f"\nMatrice de corrélation :\n")

        header = "         " + "  ".join([f"{l:>10s}" for l in r['labels']])
        lines.append(header)
        for i in range(r['n_vars']):
            row = f"{r['labels'][i]:<10s}"
            for j in range(r['n_vars']):
                val = r['correlation_matrix'][i, j]
                row += f"{val:10.4f}  "
            lines.append(row)

        lines.append(f"\nMatrice des p-values :\n")
        lines.append(header)
        for i in range(r['n_vars']):
            row = f"{r['labels'][i]:<10s}"
            for j in range(r['n_vars']):
                val = r['p_matrix'][i, j]
                if val < 0.001:
                    row += f"   <0.001  "
                else:
                    row += f"{val:10.4f}  "
            lines.append(row)

        if r["strong_correlations"]:
            lines.append(f"\n⚠ Corrélations fortes (|r| > 0.7) :")
            for c in r["strong_correlations"]:
                lines.append(f"  {c['var1']} ↔ {c['var2']} : r = {c['r']:.4f} (p = {c['p']:.4f})")

        lines.append("=" * 60)
        return "\n".join(lines)
