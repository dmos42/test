import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional


class ChiSquareTest:
    """Test du Chi-carré avec sélection du nombre de pas (bins)"""

    def __init__(self, data: np.ndarray, distribution: str = "uniform",
                 n_bins: int = 10, expected_counts: np.ndarray = None):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.distribution = distribution
        self.n_bins = n_bins
        self.expected_counts = expected_counts
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        if self.expected_counts is not None:
            observed = np.array([np.sum(self.data == v) for v in range(len(self.expected_counts))])
            expected = np.array(self.expected_counts, dtype=float)
            df = len(observed) - 1
            valid_mask = expected > 0
            observed = observed[valid_mask]
            expected = expected[valid_mask]
            chi2, p_value = stats.chisquare(observed, f_exp=expected)
        else:
            observed_counts, bin_edges = np.histogram(self.data, bins=self.n_bins)

            if self.distribution == "uniform":
                lo, hi = np.min(self.data), np.max(self.data)
                expected_counts = np.full(self.n_bins, len(self.data) / self.n_bins)
            elif self.distribution == "normal":
                loc, scale = stats.norm.fit(self.data)
                expected_counts = np.diff(stats.norm.cdf(bin_edges, loc, scale)) * len(self.data)
            elif self.distribution == "exponential":
                loc, scale = stats.expon.fit(self.data)
                expected_counts = np.diff(stats.expon.cdf(bin_edges, loc, scale)) * len(self.data)
            else:
                expected_counts = np.full(self.n_bins, len(self.data) / self.n_bins)

            combined_mask = expected_counts >= 5
            while np.sum(combined_mask) > 0 and np.any(~combined_mask):
                new_observed = []
                new_expected = []
                i = 0
                while i < len(observed_counts):
                    if combined_mask[i]:
                        new_observed.append(observed_counts[i])
                        new_expected.append(expected_counts[i])
                        i += 1
                    else:
                        obs_sum = observed_counts[i]
                        exp_sum = expected_counts[i]
                        j = i + 1
                        while j < len(expected_counts) and exp_sum < 5:
                            obs_sum += observed_counts[j]
                            exp_sum += expected_counts[j]
                            j += 1
                        new_observed.append(obs_sum)
                        new_expected.append(exp_sum)
                        i = j
                observed_counts = np.array(new_observed)
                expected_counts = np.array(new_expected)
                combined_mask = expected_counts >= 5
                if len(observed_counts) <= 1:
                    break

            observed = observed_counts
            expected = expected_counts
            df = len(observed) - 1 - (2 if self.distribution in ["normal", "exponential"] else 0)
            chi2, p_value = stats.chisquare(observed, f_exp=expected)

            self._results["bin_edges"] = bin_edges.tolist()
            self._results["observed_per_bin"] = observed_counts.tolist()
            self._results["expected_per_bin"] = expected_counts.tolist()

        self._results.update({
            "chi2_stat": chi2,
            "p_value": p_value,
            "df": df,
            "n": len(self.data),
            "n_bins_used": len(observed),
            "observed": observed.tolist(),
            "expected": expected.tolist(),
        })

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, f"TEST DU KHI-DEUX ({self.distribution.upper()})", "=" * 60]
        lines.append(f"\nNombre d'observations : {r['n']}")
        lines.append(f"Nombre de pas initial : {self.n_bins}")
        lines.append(f"Nombre de pas utilisé : {r['n_bins_used']}")
        lines.append(f"\nStatistique χ² = {r['chi2_stat']:.6f}")
        lines.append(f"Degrés de liberté = {r['df']}")
        lines.append(f"p-value = {r['p_value']:.6f}")
        if r["p_value"] > 0.05:
            lines.append(f"\n✓ L'ajustement est acceptable (p > 0.05)")
        else:
            lines.append(f"\n⚠ L'ajustement est rejeté (p ≤ 0.05)")

        if "observed_per_bin" in r:
            lines.append(f"\nDétail par pas :")
            lines.append(f"  {'Pas':<20s} | {'Observé':>8s} | {'Attendu':>8s}")
            lines.append(f"  {'-'*20} | {'-'*8} | {'-'*8}")
            for i in range(len(r["observed_per_bin"])):
                lines.append(f"  Bin {i+1:<15.4f} | {r['observed_per_bin'][i]:8d} | {r['expected_per_bin'][i]:8.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class ChiSquareIndependence:
    """Test d'indépendance du Chi-carré"""

    def __init__(self, contingency_table: np.ndarray):
        self.table = np.array(contingency_table, dtype=float)
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        chi2, p_value, dof, expected = stats.chi2_contingency(self.table)
        self._results = {
            "chi2_stat": chi2,
            "p_value": p_value,
            "df": dof,
            "observed": self.table.tolist(),
            "expected": expected.tolist(),
            "n_rows": self.table.shape[0],
            "n_cols": self.table.shape[1],
            "total": np.sum(self.table),
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "TEST D'INDÉPENDANCE DU KHI-DEUX", "=" * 60]
        lines.append(f"\nTableau de contingence : {r['n_rows']} x {r['n_cols']}")
        lines.append(f"Total des observations : {r['total']}")
        lines.append(f"\nStatistique χ² = {r['chi2_stat']:.6f}")
        lines.append(f"Degrés de liberté = {r['df']}")
        lines.append(f"p-value = {r['p_value']:.6f}")
        if r["p_value"] > 0.05:
            lines.append(f"\n✓ Indépendance acceptée (p > 0.05)")
        else:
            lines.append(f"\n⚠ Dépendance significative détectée (p ≤ 0.05)")
        lines.append("=" * 60)
        return "\n".join(lines)
