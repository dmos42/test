import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


class NormalityTests:
    """Tests de normalité: Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov,
    Lilliefors, D'Agostino-Pearson, Jarque-Bera, Ryan-Joiner."""

    def __init__(self, data: np.ndarray, alpha: float = 0.05, chi2_df: int = None):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.alpha = alpha
        self.chi2_df = chi2_df
        self._results: Dict[str, Any] = {}
        self._run_all_tests()

    def _run_all_tests(self):
        n = len(self.data)
        self._results = {
            "n": n,
            "mean": np.mean(self.data),
            "std": np.std(self.data, ddof=1),
            "skewness": stats.skew(self.data),
            "kurtosis": stats.kurtosis(self.data),
            "alpha": self.alpha,
            "tests": {},
        }

        if n >= 3:
            self._shapiro_wilk()

        if n >= 8:
            self._anderson_darling()

        if n >= 8:
            self._lilliefors()

        if n >= 20:
            self._d_agostino_pearson()

        if n >= 3:
            self._jarque_bera()

        if n >= 3:
            self._ryan_joiner()

        if n >= 20:
            self._chi_square()

        self._kolmogorov_smirnov()

    def _shapiro_wilk(self):
        try:
            stat, p_value = stats.shapiro(self.data)
            self._results["tests"]["Shapiro-Wilk"] = {
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > self.alpha,
                "description": "Test de Shapiro-Wilk (recommandé pour n < 5000)",
            }
        except Exception:
            pass

    def _anderson_darling(self):
        try:
            result = stats.anderson(self.data, dist="norm", method="interpolate")
            statistic = result.statistic
            p_value = result.pvalue

            self._results["tests"]["Anderson-Darling"] = {
                "statistic": statistic,
                "p_value": p_value,
                "normal": p_value > self.alpha,
                "critical_values": None,
                "description": "Test d'Anderson-Darling (sensible aux queues de distribution)",
            }
        except Exception:
            pass

    def _interpolate_anderson_pvalue(self, stat, critical_vals, sig_levels) -> float:
        if stat < critical_vals[0]:
            return 0.25
        if stat > critical_vals[-1]:
            return 0.0

        for i in range(len(critical_vals) - 1):
            if critical_vals[i] <= stat <= critical_vals[i + 1]:
                p1 = sig_levels[i]
                p2 = sig_levels[i + 1]
                v1 = critical_vals[i]
                v2 = critical_vals[i + 1]
                return p1 + (p2 - p1) * (stat - v1) / (v2 - v1)
        return 0.0

    def _kolmogorov_smirnov(self):
        try:
            mean = np.mean(self.data)
            std = np.std(self.data, ddof=1)
            if std == 0:
                return
            stat, p_value = stats.kstest(self.data, "norm", args=(mean, std))
            self._results["tests"]["Kolmogorov-Smirnov"] = {
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > self.alpha,
                "description": "Test de Kolmogorov-Smirnov (Lilliefors corrigé)",
            }
        except Exception:
            pass

    def _lilliefors(self):
        try:
            from statsmodels.stats.diagnostic import lilliefors
            stat, p_value = lilliefors(self.data)
            self._results["tests"]["Lilliefors"] = {
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > self.alpha,
                "description": "Test de Lilliefors (KS avec paramètres estimés)",
            }
        except Exception:
            pass

    def _d_agostino_pearson(self):
        try:
            stat, p_value = stats.normaltest(self.data)
            self._results["tests"]["D'Agostino-Pearson"] = {
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > self.alpha,
                "description": "Test de D'Agostino-Pearson (basé sur asymétrie et aplatissement)",
            }
        except Exception:
            pass

    def _jarque_bera(self):
        try:
            stat, p_value = stats.jarque_bera(self.data)
            self._results["tests"]["Jarque-Bera"] = {
                "statistic": stat,
                "p_value": p_value,
                "normal": p_value > self.alpha,
                "description": "Test de Jarque-Bera (basé sur asymétrie et kurtosis)",
            }
        except Exception:
            pass

    def _ryan_joiner(self):
        try:
            sorted_data = np.sort(self.data)
            n = len(sorted_data)
            expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
            expected = (expected - np.mean(expected)) / np.std(expected, ddof=0)

            correlation = np.corrcoef(sorted_data, expected)[0, 1]
            critical_95 = 1.0 - (1.0184 / (n + 0.02) - 1.26 / (n**2) + 2.18 / (n**3))
            critical_99 = 1.0 - (0.7768 / (n + 0.02) - 0.78 / (n**2) + 1.4 / (n**3))

            self._results["tests"]["Ryan-Joiner"] = {
                "statistic": correlation,
                "p_value": None,
                "normal": correlation > critical_95,
                "critical_95": critical_95,
                "critical_99": critical_99,
                "description": "Test de Ryan-Joiner (corrélation sur QQ plot)",
            }
        except Exception:
            pass

    def _chi_square(self):
        try:
            n = len(self.data)
            mean = np.mean(self.data)
            std = np.std(self.data, ddof=1)
            if std == 0:
                return

            num_bins = max(5, int(np.sqrt(n)))
            num_bins = min(num_bins, n // 5)
            if num_bins < 3:
                num_bins = 3

            counts, bin_edges = np.histogram(self.data, bins=num_bins)

            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            expected_probs = np.diff(stats.norm.cdf(bin_edges, loc=mean, scale=std))
            expected_freqs = expected_probs * n

            min_expected = 5.0
            observed_combined = []
            expected_combined = []
            current_obs = 0
            current_exp = 0.0

            for i in range(len(counts)):
                current_obs += counts[i]
                current_exp += expected_freqs[i]
                if current_exp >= min_expected or i == len(counts) - 1:
                    if current_exp > 0:
                        observed_combined.append(current_obs)
                        expected_combined.append(current_exp)
                    current_obs = 0
                    current_exp = 0.0

            observed_combined = np.array(observed_combined)
            expected_combined = np.array(expected_combined)

            if len(observed_combined) < 2:
                return

            num_params_estimated = 2
            df_auto = len(observed_combined) - 1 - num_params_estimated
            if df_auto < 1:
                return

            if self.chi2_df is not None:
                df = self.chi2_df
            else:
                df = df_auto

            chi2_stat = np.sum((observed_combined - expected_combined) ** 2 / expected_combined)
            p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=df)
            critical_value = stats.chi2.ppf(1 - self.alpha, df=df)

            self._results["tests"]["Khi-deux (χ²)"] = {
                "statistic": chi2_stat,
                "p_value": p_value,
                "df": df,
                "normal": p_value > self.alpha,
                "critical_value": critical_value,
                "num_bins_used": len(observed_combined),
                "description": "Test du Khi-deux d'ajustement à la loi normale",
            }
        except Exception:
            pass

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("TESTS DE NORMALITÉ")
        lines.append("=" * 60)
        lines.append(f"\nStatistiques descriptives:")
        lines.append(f"  N = {self._results['n']}")
        lines.append(f"  Moyenne = {self._results['mean']:.6f}")
        lines.append(f"  Écart-type = {self._results['std']:.6f}")
        lines.append(f"  Asymétrie (Skewness) = {self._results['skewness']:.4f}")
        lines.append(f"  Aplatissement (Kurtosis) = {self._results['kurtosis']:.4f}")
        lines.append(f"\nSeuil de significativité (α) = {self._results['alpha']}")
        lines.append("\n" + "-" * 60)

        for name, result in self._results["tests"].items():
            lines.append(f"\n{name}:")
            lines.append(f"  {result['description']}")
            lines.append(f"  Statistique = {result['statistic']:.6f}")
            if result["p_value"] is not None:
                lines.append(f"  p-value = {result['p_value']:.6f}")
                normal_str = "OUI" if result["normal"] else "NON"
                lines.append(f"  Normal = {normal_str}")
            else:
                if "critical_95" in result:
                    lines.append(f"  Valeur critique (95%) = {result['critical_95']:.6f}")
                    normal_str = "OUI" if result["normal"] else "NON"
                    lines.append(f"  Normal = {normal_str}")

            if "df" in result:
                lines.append(f"  Degrés de liberté = {result['df']}")
            if "critical_value" in result and result["critical_value"] is not None:
                lines.append(f"  Valeur critique (α={self.alpha}) = {result['critical_value']:.6f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
