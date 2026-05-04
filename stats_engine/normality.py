import numpy as np
from scipy import stats
from typing import Dict, Any


class NormalityTests:
    """Tests de normalité robustes : Shapiro-Wilk, Anderson-Darling,
    Lilliefors, D'Agostino-Pearson, Jarque-Bera, Ryan-Joiner, Khi-deux et KS indicatif.

    Améliorations :
    - exclusion explicite des NaN/Inf ;
    - statuts explicites pour tests non applicables ou en erreur ;
    - Anderson-Darling compatible SciPy classique avec p-value interpolée ;
    - KS renommé en indicatif lorsque les paramètres sont estimés sur les données.
    """

    def __init__(self, data: np.ndarray, alpha: float = 0.05, chi2_df: int = None):
        raw = np.asarray(data, dtype=float)
        finite_mask = np.isfinite(raw)
        self.n_input = int(raw.size)
        self.n_removed_non_finite = int(raw.size - np.sum(finite_mask))
        self.data = raw[finite_mask]
        self.alpha = float(alpha) if 0 < float(alpha) < 1 else 0.05
        self.chi2_df = chi2_df
        self._results: Dict[str, Any] = {}
        self._run_all_tests()

    def _safe_float(self, value):
        try:
            v = float(value)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    def _add_not_applicable(self, name: str, description: str, reason: str):
        self._results.setdefault("tests", {})[name] = {
            "statistic": None,
            "p_value": None,
            "normal": None,
            "status": "non_applicable",
            "reason": reason,
            "description": description,
        }

    def _add_error(self, name: str, description: str, exc: Exception):
        self._results.setdefault("tests", {})[name] = {
            "statistic": None,
            "p_value": None,
            "normal": None,
            "status": "error",
            "reason": str(exc),
            "description": description,
        }

    def _run_all_tests(self):
        n = len(self.data)
        if n > 0:
            mean = self._safe_float(np.mean(self.data))
            std = self._safe_float(np.std(self.data, ddof=1)) if n > 1 else None
            skewness = self._safe_float(stats.skew(self.data)) if n >= 3 and len(np.unique(self.data)) > 1 else None
            kurtosis = self._safe_float(stats.kurtosis(self.data)) if n >= 3 and len(np.unique(self.data)) > 1 else None
        else:
            mean = std = skewness = kurtosis = None
        self._results = {
            "n": n,
            "n_input": self.n_input,
            "n_removed_non_finite": self.n_removed_non_finite,
            "mean": mean,
            "std": std,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "alpha": self.alpha,
            "tests": {},
            "warnings": [],
        }
        if self.n_removed_non_finite:
            self._results["warnings"].append(
                f"{self.n_removed_non_finite} valeur(s) non finie(s) (NaN/Inf) exclue(s) de l'analyse."
            )
        if n < 3:
            self._add_not_applicable("Shapiro-Wilk", "Test de Shapiro-Wilk", "n < 3")
            self._kolmogorov_smirnov()
            return
        if len(np.unique(self.data)) <= 1:
            for name, desc in [
                ("Shapiro-Wilk", "Test de Shapiro-Wilk"),
                ("Anderson-Darling", "Test d'Anderson-Darling"),
                ("Lilliefors", "Test de Lilliefors"),
                ("D'Agostino-Pearson", "Test de D'Agostino-Pearson"),
                ("Jarque-Bera", "Test de Jarque-Bera"),
                ("Ryan-Joiner", "Test de Ryan-Joiner"),
                ("Khi-deux (χ²)", "Test du Khi-deux"),
                ("Kolmogorov-Smirnov indicatif", "KS indicatif"),
            ]:
                self._add_not_applicable(name, desc, "données constantes")
            return
        self._shapiro_wilk()
        if n >= 8:
            self._anderson_darling()
            self._lilliefors()
        else:
            self._add_not_applicable("Anderson-Darling", "Test d'Anderson-Darling", "n < 8")
            self._add_not_applicable("Lilliefors", "Test de Lilliefors", "n < 8")
        if n >= 20:
            self._d_agostino_pearson()
            self._chi_square()
        else:
            self._add_not_applicable("D'Agostino-Pearson", "Test de D'Agostino-Pearson", "n < 20")
            self._add_not_applicable("Khi-deux (χ²)", "Test du Khi-deux", "n < 20")
        self._jarque_bera()
        self._ryan_joiner()
        self._kolmogorov_smirnov()

    def _shapiro_wilk(self):
        name = "Shapiro-Wilk"; desc = "Test de Shapiro-Wilk (recommandé pour n < 5000)"
        try:
            stat, p_value = stats.shapiro(self.data)
            self._results["tests"][name] = {"statistic": float(stat), "p_value": float(p_value), "normal": bool(p_value > self.alpha), "status": "ok", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def _anderson_darling(self):
        name = "Anderson-Darling"; desc = "Test d'Anderson-Darling (sensible aux queues de distribution)"
        try:
            result = stats.anderson(self.data, dist="norm")
            statistic = float(result.statistic)
            sig_levels = np.asarray(result.significance_level, dtype=float) / 100.0
            p_value = self._interpolate_anderson_pvalue(statistic, np.asarray(result.critical_values, dtype=float), sig_levels)
            critical_at_alpha = float(np.interp(self.alpha, sig_levels[::-1], np.asarray(result.critical_values)[::-1]))
            self._results["tests"][name] = {
                "statistic": statistic,
                "p_value": p_value,
                "normal": bool(p_value > self.alpha),
                "critical_values": result.critical_values.tolist(),
                "significance_levels": result.significance_level.tolist(),
                "critical_value_alpha": critical_at_alpha,
                "status": "ok",
                "description": desc,
            }
        except Exception as e:
            self._add_error(name, desc, e)

    def _interpolate_anderson_pvalue(self, stat, critical_vals, sig_levels) -> float:
        # critical_vals augmente lorsque la p-value diminue. sig_levels est fourni en probabilités (ex. 0.15..0.01).
        if stat < critical_vals[0]:
            return float(sig_levels[0])
        if stat > critical_vals[-1]:
            return float(sig_levels[-1])
        for i in range(len(critical_vals) - 1):
            if critical_vals[i] <= stat <= critical_vals[i + 1]:
                p1, p2 = sig_levels[i], sig_levels[i + 1]
                v1, v2 = critical_vals[i], critical_vals[i + 1]
                return float(p1 + (p2 - p1) * (stat - v1) / (v2 - v1))
        return float(sig_levels[-1])

    def _kolmogorov_smirnov(self):
        name = "Kolmogorov-Smirnov indicatif"; desc = "KS indicatif avec paramètres estimés sur les données ; préférer Lilliefors pour le verdict."
        try:
            mean = np.mean(self.data); std = np.std(self.data, ddof=1)
            if not np.isfinite(std) or std <= 0:
                self._add_not_applicable(name, desc, "écart-type nul ou invalide")
                return
            stat, p_value = stats.kstest(self.data, "norm", args=(mean, std))
            self._results["tests"][name] = {"statistic": float(stat), "p_value": float(p_value), "normal": bool(p_value > self.alpha), "status": "indicatif", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def _lilliefors(self):
        name = "Lilliefors"; desc = "Test de Lilliefors (KS avec paramètres estimés)"
        try:
            from statsmodels.stats.diagnostic import lilliefors
            stat, p_value = lilliefors(self.data)
            self._results["tests"][name] = {"statistic": float(stat), "p_value": float(p_value), "normal": bool(p_value > self.alpha), "status": "ok", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def _d_agostino_pearson(self):
        name = "D'Agostino-Pearson"; desc = "Test de D'Agostino-Pearson (basé sur asymétrie et aplatissement)"
        try:
            stat, p_value = stats.normaltest(self.data)
            self._results["tests"][name] = {"statistic": float(stat), "p_value": float(p_value), "normal": bool(p_value > self.alpha), "status": "ok", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def _jarque_bera(self):
        name = "Jarque-Bera"; desc = "Test de Jarque-Bera (basé sur asymétrie et kurtosis)"
        try:
            stat, p_value = stats.jarque_bera(self.data)
            self._results["tests"][name] = {"statistic": float(stat), "p_value": float(p_value), "normal": bool(p_value > self.alpha), "status": "ok", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def _ryan_joiner(self):
        name = "Ryan-Joiner"; desc = "Test de Ryan-Joiner (corrélation sur QQ plot)"
        try:
            sorted_data = np.sort(self.data); n = len(sorted_data)
            expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
            expected = (expected - np.mean(expected)) / np.std(expected, ddof=0)
            correlation = float(np.corrcoef(sorted_data, expected)[0, 1])
            critical_95 = 1.0 - (1.0184 / (n + 0.02) - 1.26 / (n**2) + 2.18 / (n**3))
            critical_99 = 1.0 - (0.7768 / (n + 0.02) - 0.78 / (n**2) + 1.4 / (n**3))
            self._results["tests"][name] = {"statistic": correlation, "p_value": None, "normal": bool(correlation > critical_95), "critical_95": critical_95, "critical_99": critical_99, "status": "ok", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def _chi_square(self):
        name = "Khi-deux (χ²)"; desc = "Test du Khi-deux d'ajustement à la loi normale"
        try:
            n = len(self.data); mean = np.mean(self.data); std = np.std(self.data, ddof=1)
            if not np.isfinite(std) or std <= 0:
                self._add_not_applicable(name, desc, "écart-type nul ou invalide")
                return
            num_bins = max(5, int(np.sqrt(n))); num_bins = min(num_bins, max(3, n // 5))
            counts, bin_edges = np.histogram(self.data, bins=num_bins)
            expected_freqs = np.diff(stats.norm.cdf(bin_edges, loc=mean, scale=std)) * n
            observed_combined, expected_combined = [], []
            current_obs, current_exp = 0, 0.0
            for i in range(len(counts)):
                current_obs += counts[i]; current_exp += expected_freqs[i]
                if current_exp >= 5.0 or i == len(counts) - 1:
                    if current_exp > 0:
                        observed_combined.append(current_obs); expected_combined.append(current_exp)
                    current_obs, current_exp = 0, 0.0
            observed_combined = np.asarray(observed_combined); expected_combined = np.asarray(expected_combined)
            if len(observed_combined) < 2:
                self._add_not_applicable(name, desc, "classes attendues insuffisantes")
                return
            df_auto = len(observed_combined) - 1 - 2
            df = int(self.chi2_df) if self.chi2_df is not None else df_auto
            if df < 1:
                self._add_not_applicable(name, desc, "degrés de liberté insuffisants")
                return
            chi2_stat = float(np.sum((observed_combined - expected_combined) ** 2 / expected_combined))
            p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=df))
            critical_value = float(stats.chi2.ppf(1 - self.alpha, df=df))
            self._results["tests"][name] = {"statistic": chi2_stat, "p_value": p_value, "df": df, "normal": bool(p_value > self.alpha), "critical_value": critical_value, "num_bins_used": int(len(observed_combined)), "status": "ok", "description": desc}
        except Exception as e:
            self._add_error(name, desc, e)

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        def fmt(value, digits=6):
            return "n/a" if value is None else f"{float(value):.{digits}f}"
        lines = ["=" * 60, "TESTS DE NORMALITÉ", "=" * 60]
        lines.append("\nStatistiques descriptives:")
        lines.append(f" N = {self._results['n']} / entrée = {self._results.get('n_input', self._results['n'])}")
        if self._results.get("n_removed_non_finite", 0):
            lines.append(f" Valeurs non finies exclues = {self._results['n_removed_non_finite']}")
        lines.append(f" Moyenne = {fmt(self._results['mean'])}")
        lines.append(f" Écart-type = {fmt(self._results['std'])}")
        lines.append(f" Asymétrie (Skewness) = {fmt(self._results['skewness'], 4)}")
        lines.append(f" Aplatissement (Kurtosis) = {fmt(self._results['kurtosis'], 4)}")
        lines.append(f"\nSeuil de significativité (α) = {self._results['alpha']}")
        if self._results.get("warnings"):
            lines.append("\nAvertissements:")
            lines.extend([f" - {w}" for w in self._results["warnings"]])
        lines.append("\n" + "-" * 60)
        for name, result in self._results["tests"].items():
            lines.append(f"\n{name}:")
            lines.append(f" {result.get('description', '')}")
            status = result.get("status", "ok")
            if status not in ("ok", "indicatif"):
                lines.append(f" Statut = {status}")
                if result.get("reason"):
                    lines.append(f" Raison = {result['reason']}")
                continue
            if result.get("statistic") is not None:
                lines.append(f" Statistique = {result['statistic']:.6f}")
            if result.get("p_value") is not None:
                lines.append(f" p-value = {result['p_value']:.6f}")
            if result.get("critical_95") is not None:
                lines.append(f" Valeur critique (95%) = {result['critical_95']:.6f}")
            if result.get("df") is not None:
                lines.append(f" Degrés de liberté = {result['df']}")
            if result.get("critical_value") is not None:
                lines.append(f" Valeur critique (α={self.alpha}) = {result['critical_value']:.6f}")
            normal = result.get("normal")
            lines.append(f" Normal = {'OUI' if normal else 'NON' if normal is False else 'n/a'}")
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
