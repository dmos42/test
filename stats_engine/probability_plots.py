import numpy as np
from scipy import stats
from typing import Dict, Any


class ProbabilityPlot:
    """Graphique de probabilité robuste pour différentes distributions."""

    SUPPORTED_DISTRIBUTIONS = {
        "normal", "lognormal", "weibull_min", "exponential", "gamma", "logistic",
        "gumbel_max", "gumbel_min", "cauchy", "rayleigh", "uniform", "t",
        "laplace", "beta", "pareto",
    }
    POSITIVE_DISTRIBUTIONS = {"lognormal", "weibull_min", "gamma", "rayleigh", "pareto"}

    def __init__(self, data: np.ndarray, distribution: str = "normal"):
        raw = np.asarray(data, dtype=float)
        self.n_input = int(raw.size)
        finite_mask = np.isfinite(raw)
        self.n_removed_non_finite = int(raw.size - np.sum(finite_mask))
        self.data = raw[finite_mask]
        self.distribution = str(distribution or "normal")
        self._results: Dict[str, Any] = {}
        self._compute()

    def _invalid(self, errors, warnings=None):
        self._results = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings or [],
            "n": int(len(self.data)),
            "n_input": self.n_input,
            "n_removed_non_finite": self.n_removed_non_finite,
            "distribution": self.distribution,
            "sorted_data": np.sort(self.data),
            "percentiles": np.array([]),
            "z_scores": np.array([]),
            "theoretical": np.array([]),
            "params": None,
            "param_labels": [],
            "ks_stat": None,
            "ks_p": None,
        }

    def _finalize(self, sorted_data, percentiles, z_scores, theoretical, params, param_labels, ks_stat, ks_p, warnings=None):
        self._results = {
            "is_valid": True,
            "errors": [],
            "warnings": warnings or [],
            "sorted_data": sorted_data,
            "percentiles": percentiles,
            "z_scores": z_scores,
            "theoretical": theoretical,
            "params": params,
            "param_labels": param_labels,
            "ks_stat": float(ks_stat) if np.isfinite(ks_stat) else None,
            "ks_p": float(ks_p) if np.isfinite(ks_p) else None,
            "n": int(len(self.data)),
            "n_input": self.n_input,
            "n_removed_non_finite": self.n_removed_non_finite,
            "distribution": self.distribution,
            "ks_note": "p-value indicative car les paramètres sont estimés sur les mêmes données.",
        }

    def _compute(self):
        warnings = []
        if self.n_removed_non_finite:
            warnings.append(f"{self.n_removed_non_finite} valeur(s) non finie(s) (NaN/Inf) exclue(s).")
        if self.distribution not in self.SUPPORTED_DISTRIBUTIONS:
            self._invalid([f"Distribution '{self.distribution}' non supportée."], warnings)
            return
        n = len(self.data)
        if n < 3:
            self._invalid(["Au moins 3 valeurs numériques finies sont nécessaires."], warnings)
            return
        if len(np.unique(self.data)) <= 1:
            self._invalid(["Toutes les données sont identiques : graphique de probabilité non calculable."], warnings)
            return
        if self.distribution in self.POSITIVE_DISTRIBUTIONS and np.any(self.data <= 0):
            self._invalid([f"La distribution {self.distribution} exige des données strictement positives."], warnings)
            return

        sorted_data = np.sort(self.data)
        percentiles = np.arange(1, n + 1) / (n + 1)
        z_scores = stats.norm.ppf(percentiles)

        try:
            d = self.distribution
            if d == "normal":
                loc, scale = stats.norm.fit(self.data)
                theoretical = stats.norm.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "norm", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["μ", "σ"]
            elif d == "lognormal":
                shape, loc, scale = stats.lognorm.fit(self.data, floc=0)
                theoretical = stats.lognorm.ppf(percentiles, shape, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "lognorm", args=(shape, loc, scale))
                params = {"shape": float(shape), "loc": float(loc), "scale": float(scale)}; labels = ["σ", "loc", "μ"]
            elif d == "weibull_min":
                shape, loc, scale = stats.weibull_min.fit(self.data, floc=0)
                theoretical = stats.weibull_min.ppf(percentiles, shape, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "weibull_min", args=(shape, loc, scale))
                params = {"shape": float(shape), "loc": float(loc), "scale": float(scale)}; labels = ["β", "γ", "η"]
            elif d == "exponential":
                loc, scale = stats.expon.fit(self.data)
                theoretical = stats.expon.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "expon", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["loc", "λ"]
            elif d == "gamma":
                shape, loc, scale = stats.gamma.fit(self.data, floc=0)
                theoretical = stats.gamma.ppf(percentiles, shape, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "gamma", args=(shape, loc, scale))
                params = {"shape": float(shape), "loc": float(loc), "scale": float(scale)}; labels = ["α", "loc", "β"]
            elif d == "logistic":
                loc, scale = stats.logistic.fit(self.data)
                theoretical = stats.logistic.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "logistic", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["μ", "s"]
            elif d == "gumbel_max":
                loc, scale = stats.gumbel_r.fit(self.data)
                theoretical = stats.gumbel_r.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "gumbel_r", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["μ", "β"]
            elif d == "gumbel_min":
                loc, scale = stats.gumbel_l.fit(self.data)
                theoretical = stats.gumbel_l.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "gumbel_l", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["μ", "β"]
            elif d == "cauchy":
                loc, scale = stats.cauchy.fit(self.data)
                theoretical = stats.cauchy.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "cauchy", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["x₀", "γ"]
            elif d == "rayleigh":
                loc, scale = stats.rayleigh.fit(self.data, floc=0)
                theoretical = stats.rayleigh.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "rayleigh", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["loc", "σ"]
            elif d == "uniform":
                loc, scale = stats.uniform.fit(self.data)
                theoretical = stats.uniform.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "uniform", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["min", "width"]
            elif d == "t":
                df, loc, scale = stats.t.fit(self.data)
                theoretical = stats.t.ppf(percentiles, df, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "t", args=(df, loc, scale))
                params = {"df": float(df), "loc": float(loc), "scale": float(scale)}; labels = ["ν", "loc", "σ"]
            elif d == "laplace":
                loc, scale = stats.laplace.fit(self.data)
                theoretical = stats.laplace.ppf(percentiles, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "laplace", args=(loc, scale))
                params = {"loc": float(loc), "scale": float(scale)}; labels = ["μ", "b"]
            elif d == "beta":
                d_min = float(np.min(self.data)); d_max = float(np.max(self.data)); rng = d_max - d_min
                if rng <= 0:
                    self._invalid(["Toutes les données sont identiques, Beta impossible."], warnings); return
                eps = max(rng * 0.001, np.finfo(float).eps)
                scaled = (self.data - d_min + eps) / (rng + 2 * eps)
                a, b, loc, scale = stats.beta.fit(scaled, floc=0, fscale=1)
                theoretical_scaled = stats.beta.ppf(percentiles, a, b, loc, scale)
                theoretical = d_min - eps + theoretical_scaled * (rng + 2 * eps)
                ks_stat, ks_p = stats.kstest(scaled, "beta", args=(a, b, loc, scale))
                params = {"a": float(a), "b": float(b), "min": d_min, "max": d_max}; labels = ["α", "β", "min", "max"]
            elif d == "pareto":
                shape, loc, scale = stats.pareto.fit(self.data, floc=0)
                theoretical = stats.pareto.ppf(percentiles, shape, loc, scale)
                ks_stat, ks_p = stats.kstest(self.data, "pareto", args=(shape, loc, scale))
                params = {"shape": float(shape), "loc": float(loc), "scale": float(scale)}; labels = ["α", "xm", "éch."]
            if not np.all(np.isfinite(theoretical)):
                self._invalid(["Quantiles théoriques non finis : ajustement non exploitable."], warnings); return
            self._finalize(sorted_data, percentiles, z_scores, theoretical, params, labels, ks_stat, ks_p, warnings)
        except Exception as e:
            self._invalid([f"Erreur lors de l'ajustement : {e}"], warnings)

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        dist_names = {
            "normal": "Normale", "weibull_min": "Weibull (2P)", "exponential": "Exponentielle",
            "lognormal": "Log-Normale", "gamma": "Gamma", "logistic": "Logistique",
            "gumbel_max": "Gumbel (max)", "gumbel_min": "Gumbel (min)", "cauchy": "Cauchy",
            "rayleigh": "Rayleigh", "uniform": "Uniforme", "t": "Student-t", "laplace": "Laplace",
            "beta": "Beta", "pareto": "Pareto",
        }
        lines = ["=" * 60, f"GRAPHIQUE DE PROBABILITÉ - {dist_names.get(r['distribution'], r['distribution']).upper()}", "=" * 60]
        if not r.get("is_valid", False):
            lines.append("\nÉtude non valide :")
            lines.extend([f"- {e}" for e in r.get("errors", [])])
            return "\n".join(lines)
        lines.append(f"\nNombre d'observations : {r['n']}")
        if r.get("warnings"):
            lines.append("\nAvertissements :")
            lines.extend([f"- {w}" for w in r["warnings"]])
        lines.append("\nParamètres estimés :")
        for label, val in zip(r["param_labels"], r["params"].values()):
            lines.append(f" {label} = {val:.6f}")
        lines.append("\nTest d'ajustement (Kolmogorov-Smirnov indicatif) :")
        lines.append(f" Statistique KS = {r['ks_stat']:.6f}")
        lines.append(f" p-value = {r['ks_p']:.6f}")
        lines.append("=" * 60)
        return "\n".join(lines)


class EWMAChart:
    """Carte de contrôle EWMA robuste."""

    def __init__(self, data: np.ndarray, lambda_: float = 0.2, l: float = 3.0, target: float = None):
        raw = np.asarray(data, dtype=float)
        self.n_input = int(raw.size)
        self.data = raw[np.isfinite(raw)]
        self.n_removed_non_finite = int(raw.size - len(self.data))
        self.lambda_ = float(lambda_)
        self.l = float(l)
        self.target = target
        self._results: Dict[str, Any] = {}
        self._compute()

    def _invalid(self, errors):
        self._results = {"is_valid": False, "errors": errors, "n": int(len(self.data)), "n_input": self.n_input, "n_removed_non_finite": self.n_removed_non_finite}

    def _compute(self):
        if len(self.data) < 2:
            self._invalid(["Au moins 2 valeurs numériques finies sont nécessaires pour EWMA."]); return
        if not (0 < self.lambda_ <= 1):
            self._invalid(["lambda_ doit être dans l'intervalle ]0 ; 1]."]); return
        if self.l <= 0:
            self._invalid(["L doit être strictement positif."]); return
        n = len(self.data)
        target = float(self.target) if self.target is not None and np.isfinite(self.target) else float(np.mean(self.data))
        z = np.zeros(n); z[0] = self.data[0]
        for i in range(1, n):
            z[i] = self.lambda_ * self.data[i] + (1 - self.lambda_) * z[i - 1]
        sigma = float(np.std(self.data, ddof=1))
        ucls, lcls = [], []
        for i in range(n):
            var_factor = (self.lambda_ / (2 - self.lambda_)) * (1 - (1 - self.lambda_) ** (2 * (i + 1)))
            se = sigma * np.sqrt(var_factor)
            ucls.append(target + self.l * se); lcls.append(target - self.l * se)
        self._results = {
            "is_valid": True, "errors": [], "warnings": [], "n": n, "n_input": self.n_input,
            "n_removed_non_finite": self.n_removed_non_finite, "z_values": z.tolist(), "ucls": ucls,
            "lcls": lcls, "cl": target, "sigma": sigma, "lambda_": self.lambda_, "l": self.l,
            "violations": {"rule_1": [i for i in range(n) if z[i] > ucls[i] or z[i] < lcls[i]]},
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "CARTE DE CONTRÔLE EWMA", "=" * 60]
        if not r.get("is_valid", False):
            lines.extend([f"- {e}" for e in r.get("errors", [])]); return "\n".join(lines)
        lines.append(f"\nNombre de points : {r['n']}")
        lines.append(f"λ (lambda) = {r['lambda_']:.2f}")
        lines.append(f"L = {r['l']:.1f}")
        lines.append(f"\nCentre (CL) = {r['cl']:.6f}")
        lines.append(f"σ = {r['sigma']:.6f}")
        if r["violations"]["rule_1"]:
            lines.append(f"\n⚠ {len(r['violations']['rule_1'])} point(s) hors contrôle")
        else:
            lines.append("\n✓ Procédé sous contrôle statistique")
        lines.append("=" * 60)
        return "\n".join(lines)


class CUSUMChart:
    """Carte de contrôle CUSUM robuste."""

    def __init__(self, data: np.ndarray, k: float = 0.5, h: float = 4.0, target: float = None):
        raw = np.asarray(data, dtype=float)
        self.n_input = int(raw.size)
        self.data = raw[np.isfinite(raw)]
        self.n_removed_non_finite = int(raw.size - len(self.data))
        self.k = float(k); self.h = float(h); self.target = target
        self._results: Dict[str, Any] = {}
        self._compute()

    def _invalid(self, errors):
        self._results = {"is_valid": False, "errors": errors, "n": int(len(self.data)), "n_input": self.n_input, "n_removed_non_finite": self.n_removed_non_finite}

    def _compute(self):
        if len(self.data) < 2:
            self._invalid(["Au moins 2 valeurs numériques finies sont nécessaires pour CUSUM."]); return
        if self.k < 0 or self.h <= 0:
            self._invalid(["k doit être >= 0 et h doit être strictement positif."]); return
        n = len(self.data)
        target = float(self.target) if self.target is not None and np.isfinite(self.target) else float(np.mean(self.data))
        sigma = float(np.std(self.data, ddof=1))
        k_sigma = self.k * sigma; h_sigma = self.h * sigma
        s_pos = np.zeros(n); s_neg = np.zeros(n)
        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i - 1] + (self.data[i] - target) - k_sigma)
            s_neg[i] = max(0, s_neg[i - 1] - (self.data[i] - target) - k_sigma)
        self._results = {
            "is_valid": True, "errors": [], "warnings": [], "n": n, "n_input": self.n_input,
            "n_removed_non_finite": self.n_removed_non_finite, "s_pos": s_pos.tolist(), "s_neg": s_neg.tolist(),
            "cl": target, "sigma": sigma, "k": self.k, "h": self.h, "k_sigma": k_sigma, "h_sigma": h_sigma,
            "violations": {"high": [i for i in range(n) if s_pos[i] > h_sigma], "low": [i for i in range(n) if s_neg[i] > h_sigma]},
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "CARTE DE CONTRÔLE CUSUM", "=" * 60]
        if not r.get("is_valid", False):
            lines.extend([f"- {e}" for e in r.get("errors", [])]); return "\n".join(lines)
        lines.append(f"\nNombre de points : {r['n']}")
        lines.append(f"k = {r['k']:.2f} (kσ = {r['k_sigma']:.6f})")
        lines.append(f"h = {r['h']:.1f} (hσ = {r['h_sigma']:.6f})")
        lines.append(f"\nCentre (CL) = {r['cl']:.6f}")
        high, low = len(r["violations"]["high"]), len(r["violations"]["low"])
        if high: lines.append(f"\n⚠ Décalage vers le HAUT détecté ({high} points)")
        if low: lines.append(f"\n⚠ Décalage vers le BAS détecté ({low} points)")
        if not high and not low: lines.append("\n✓ Procédé sous contrôle statistique")
        lines.append("=" * 60)
        return "\n".join(lines)
