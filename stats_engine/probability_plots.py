import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


class ProbabilityPlot:
    """Graphique de probabilité pour différentes distributions"""

    def __init__(self, data: np.ndarray, distribution: str = "normal"):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.distribution = distribution
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        n = len(self.data)
        sorted_data = np.sort(self.data)
        percentiles = np.arange(1, n + 1) / (n + 1)
        z_scores = stats.norm.ppf(percentiles)

        if self.distribution == "normal":
            loc, scale = stats.norm.fit(self.data)
            theoretical = stats.norm.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'norm', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["μ", "σ"]
        elif self.distribution == "lognormal":
            shape, loc, scale = stats.lognorm.fit(self.data, floc=0)
            theoretical = stats.lognorm.ppf(percentiles, shape, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'lognorm', args=(shape, loc, scale))
            params = {"shape": shape, "loc": loc, "scale": scale}
            param_labels = ["σ", "loc", "μ"]
        elif self.distribution == "weibull_min":
            shape, loc, scale = stats.weibull_min.fit(self.data, floc=0)
            theoretical = stats.weibull_min.ppf(percentiles, shape, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'weibull_min', args=(shape, loc, scale))
            params = {"shape": shape, "loc": loc, "scale": scale}
            param_labels = ["β", "γ", "η"]
        elif self.distribution == "exponential":
            loc, scale = stats.expon.fit(self.data)
            theoretical = stats.expon.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'expon', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["loc", "λ"]
        elif self.distribution == "gamma":
            shape, loc, scale = stats.gamma.fit(self.data, floc=0)
            theoretical = stats.gamma.ppf(percentiles, shape, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'gamma', args=(shape, loc, scale))
            params = {"shape": shape, "loc": loc, "scale": scale}
            param_labels = ["α", "loc", "β"]
        elif self.distribution == "logistic":
            loc, scale = stats.logistic.fit(self.data)
            theoretical = stats.logistic.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'logistic', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["μ", "s"]
        elif self.distribution == "gumbel_max":
            loc, scale = stats.gumbel_r.fit(self.data)
            theoretical = stats.gumbel_r.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'gumbel_r', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["μ", "β"]
        elif self.distribution == "gumbel_min":
            loc, scale = stats.gumbel_l.fit(self.data)
            theoretical = stats.gumbel_l.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'gumbel_l', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["μ", "β"]
        elif self.distribution == "cauchy":
            loc, scale = stats.cauchy.fit(self.data)
            theoretical = stats.cauchy.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'cauchy', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["x₀", "γ"]
        elif self.distribution == "rayleigh":
            loc, scale = stats.rayleigh.fit(self.data, floc=0)
            theoretical = stats.rayleigh.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'rayleigh', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["loc", "σ"]
        elif self.distribution == "uniform":
            loc, scale = stats.uniform.fit(self.data)
            theoretical = stats.uniform.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'uniform', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["min", "width"]
        elif self.distribution == "t":
            df, loc, scale = stats.t.fit(self.data)
            theoretical = stats.t.ppf(percentiles, df, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 't', args=(df, loc, scale))
            params = {"df": df, "loc": loc, "scale": scale}
            param_labels = ["ν", "loc", "σ"]
        elif self.distribution == "laplace":
            loc, scale = stats.laplace.fit(self.data)
            theoretical = stats.laplace.ppf(percentiles, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'laplace', args=(loc, scale))
            params = {"loc": loc, "scale": scale}
            param_labels = ["μ", "b"]
        elif self.distribution == "beta":
            d_min = np.min(self.data)
            d_max = np.max(self.data)
            rng = d_max - d_min
            if rng == 0:
                raise ValueError("Toutes les données sont identiques, Beta impossible")
            eps = rng * 0.001
            scaled = (self.data - d_min + eps) / (rng + 2 * eps)
            a, b = stats.beta.fit(scaled, floc=0, fscale=1)
            theoretical = stats.beta.ppf(percentiles, a, b, 0, 1)
            theoretical_orig = d_min + theoretical * (d_max - d_min)
            ks_stat, ks_p = stats.kstest(self.data, 'beta', args=(a, b, d_min, d_max - d_min))
            params = {"a": a, "b": b, "min": d_min, "max": d_max}
            param_labels = ["α", "β", "min", "max"]
        elif self.distribution == "pareto":
            shape, loc, scale = stats.pareto.fit(self.data, floc=0)
            theoretical = stats.pareto.ppf(percentiles, shape, loc, scale)
            ks_stat, ks_p = stats.kstest(self.data, 'pareto', args=(shape, loc, scale))
            params = {"shape": shape, "loc": loc, "scale": scale}
            param_labels = ["α", "xm", "éch."]
        else:
            raise ValueError(f"Distribution '{self.distribution}' non supportée")

        self._results = {
            "sorted_data": sorted_data,
            "percentiles": percentiles,
            "z_scores": z_scores,
            "theoretical": theoretical,
            "params": params,
            "param_labels": param_labels,
            "ks_stat": ks_stat,
            "ks_p": ks_p,
            "n": n,
            "distribution": self.distribution,
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        dist_names = {
            "normal": "Normale", "weibull_min": "Weibull (2P)",
            "exponential": "Exponentielle", "lognormal": "Log-Normale",
            "gamma": "Gamma", "logistic": "Logistique",
            "gumbel_max": "Gumbel (max)", "gumbel_min": "Gumbel (min)",
            "cauchy": "Cauchy", "rayleigh": "Rayleigh",
            "uniform": "Uniforme", "t": "Student-t",
            "laplace": "Laplace", "beta": "Beta",
            "pareto": "Pareto",
        }
        lines = ["=" * 60, f"GRAPHIQUE DE PROBABILITÉ - {dist_names.get(r['distribution'], r['distribution']).upper()}", "=" * 60]
        lines.append(f"\nNombre d'observations : {r['n']}")
        lines.append(f"\nParamètres estimés :")
        for label, val in zip(r["param_labels"], r["params"].values()):
            lines.append(f"  {label} = {val:.6f}")
        lines.append(f"\nTest d'ajustement (Kolmogorov-Smirnov) :")
        lines.append(f"  Statistique KS = {r['ks_stat']:.6f}")
        lines.append(f"  p-value        = {r['ks_p']:.6f}")
        if r["ks_p"] > 0.05:
            lines.append(f"  ✓ L'ajustement est acceptable (p > 0.05)")
        else:
            lines.append(f"  ⚠ L'ajustement est rejeté (p ≤ 0.05)")
        lines.append("=" * 60)
        return "\n".join(lines)


class EWMAChart:
    """Carte de contrôle EWMA (Exponentially Weighted Moving Average)"""

    def __init__(self, data: np.ndarray, lambda_: float = 0.2, l: float = 3.0,
                 target: float = None):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.lambda_ = lambda_
        self.l = l
        self.target = target if target is not None else np.mean(self.data)
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        n = len(self.data)
        z = np.zeros(n)
        z[0] = self.data[0]
        for i in range(1, n):
            z[i] = self.lambda_ * self.data[i] + (1 - self.lambda_) * z[i - 1]

        sigma = np.std(self.data, ddof=1)
        cl = self.target

        ucls = []
        lcls = []
        for i in range(n):
            var_factor = (self.lambda_ / (2 - self.lambda_)) * (1 - (1 - self.lambda_) ** (2 * (i + 1)))
            se = sigma * np.sqrt(var_factor)
            ucls.append(cl + self.l * se)
            lcls.append(cl - self.l * se)

        self._results = {
            "n": n,
            "z_values": z.tolist(),
            "ucls": ucls,
            "lcls": lcls,
            "cl": cl,
            "sigma": sigma,
            "lambda_": self.lambda_,
            "l": self.l,
            "violations": {"rule_1": [i for i in range(n) if z[i] > ucls[i] or z[i] < lcls[i]]},
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "CARTE DE CONTRÔLE EWMA", "=" * 60]
        lines.append(f"\nNombre de points : {r['n']}")
        lines.append(f"λ (lambda) = {r['lambda_']:.2f}")
        lines.append(f"L = {r['l']:.1f}")
        lines.append(f"\nCentre (CL) = {r['cl']:.6f}")
        lines.append(f"σ = {r['sigma']:.6f}")
        lines.append(f"UCL (asymptotique) = {r['cl'] + r['l'] * r['sigma'] * np.sqrt(r['lambda_'] / (2 - r['lambda_'])):.6f}")
        lines.append(f"LCL (asymptotique) = {r['cl'] - r['l'] * r['sigma'] * np.sqrt(r['lambda_'] / (2 - r['lambda_'])):.6f}")
        if r["violations"]["rule_1"]:
            lines.append(f"\n⚠ {len(r['violations']['rule_1'])} point(s) hors contrôle")
        else:
            lines.append(f"\n✓ Procédé sous contrôle statistique")
        lines.append("=" * 60)
        return "\n".join(lines)


class CUSUMChart:
    """Carte de contrôle CUSUM (Cumulative Sum)"""

    def __init__(self, data: np.ndarray, k: float = 0.5, h: float = 4.0,
                 target: float = None):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.k = k
        self.h = h
        self.target = target if target is not None else np.mean(self.data)
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        n = len(self.data)
        sigma = np.std(self.data, ddof=1)
        k_sigma = self.k * sigma
        h_sigma = self.h * sigma

        s_pos = np.zeros(n)
        s_neg = np.zeros(n)

        for i in range(1, n):
            s_pos[i] = max(0, s_pos[i - 1] + (self.data[i] - self.target) - k_sigma)
            s_neg[i] = max(0, s_neg[i - 1] - (self.data[i] - self.target) - k_sigma)

        self._results = {
            "n": n,
            "s_pos": s_pos.tolist(),
            "s_neg": s_neg.tolist(),
            "cl": self.target,
            "sigma": sigma,
            "k": self.k,
            "h": self.h,
            "k_sigma": k_sigma,
            "h_sigma": h_sigma,
            "violations": {
                "high": [i for i in range(n) if s_pos[i] > h_sigma],
                "low": [i for i in range(n) if s_neg[i] > h_sigma],
            },
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "CARTE DE CONTRÔLE CUSUM", "=" * 60]
        lines.append(f"\nNombre de points : {r['n']}")
        lines.append(f"k = {r['k']:.2f} (kσ = {r['k_sigma']:.6f})")
        lines.append(f"h = {r['h']:.1f} (hσ = {r['h_sigma']:.6f})")
        lines.append(f"\nCentre (CL) = {r['cl']:.6f}")
        lines.append(f"σ = {r['sigma']:.6f}")
        high_violations = len(r["violations"]["high"])
        low_violations = len(r["violations"]["low"])
        if high_violations > 0:
            lines.append(f"\n⚠ Décalage vers le HAUT détecté ({high_violations} points)")
        if low_violations > 0:
            lines.append(f"\n⚠ Décalage vers le BAS détecté ({low_violations} points)")
        if high_violations == 0 and low_violations == 0:
            lines.append(f"\n✓ Procédé sous contrôle statistique")
        lines.append("=" * 60)
        return "\n".join(lines)
