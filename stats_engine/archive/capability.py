
"""
Module capability.py amélioré pour StatPro / MiniQual.

Objectifs :
- sécuriser les entrées ;
- éviter les divisions par zéro / NaN / inf ;
- enrichir les résultats pour les rapports ;
- conserver la compatibilité avec l'API existante : CapabilityAnalysis(...).get_results() / get_summary().
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any, List

import numpy as np
from scipy import stats


DISPLAY_LABELS = {
    "cp": "Cp",
    "cpl": "Cpk inférieur / Cpl",
    "cpu": "Cpk supérieur / Cpu",
    "cpk": "Cpk global",
    "pp": "Pp",
    "ppl": "Ppk inférieur / Ppl",
    "ppu": "Ppk supérieur / Ppu",
    "ppk": "Ppk global",
    "cpm": "Cpm",
}


class CapabilityAnalysis:
    """
    Analyse de capabilité procédé.

    Cp/Cpk : indices basés sur la variabilité intra-processus.
    Pp/Ppk : indices basés sur la variabilité globale.
    Cpm : indice tenant compte de l'écart à la cible.

    Les résultats sont retournés sous forme de dictionnaire via get_results().
    Les erreurs bloquantes ne provoquent pas de crash : elles sont stockées dans
    les champs `errors`, `warnings`, `is_valid` et `error`.
    """

    VALID_ESTIMATION_METHODS = {"rbar", "sbar", "pooled"}
    VALID_PPK_METHODS = {"within", "overall"}

    def __init__(
        self,
        data: np.ndarray,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        target: Optional[float] = None,
        subgroup_size: int = 1,
        estimation_method: str = "rbar",
        confidence_level: float = 0.95,
        ppk_method: str = "within",
    ):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.usl = usl
        self.lsl = lsl
        self.target = target
        self.subgroup_size = int(subgroup_size) if subgroup_size is not None else 1
        self.estimation_method = estimation_method or "rbar"
        self.confidence_level = float(confidence_level)
        self.ppk_method = ppk_method or "within"
        self._results: Dict[str, Any] = {}
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._compute()

    # ------------------------------------------------------------------
    # Validation et utilitaires
    # ------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        n = len(self.data)

        if n == 0:
            self._errors.append("Aucune donnée numérique exploitable.")
            return
        if n < 2:
            self._errors.append("Au moins 2 valeurs numériques sont nécessaires.")
        if self.usl is None and self.lsl is None:
            self._errors.append("Au moins une limite de spécification USL ou LSL doit être renseignée.")
        if self.usl is not None and self.lsl is not None and self.usl <= self.lsl:
            self._errors.append("USL doit être strictement supérieure à LSL.")
        if self.target is not None and self.lsl is not None and self.usl is not None:
            if not (self.lsl <= self.target <= self.usl):
                self._warnings.append("La cible est hors intervalle [LSL ; USL].")
        if self.subgroup_size < 1:
            self._errors.append("La taille de sous-groupe doit être >= 1.")
        if n > 0 and self.subgroup_size > n:
            self._warnings.append("La taille de sous-groupe est supérieure au nombre de valeurs ; utilisation de sigma global en repli.")
        if self.estimation_method not in self.VALID_ESTIMATION_METHODS:
            self._warnings.append(f"Méthode sigma inconnue '{self.estimation_method}', remplacement par 'rbar'.")
            self.estimation_method = "rbar"
        if self.ppk_method not in self.VALID_PPK_METHODS:
            self._warnings.append(f"Méthode Ppk inconnue '{self.ppk_method}', remplacement par 'within'.")
            self.ppk_method = "within"
        if not (0 < self.confidence_level < 1):
            self._warnings.append("Niveau de confiance invalide ; remplacement par 0.95.")
            self.confidence_level = 0.95
        if n >= 2 and np.nanmax(self.data) == np.nanmin(self.data):
            self._errors.append("Les données sont constantes : sigma nul, capabilité non calculable.")

    @staticmethod
    def _is_positive_finite(value: Optional[float]) -> bool:
        return value is not None and np.isfinite(value) and value > 0

    @staticmethod
    def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
        if numerator is None or denominator is None:
            return None
        if not np.isfinite(numerator) or not np.isfinite(denominator):
            return None
        if denominator <= 0:
            return None
        return float(numerator / denominator)

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except Exception:
            return None

    def _spec_mode(self) -> str:
        if self.lsl is not None and self.usl is not None:
            return "bilateral"
        if self.usl is not None:
            return "upper_only"
        if self.lsl is not None:
            return "lower_only"
        return "none"

    # ------------------------------------------------------------------
    # Calcul principal
    # ------------------------------------------------------------------
    def _compute(self) -> None:
        self._validate_inputs()
        n = len(self.data)

        self._results = {
            "n": n,
            "errors": list(self._errors),
            "warnings": list(self._warnings),
            "is_valid": len(self._errors) == 0,
            "spec_mode": self._spec_mode(),
            "has_lsl": self.lsl is not None,
            "has_usl": self.usl is not None,
            "has_target": self.target is not None,
            "usl": self.usl,
            "lsl": self.lsl,
            "target": self.target,
            "confidence_level": self.confidence_level,
            "ci_method": "normal_approximation",
            "sigma_method_requested": self.estimation_method,
            "subgroup_size": self.subgroup_size,
            "display_labels": DISPLAY_LABELS.copy(),
        }

        if n > 0:
            self._results.update({
                "mean": self._safe_float(np.mean(self.data)),
                "median": self._safe_float(np.median(self.data)),
                "std_overall": self._safe_float(np.std(self.data, ddof=1)) if n > 1 else None,
                "min": self._safe_float(np.min(self.data)),
                "max": self._safe_float(np.max(self.data)),
                "range": self._safe_float(np.ptp(self.data)),
            })

        if self._errors:
            self._results["error"] = "; ".join(self._errors)
            self._compute_observed_nonconformities()
            return

        mean = self._results["mean"]
        sigma_within = self._estimate_sigma_within()
        sigma_overall = self._results["std_overall"]

        self._results["std_within"] = self._safe_float(sigma_within)
        self._results["std_overall"] = self._safe_float(sigma_overall)

        if not self._is_positive_finite(self._results["std_within"]):
            self._results["warnings"].append("Sigma intra nul ou invalide : indices Cp/Cpk non calculables.")
        if not self._is_positive_finite(self._results["std_overall"]):
            self._results["warnings"].append("Sigma global nul ou invalide : indices Pp/Ppk non calculables.")

        if self.usl is not None and self.lsl is not None:
            self._compute_two_sided(mean, self._results["std_within"], self._results["std_overall"])
        elif self.usl is not None:
            self._compute_upper_only(mean, self._results["std_within"], self._results["std_overall"])
        elif self.lsl is not None:
            self._compute_lower_only(mean, self._results["std_within"], self._results["std_overall"])

        if self.target is not None:
            self._compute_cpm(mean, self._results["std_within"])

        self._compute_confidence_intervals(self._results["std_within"], self._results["std_overall"], n)
        self._compute_ppm(mean, self._results["std_within"], self._results["std_overall"])
        self._compute_observed_nonconformities()

    # ------------------------------------------------------------------
    # Sigma intra
    # ------------------------------------------------------------------
    def _estimate_sigma_within(self) -> Optional[float]:
        n = len(self.data)
        self._results.update({
            "sigma_method_used": self.estimation_method,
            "subgroup_count": None,
            "ignored_observations": 0,
        })

        if n < 2:
            return None

        if self.subgroup_size <= 1:
            if self.estimation_method == "rbar":
                return self._sigma_from_moving_range()
            if self.estimation_method == "sbar":
                return self._sigma_from_moving_std()
            return self._sigma_pooled()

        subgroups = self._create_subgroups()
        self._results["subgroup_count"] = len(subgroups)
        self._results["ignored_observations"] = n - len(subgroups) * self.subgroup_size

        if len(subgroups) < 2:
            self._results["warnings"].append("Nombre de sous-groupes insuffisant ; utilisation de sigma global en repli.")
            self._results["sigma_method_used"] = "overall_fallback"
            return self._sigma_pooled()

        if self.estimation_method == "rbar":
            return self._sigma_from_rbar(subgroups)
        if self.estimation_method == "sbar":
            return self._sigma_from_sbar(subgroups)
        return self._sigma_pooled(subgroups)

    def _sigma_from_moving_range(self) -> Optional[float]:
        if len(self.data) < 2:
            return None
        mr = np.abs(np.diff(self.data))
        mrbar = np.mean(mr)
        return self._safe_divide(mrbar, 1.128)

    def _sigma_from_moving_std(self) -> Optional[float]:
        if len(self.data) < 3:
            return self._sigma_pooled()
        window_size = 2
        c4 = 0.8862
        stds = []
        for i in range(len(self.data) - window_size + 1):
            window = self.data[i:i + window_size]
            if len(window) > 1:
                stds.append(np.std(window, ddof=1))
        if not stds:
            return self._sigma_pooled()
        return self._safe_divide(np.mean(stds), c4)

    def _create_subgroups(self) -> List[np.ndarray]:
        n_subgroups = len(self.data) // self.subgroup_size
        return [self.data[i * self.subgroup_size:(i + 1) * self.subgroup_size] for i in range(n_subgroups)]

    def _sigma_from_rbar(self, subgroups: List[np.ndarray]) -> Optional[float]:
        d2_table = {
            2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
            7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173,
            12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472,
        }
        ranges = [np.max(sg) - np.min(sg) for sg in subgroups if len(sg) >= 2]
        if not ranges:
            return None
        d2 = d2_table.get(self.subgroup_size)
        if d2 is None:
            self._results["warnings"].append("d2 non disponible pour cette taille de sous-groupe ; utilisation de d2=3.078.")
            d2 = 3.078
        return self._safe_divide(np.mean(ranges), d2)

    def _sigma_from_sbar(self, subgroups: List[np.ndarray]) -> Optional[float]:
        c4_table = {
            2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515,
            7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727, 11: 0.9754,
            12: 0.9776, 13: 0.9794, 14: 0.9810, 15: 0.9823,
        }
        stds = [np.std(sg, ddof=1) for sg in subgroups if len(sg) >= 2]
        if not stds:
            return None
        c4 = c4_table.get(self.subgroup_size)
        if c4 is None:
            self._results["warnings"].append("c4 non disponible pour cette taille de sous-groupe ; utilisation de c4=0.9727.")
            c4 = 0.9727
        return self._safe_divide(np.mean(stds), c4)

    def _sigma_pooled(self, subgroups: Optional[List[np.ndarray]] = None) -> Optional[float]:
        if subgroups is None:
            return self._safe_float(np.std(self.data, ddof=1)) if len(self.data) > 1 else None
        variances = []
        weights = []
        for sg in subgroups:
            if len(sg) >= 2:
                variances.append(np.var(sg, ddof=1))
                weights.append(len(sg) - 1)
        denom = sum(weights)
        if denom <= 0:
            return None
        pooled_var = sum(w * v for w, v in zip(weights, variances)) / denom
        return self._safe_float(np.sqrt(pooled_var))

    # ------------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------------
    def _compute_two_sided(self, mean, sigma_within, sigma_overall) -> None:
        tol = self.usl - self.lsl
        cp = self._safe_divide(tol, 6 * sigma_within if sigma_within else None)
        cpu = self._safe_divide(self.usl - mean, 3 * sigma_within if sigma_within else None)
        cpl = self._safe_divide(mean - self.lsl, 3 * sigma_within if sigma_within else None)
        cpk = min([v for v in (cpu, cpl) if v is not None], default=None)

        pp = self._safe_divide(tol, 6 * sigma_overall if sigma_overall else None)
        ppu = self._safe_divide(self.usl - mean, 3 * sigma_overall if sigma_overall else None)
        ppl = self._safe_divide(mean - self.lsl, 3 * sigma_overall if sigma_overall else None)
        ppk = min([v for v in (ppu, ppl) if v is not None], default=None)

        self._results.update({
            "cp": cp, "cpu": cpu, "cpl": cpl, "cpk": cpk,
            "cpk_upper": cpu, "cpk_lower": cpl,
            "pp": pp, "ppu": ppu, "ppl": ppl, "ppk": ppk,
            "ppk_upper": ppu, "ppk_lower": ppl,
            "min_index": "USL" if (cpu is not None and cpl is not None and cpu < cpl) else "LSL",
        })

    def _compute_upper_only(self, mean, sigma_within, sigma_overall) -> None:
        cpu = self._safe_divide(self.usl - mean, 3 * sigma_within if sigma_within else None)
        ppu = self._safe_divide(self.usl - mean, 3 * sigma_overall if sigma_overall else None)
        self._results.update({
            "cpu": cpu, "cpk": cpu, "cpk_upper": cpu, "cpk_lower": None,
            "ppu": ppu, "ppk": ppu, "ppk_upper": ppu, "ppk_lower": None,
        })

    def _compute_lower_only(self, mean, sigma_within, sigma_overall) -> None:
        cpl = self._safe_divide(mean - self.lsl, 3 * sigma_within if sigma_within else None)
        ppl = self._safe_divide(mean - self.lsl, 3 * sigma_overall if sigma_overall else None)
        self._results.update({
            "cpl": cpl, "cpk": cpl, "cpk_lower": cpl, "cpk_upper": None,
            "ppl": ppl, "ppk": ppl, "ppk_lower": ppl, "ppk_upper": None,
        })

    def _compute_cpm(self, mean, sigma_within) -> None:
        if self.target is None or not self._is_positive_finite(sigma_within):
            self._results["cpm"] = None
            self._results["cpm_status"] = "sigma intra invalide ou cible absente"
            return
        sigma_prime = np.sqrt(sigma_within ** 2 + (mean - self.target) ** 2)
        if self.usl is not None and self.lsl is not None:
            self._results["cpm"] = self._safe_divide(self.usl - self.lsl, 6 * sigma_prime)
        elif self.usl is not None:
            self._results["cpm"] = self._safe_divide(self.usl - self.target, 3 * sigma_prime)
        elif self.lsl is not None:
            self._results["cpm"] = self._safe_divide(self.target - self.lsl, 3 * sigma_prime)

    # ------------------------------------------------------------------
    # IC, PPM, NC observées
    # ------------------------------------------------------------------
    def _compute_confidence_intervals(self, sigma_within, sigma_overall, n) -> None:
        if n < 2:
            self._results["ci_status"] = "effectif insuffisant"
            return
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        for key in ("cpk", "ppk"):
            val = self._results.get(key)
            if val is None or not np.isfinite(val):
                self._results[f"{key}_ci_lower"] = None
                self._results[f"{key}_ci_upper"] = None
                self._results[f"{key}_ci_status"] = "indice non calculable"
                continue
            se = np.sqrt(val ** 2 / (2 * (n - 1)) + 1 / (9 * n))
            self._results[f"{key}_ci_lower"] = max(0.0, float(val - z * se))
            self._results[f"{key}_ci_upper"] = float(val + z * se)
            self._results[f"{key}_ci_status"] = "OK"

        cp = self._results.get("cp")
        if cp is not None and np.isfinite(cp) and cp > 0:
            se_cp = cp * np.sqrt(1 / (2 * (n - 1)))
            denom_lo = 1 + z * se_cp / cp
            denom_hi = 1 - z * se_cp / cp
            self._results["cp_ci_lower"] = self._safe_divide(cp, denom_lo)
            self._results["cp_ci_upper"] = self._safe_divide(cp, denom_hi) if denom_hi > 0 else None
            self._results["cp_ci_status"] = "OK" if self._results["cp_ci_upper"] is not None else "borne supérieure non calculable"

    def _compute_ppm(self, mean, sigma_within, sigma_overall) -> None:
        def ppm_above(limit, sigma):
            if not self._is_positive_finite(sigma):
                return None
            return float((1 - stats.norm.cdf((limit - mean) / sigma)) * 1e6)

        def ppm_below(limit, sigma):
            if not self._is_positive_finite(sigma):
                return None
            return float(stats.norm.cdf((limit - mean) / sigma) * 1e6)

        if self.usl is not None:
            self._results["ppm_above_usl_within"] = ppm_above(self.usl, sigma_within)
            self._results["ppm_above_usl_overall"] = ppm_above(self.usl, sigma_overall)
        if self.lsl is not None:
            self._results["ppm_below_lsl_within"] = ppm_below(self.lsl, sigma_within)
            self._results["ppm_below_lsl_overall"] = ppm_below(self.lsl, sigma_overall)
        if self.usl is not None and self.lsl is not None:
            for suffix in ("within", "overall"):
                below = self._results.get(f"ppm_below_lsl_{suffix}")
                above = self._results.get(f"ppm_above_usl_{suffix}")
                self._results[f"ppm_total_{suffix}"] = None if below is None or above is None else float(below + above)

    def _compute_observed_nonconformities(self) -> None:
        if len(self.data) == 0:
            return
        below = int(np.sum(self.data < self.lsl)) if self.lsl is not None else 0
        above = int(np.sum(self.data > self.usl)) if self.usl is not None else 0
        total = below + above
        n = len(self.data)
        self._results.update({
            "observed_below_lsl_count": below,
            "observed_above_usl_count": above,
            "observed_total_nc_count": total,
            "observed_nc_count": total,  # alias compatibilité MiniQual
            "observed_total_nc_percent": float(total / n * 100),
            "observed_nc_percent": float(total / n * 100),  # alias compatibilité MiniQual
            "observed_ppm": float(total / n * 1e6),
        })

    # ------------------------------------------------------------------
    # Sorties
    # ------------------------------------------------------------------
    def get_results(self) -> Dict[str, Any]:
        return self._results

    def to_table(self) -> List[Dict[str, Any]]:
        rows = []
        for key, label in DISPLAY_LABELS.items():
            if key in self._results:
                rows.append({"Indicateur": label, "Clé": key, "Valeur": self._results.get(key)})
        return rows

    def to_report_sections(self) -> Dict[str, Dict[str, Any]]:
        r = self._results
        return {
            "Statistiques descriptives": {
                "N": r.get("n"),
                "Moyenne": r.get("mean"),
                "Médiane": r.get("median"),
                "Écart-type intra": r.get("std_within"),
                "Écart-type global": r.get("std_overall"),
                "Min": r.get("min"),
                "Max": r.get("max"),
            },
            "Limites": {
                "LSL": r.get("lsl"),
                "USL": r.get("usl"),
                "Cible": r.get("target"),
                "Mode": r.get("spec_mode"),
            },
            "Indices intra": {
                "Cp": r.get("cp"),
                "Cpl": r.get("cpl"),
                "Cpu": r.get("cpu"),
                "Cpk": r.get("cpk"),
                "Cpm": r.get("cpm"),
            },
            "Indices globaux": {
                "Pp": r.get("pp"),
                "Ppl": r.get("ppl"),
                "Ppu": r.get("ppu"),
                "Ppk": r.get("ppk"),
            },
            "Non-conformités observées": {
                "Sous LSL": r.get("observed_below_lsl_count"),
                "Au-dessus USL": r.get("observed_above_usl_count"),
                "Total": r.get("observed_total_nc_count"),
                "% total": r.get("observed_total_nc_percent"),
                "PPM observé": r.get("observed_ppm"),
            },
            "Sous-groupes et sigma": {
                "Méthode demandée": r.get("sigma_method_requested"),
                "Méthode utilisée": r.get("sigma_method_used"),
                "Taille sous-groupe": r.get("subgroup_size"),
                "Nombre de sous-groupes": r.get("subgroup_count"),
                "Observations ignorées": r.get("ignored_observations"),
            },
            "Statut": {
                "Valide": r.get("is_valid"),
                "Erreurs": "; ".join(r.get("errors", [])),
                "Avertissements": "; ".join(r.get("warnings", [])),
            },
        }

    def _fmt(self, value: Any, digits: int = 6) -> str:
        if value is None:
            return "non calculable"
        if isinstance(value, float):
            if not np.isfinite(value):
                return "non calculable"
            return f"{value:.{digits}f}"
        return str(value)

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "ANALYSE DE CAPABILITÉ", "=" * 60]

        if r.get("errors"):
            lines.append("\nErreurs :")
            for err in r.get("errors", []):
                lines.append(f" - {err}")
        if r.get("warnings"):
            lines.append("\nAvertissements :")
            for warn in r.get("warnings", []):
                lines.append(f" - {warn}")

        lines.append("\nStatistiques descriptives:")
        lines.append(f" N = {r.get('n')}")
        lines.append(f" Moyenne = {self._fmt(r.get('mean'))}")
        lines.append(f" Médiane = {self._fmt(r.get('median'))}")
        lines.append(f" Écart-type (intra) = {self._fmt(r.get('std_within'))}")
        lines.append(f" Écart-type (global) = {self._fmt(r.get('std_overall'))}")
        lines.append(f" Min = {self._fmt(r.get('min'))}")
        lines.append(f" Max = {self._fmt(r.get('max'))}")

        lines.append("\nLimites:")
        lines.append(f" Mode = {r.get('spec_mode')}")
        if r.get("lsl") is not None:
            lines.append(f" LSL = {self._fmt(r.get('lsl'))}")
        if r.get("usl") is not None:
            lines.append(f" USL = {self._fmt(r.get('usl'))}")
        if r.get("target") is not None:
            lines.append(f" Cible = {self._fmt(r.get('target'))}")

        lines.append("\nIndices de capabilité:")
        for key in ("cp", "cpl", "cpu", "cpk", "cpm", "pp", "ppl", "ppu", "ppk"):
            if key in r:
                lines.append(f" {DISPLAY_LABELS.get(key, key)} = {self._fmt(r.get(key), 4)}")

        if "cpk_ci_lower" in r:
            lines.append(f"\n IC Cpk ({self.confidence_level * 100:.0f}%): [{self._fmt(r.get('cpk_ci_lower'), 4)}, {self._fmt(r.get('cpk_ci_upper'), 4)}]")
        if "ppk_ci_lower" in r:
            lines.append(f" IC Ppk ({self.confidence_level * 100:.0f}%): [{self._fmt(r.get('ppk_ci_lower'), 4)}, {self._fmt(r.get('ppk_ci_upper'), 4)}]")

        lines.append("\nNon-conformités observées:")
        lines.append(f" Sous LSL = {r.get('observed_below_lsl_count', 0)}")
        lines.append(f" Au-dessus USL = {r.get('observed_above_usl_count', 0)}")
        lines.append(f" Total = {r.get('observed_total_nc_count', 0)}")
        lines.append(f" % total = {self._fmt(r.get('observed_total_nc_percent'), 4)}")
        lines.append(f" PPM observé = {self._fmt(r.get('observed_ppm'), 2)}")

        lines.append("\nSous-groupes et sigma:")
        lines.append(f" Méthode demandée = {r.get('sigma_method_requested')}")
        lines.append(f" Méthode utilisée = {r.get('sigma_method_used')}")
        lines.append(f" Taille sous-groupe = {r.get('subgroup_size')}")
        if r.get("subgroup_count") is not None:
            lines.append(f" Nombre de sous-groupes = {r.get('subgroup_count')}")
            lines.append(f" Observations ignorées = {r.get('ignored_observations')}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
