
"""
outliers_ameliore.py — Détection robuste des valeurs aberrantes pour StatPro.

Améliorations principales :
- validation des entrées ;
- conservation des indices d'origine / lignes affichables ;
- statuts explicites par test au lieu de retours silencieux ;
- Z-score configurable : classique, alpha ou Bonferroni ;
- correction du critère de Chauvenet ;
- consensus entre méthodes ;
- sorties structurées pour interface, rapport et DataFrame.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterable, Set
from collections import defaultdict

import numpy as np
from scipy import stats


class OutlierDetection:
    """
    Détection de valeurs aberrantes.

    Paramètres
    ----------
    data:
        Données numériques. Les NaN sont ignorés.
    alpha:
        Seuil de significativité pour les tests statistiques.
    original_indices:
        Indices originaux des valeurs dans le tableur ou la source.
        Si None, les indices 0..n-1 sont utilisés avant nettoyage des NaN.
    display_row_offset:
        Décalage pour afficher les lignes utilisateur. Par défaut 1 : index 0 -> ligne 1.
    zscore_mode:
        "classic" => seuil fixe |Z| > zscore_threshold.
        "alpha" => seuil normal bilatéral basé sur alpha.
        "bonferroni" => seuil normal bilatéral corrigé par n.
    zscore_threshold:
        Seuil du Z-score classique.
    modified_z_threshold:
        Seuil du Z-score modifié basé sur MAD.
    consensus_min_methods:
        Nombre minimal de méthodes pour classer une valeur comme forte suspicion.
    """

    def __init__(
        self,
        data: np.ndarray,
        alpha: float = 0.05,
        original_indices: Optional[Iterable[int]] = None,
        display_row_offset: int = 1,
        zscore_mode: str = "classic",
        zscore_threshold: float = 3.0,
        modified_z_threshold: float = 3.5,
        consensus_min_methods: int = 2,
        enabled_methods: Optional[Iterable[str]] = None,
    ):
        raw = np.asarray(data, dtype=float)
        if original_indices is None:
            raw_indices = np.arange(len(raw), dtype=int)
        else:
            raw_indices = np.asarray(list(original_indices), dtype=int)
            if len(raw_indices) != len(raw):
                raise ValueError("original_indices doit avoir la même longueur que data.")

        finite_mask = np.isfinite(raw)
        self.n_input = int(raw.size)
        self.n_removed_non_finite = int(raw.size - np.sum(finite_mask))
        self.non_finite_original_indices = raw_indices[~finite_mask].astype(int).tolist()
        self.non_finite_display_rows = (raw_indices[~finite_mask] + int(display_row_offset)).astype(int).tolist()
        self.data = raw[finite_mask]
        self.original_indices = raw_indices[finite_mask]
        self.display_rows = self.original_indices + int(display_row_offset)

        self.alpha = float(alpha)
        if not (0 < self.alpha < 1):
            self.alpha = 0.05
        self.display_row_offset = int(display_row_offset)
        self.zscore_mode = str(zscore_mode).lower()
        self.zscore_threshold = float(zscore_threshold)
        self.modified_z_threshold = float(modified_z_threshold)
        self.consensus_min_methods = int(consensus_min_methods)
        self.enabled_methods = set(enabled_methods) if enabled_methods is not None else None

        self._results: Dict[str, Any] = {}
        self._run_all_tests()

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    def _method_enabled(self, method_key: str) -> bool:
        return self.enabled_methods is None or method_key in self.enabled_methods

    def _base_result(self, method: str, description: str, status: str = "ok", reason: str = "") -> Dict[str, Any]:
        return {
            "method": method,
            "status": status,
            "reason": reason,
            "description": description,
            "outlier_values": [],
            "outlier_indices": [],
            "outlier_original_indices": [],
            "outlier_display_rows": [],
            "outlier_count": 0,
            "recommendation": "Ne pas supprimer automatiquement sans justification technique ou qualité.",
        }

    def _finalize_result(self, result: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
        mask = np.asarray(mask, dtype=bool)
        result["outlier_values"] = self.data[mask].astype(float).tolist()
        result["outlier_indices"] = np.where(mask)[0].astype(int).tolist()
        result["outlier_original_indices"] = self.original_indices[mask].astype(int).tolist()
        result["outlier_display_rows"] = self.display_rows[mask].astype(int).tolist()
        result["outlier_count"] = int(mask.sum())
        return result

    def _add_test_result(self, name: str, result: Dict[str, Any]) -> None:
        self._results["tests"][name] = result

    @staticmethod
    def _safe_std(x: np.ndarray) -> Optional[float]:
        if len(x) < 2:
            return None
        std = float(np.std(x, ddof=1))
        return std if np.isfinite(std) and std > 0 else None

    # ------------------------------------------------------------------
    # Exécution
    # ------------------------------------------------------------------
    def _run_all_tests(self) -> None:
        n = len(self.data)
        self._results = {
            "n": int(n),
            "n_input": getattr(self, "n_input", int(n)),
            "n_removed_non_finite": getattr(self, "n_removed_non_finite", 0),
            "non_finite_original_indices": getattr(self, "non_finite_original_indices", []),
            "non_finite_display_rows": getattr(self, "non_finite_display_rows", []),
            "alpha": self.alpha,
            "input_valid": n >= 2,
            "warnings": [],
            "tests": {},
            "consensus": [],
            "consensus_min_methods": self.consensus_min_methods,
        }

        if getattr(self, "n_removed_non_finite", 0):
            self._results["warnings"].append(
                f"{self.n_removed_non_finite} valeur(s) non finie(s) (NaN/Inf) exclue(s) de l'analyse."
            )

        if n == 0:
            self._results.update({
                "mean": None, "std": None, "median": None, "q1": None, "q3": None,
                "iqr": None, "min": None, "max": None,
                "error": "Aucune donnée numérique exploitable.",
            })
            return

        q1, q3 = np.percentile(self.data, [25, 75])
        std = self._safe_std(self.data)
        self._results.update({
            "mean": float(np.mean(self.data)),
            "std": std,
            "median": float(np.median(self.data)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
        })

        if n < 2:
            self._results["input_valid"] = False
            self._results["error"] = "Au moins 2 valeurs numériques sont nécessaires."
            return

        if np.max(self.data) == np.min(self.data):
            self._results["warnings"].append("Données constantes : plusieurs tests ne sont pas applicables.")

        if self._method_enabled("iqr"):
            self._iqr_method()
        if self._method_enabled("zscore"):
            self._z_score_method()
        if self._method_enabled("modified_zscore"):
            self._modified_z_score_method()
        if self._method_enabled("grubbs"):
            self._grubbs_test()
        if self._method_enabled("dixon"):
            self._dixon_test()
        if self._method_enabled("chauvenet"):
            self._chauvenet_criterion()
        if self._method_enabled("tukey"):
            self._tukey_fences()
        if self._method_enabled("esd"):
            self._generalized_esd_test()

        self._build_consensus()

    # ------------------------------------------------------------------
    # Méthodes de détection
    # ------------------------------------------------------------------
    def _iqr_method(self) -> None:
        result = self._base_result("IQR", "Méthode de l'écart interquartile, seuil 1,5×IQR")
        iqr = self._results.get("iqr")
        if iqr is None or iqr == 0:
            result.update({"status": "non_applicable", "reason": "IQR nul."})
            self._add_test_result("IQR (1.5×IQR)", result)
            return
        lower = self._results["q1"] - 1.5 * iqr
        upper = self._results["q3"] + 1.5 * iqr
        mask = (self.data < lower) | (self.data > upper)
        result.update({"lower_bound": float(lower), "upper_bound": float(upper)})
        self._add_test_result("IQR (1.5×IQR)", self._finalize_result(result, mask))

    def _z_score_method(self) -> None:
        result = self._base_result("Z-score", "Détection par écart à la moyenne en nombre d'écarts-types")
        std = self._safe_std(self.data)
        if std is None:
            result.update({"status": "non_applicable", "reason": "Écart-type nul ou effectif insuffisant."})
            self._add_test_result("Z-score", result)
            return

        n = len(self.data)
        if self.zscore_mode == "alpha":
            threshold = float(stats.norm.ppf(1 - self.alpha / 2))
            mode_label = "alpha"
        elif self.zscore_mode == "bonferroni":
            threshold = float(stats.norm.ppf(1 - self.alpha / (2 * n)))
            mode_label = "bonferroni"
        else:
            threshold = float(self.zscore_threshold)
            mode_label = "classic"

        z = np.abs((self.data - np.mean(self.data)) / std)
        mask = z > threshold
        result.update({"threshold": threshold, "mode": mode_label, "z_scores": z.astype(float).tolist()})
        self._add_test_result("Z-score", self._finalize_result(result, mask))

    def _modified_z_score_method(self) -> None:
        result = self._base_result("Modified Z-score", "Z-score modifié basé sur médiane et MAD, robuste")
        median = np.median(self.data)
        mad = np.median(np.abs(self.data - median))
        if mad == 0 or not np.isfinite(mad):
            result.update({"status": "non_applicable", "reason": "MAD nul."})
            self._add_test_result("Z-score modifié (MAD)", result)
            return
        modified_z = 0.6745 * (self.data - median) / mad
        mask = np.abs(modified_z) > self.modified_z_threshold
        result.update({
            "threshold": float(self.modified_z_threshold),
            "median": float(median),
            "mad": float(mad),
            "modified_z_scores": modified_z.astype(float).tolist(),
        })
        self._add_test_result("Z-score modifié (MAD)", self._finalize_result(result, mask))

    def _grubbs_test(self) -> None:
        result = self._base_result("Grubbs", "Test de Grubbs, une valeur aberrante extrême à la fois")
        n = len(self.data)
        if n < 3:
            result.update({"status": "non_applicable", "reason": "Grubbs nécessite au moins 3 valeurs."})
            self._add_test_result("Grubbs", result)
            return
        std = self._safe_std(self.data)
        if std is None:
            result.update({"status": "non_applicable", "reason": "Écart-type nul."})
            self._add_test_result("Grubbs", result)
            return
        mean = np.mean(self.data)
        deviations = np.abs(self.data - mean)
        extreme_idx = int(np.argmax(deviations))
        g_stat = float(deviations[extreme_idx] / std)
        t_crit = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        g_crit = float(((n - 1) * t_crit) / np.sqrt(n * (n - 2 + t_crit ** 2)))
        mask = np.zeros(n, dtype=bool)
        if g_stat > g_crit:
            mask[extreme_idx] = True
        result.update({"g_stat": g_stat, "critical_value": g_crit})
        self._add_test_result("Grubbs", self._finalize_result(result, mask))

    def _dixon_test(self) -> None:
        result = self._base_result("Dixon", "Test de Dixon pour petits échantillons, table alpha=0,05")
        n = len(self.data)
        if not (3 <= n <= 30):
            result.update({"status": "non_applicable", "reason": "Dixon applicable uniquement pour 3 ≤ n ≤ 30."})
            self._add_test_result("Dixon", result)
            return
        if abs(self.alpha - 0.05) > 1e-12:
            result["reason"] = "Table Dixon disponible uniquement à alpha=0,05 ; alpha utilisateur ignoré pour ce test."
            result["status"] = "ok_with_warning"
        sorted_idx = np.argsort(self.data)
        sorted_data = self.data[sorted_idx]
        range_val = sorted_data[-1] - sorted_data[0]
        if range_val == 0:
            result.update({"status": "non_applicable", "reason": "Étendue nulle."})
            self._add_test_result("Dixon", result)
            return
        q_low, q_high = self._compute_dixon_q(sorted_data, n)
        q_crit = self._get_dixon_critical_values_05().get(n, 0.5)
        mask = np.zeros(n, dtype=bool)
        if q_high > q_crit:
            mask[sorted_idx[-1]] = True
        if q_low > q_crit:
            mask[sorted_idx[0]] = True
        result.update({"q_low": float(q_low), "q_high": float(q_high), "critical_value": float(q_crit)})
        self._add_test_result("Dixon", self._finalize_result(result, mask))

    def _compute_dixon_q(self, sorted_data: np.ndarray, n: int):
        range_val = sorted_data[-1] - sorted_data[0]
        if n <= 10:
            q_low = (sorted_data[1] - sorted_data[0]) / range_val
            q_high = (sorted_data[-1] - sorted_data[-2]) / range_val
        elif n <= 13:
            q_low = (sorted_data[1] - sorted_data[0]) / (sorted_data[-2] - sorted_data[0])
            q_high = (sorted_data[-1] - sorted_data[-2]) / (sorted_data[-1] - sorted_data[1])
        else:
            q_low = (sorted_data[2] - sorted_data[0]) / (sorted_data[-3] - sorted_data[0])
            q_high = (sorted_data[-1] - sorted_data[-3]) / (sorted_data[-1] - sorted_data[2])
        return float(q_low), float(q_high)

    @staticmethod
    def _get_dixon_critical_values_05() -> Dict[int, float]:
        return {
            3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568,
            8: 0.526, 9: 0.493, 10: 0.466, 11: 0.517, 12: 0.490,
            13: 0.467, 14: 0.446, 15: 0.428, 16: 0.412, 17: 0.397,
            18: 0.384, 19: 0.372, 20: 0.361, 21: 0.386, 22: 0.376,
            23: 0.367, 24: 0.358, 25: 0.350, 26: 0.343, 27: 0.337,
            28: 0.331, 29: 0.325, 30: 0.320,
        }

    def _chauvenet_criterion(self) -> None:
        result = self._base_result("Chauvenet", "Critère de Chauvenet, probabilité bilatérale < 1/(2n)")
        n = len(self.data)
        std = self._safe_std(self.data)
        if std is None:
            result.update({"status": "non_applicable", "reason": "Écart-type nul ou effectif insuffisant."})
            self._add_test_result("Chauvenet", result)
            return
        # Correction : pas de +0.5. Seuil bilatéral : 2*(1-Phi(z)) < 1/(2n)
        threshold = float(stats.norm.ppf(1 - 1 / (4 * n)))
        z = np.abs((self.data - np.mean(self.data)) / std)
        mask = z > threshold
        result.update({"threshold": threshold, "z_scores": z.astype(float).tolist()})
        self._add_test_result("Chauvenet", self._finalize_result(result, mask))

    def _tukey_fences(self) -> None:
        result = self._base_result("Tukey fences", "Valeurs extrêmes avec seuil strict 3×IQR")
        iqr = self._results.get("iqr")
        if iqr is None or iqr == 0:
            result.update({"status": "non_applicable", "reason": "IQR nul."})
            self._add_test_result("Tukey (3×IQR, extrêmes)", result)
            return
        lower = self._results["q1"] - 3 * iqr
        upper = self._results["q3"] + 3 * iqr
        mask = (self.data < lower) | (self.data > upper)
        result.update({"lower_fence": float(lower), "upper_fence": float(upper)})
        self._add_test_result("Tukey (3×IQR, extrêmes)", self._finalize_result(result, mask))

    def _generalized_esd_test(self, max_outliers: Optional[int] = None) -> None:
        result = self._base_result("Generalized ESD", "Rosner / Generalized ESD pour plusieurs valeurs aberrantes")
        n = len(self.data)
        if n < 5:
            result.update({"status": "non_applicable", "reason": "Generalized ESD nécessite au moins 5 valeurs."})
            self._add_test_result("Generalized ESD", result)
            return
        if max_outliers is None:
            max_outliers = max(1, min(int(np.floor(n * 0.1)), n - 3))

        working = self.data.copy()
        working_indices = np.arange(n)
        removed_indices = []
        r_stats = []
        lambdas = []

        for i in range(1, max_outliers + 1):
            std = self._safe_std(working)
            if std is None or len(working) < 3:
                break
            mean = np.mean(working)
            deviations = np.abs(working - mean)
            local_idx = int(np.argmax(deviations))
            r_i = float(deviations[local_idx] / std)
            p = 1 - self.alpha / (2 * (len(working)))
            t = stats.t.ppf(p, len(working) - 2)
            lam = float((len(working) - 1) * t / np.sqrt((len(working) - 2 + t ** 2) * len(working)))
            r_stats.append(r_i)
            lambdas.append(lam)
            removed_indices.append(int(working_indices[local_idx]))
            working = np.delete(working, local_idx)
            working_indices = np.delete(working_indices, local_idx)

        k = 0
        for i, (r_i, lam) in enumerate(zip(r_stats, lambdas), start=1):
            if r_i > lam:
                k = i
        mask = np.zeros(n, dtype=bool)
        for idx in removed_indices[:k]:
            mask[idx] = True
        result.update({"max_outliers": int(max_outliers), "r_statistics": r_stats, "critical_values": lambdas})
        self._add_test_result("Generalized ESD", self._finalize_result(result, mask))

    # ------------------------------------------------------------------
    # Consensus et sorties
    # ------------------------------------------------------------------
    def _build_consensus(self) -> None:
        detected = defaultdict(lambda: {"methods": [], "value": None, "original_index": None, "display_row": None})
        for test_name, result in self._results.get("tests", {}).items():
            if result.get("status") not in ("ok", "ok_with_warning"):
                continue
            for local_idx, value, original_idx, display_row in zip(
                result.get("outlier_indices", []),
                result.get("outlier_values", []),
                result.get("outlier_original_indices", []),
                result.get("outlier_display_rows", []),
            ):
                item = detected[int(local_idx)]
                item["methods"].append(test_name)
                item["value"] = float(value)
                item["original_index"] = int(original_idx)
                item["display_row"] = int(display_row)

        all_detected = []
        confirmed_outliers = []
        suspect_points = []
        for local_idx, item in detected.items():
            score = len(item["methods"])
            confirmed = score >= self.consensus_min_methods
            if confirmed:
                severity = "forte" if score >= max(self.consensus_min_methods + 2, 4) else "confirmée"
            elif score == 1:
                severity = "à surveiller"
            else:
                severity = "modérée"
            record = {
                "index": int(local_idx),
                "original_index": item["original_index"],
                "display_row": item["display_row"],
                "value": item["value"],
                "detected_by": item["methods"],
                "score": int(score),
                "confirmed": bool(confirmed),
                "severity": severity,
                "recommendation": "Investiguer avant toute exclusion. Ne pas supprimer automatiquement.",
            }
            all_detected.append(record)
            if confirmed:
                confirmed_outliers.append(record)
            else:
                suspect_points.append(record)

        all_detected.sort(key=lambda d: (-d["score"], d["display_row"]))
        confirmed_outliers.sort(key=lambda d: (-d["score"], d["display_row"]))
        suspect_points.sort(key=lambda d: (-d["score"], d["display_row"]))
        self._results["all_detected_points"] = all_detected
        self._results["confirmed_outliers"] = confirmed_outliers
        self._results["suspect_points"] = suspect_points
        # Compatibilité API historique : consensus conserve tous les points détectés, avec le champ confirmed.
        self._results["consensus"] = all_detected

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_outlier_indices(self, min_methods: int = 1, original: bool = False) -> Set[int]:
        key = "original_index" if original else "index"
        return {int(row[key]) for row in self._results.get("consensus", []) if row.get("score", 0) >= min_methods}

    def to_table(self, consensus_only: bool = True) -> List[Dict[str, Any]]:
        if consensus_only:
            return [
                {
                    "Ligne": r["display_row"],
                    "Index original": r["original_index"],
                    "Valeur": r["value"],
                    "Méthodes": ", ".join(r["detected_by"]),
                    "Score": r["score"],
                    "Sévérité": r["severity"],
                    "Recommandation": r["recommendation"],
                }
                for r in self._results.get("consensus", [])
            ]
        rows = []
        for test_name, result in self._results.get("tests", {}).items():
            if result.get("outlier_count", 0) == 0:
                rows.append({
                    "Méthode": test_name,
                    "Statut": result.get("status"),
                    "Raison": result.get("reason", ""),
                    "Ligne": "",
                    "Valeur": "",
                    "Nombre": 0,
                })
            else:
                for line, value in zip(result.get("outlier_display_rows", []), result.get("outlier_values", [])):
                    rows.append({
                        "Méthode": test_name,
                        "Statut": result.get("status"),
                        "Raison": result.get("reason", ""),
                        "Ligne": line,
                        "Valeur": value,
                        "Nombre": result.get("outlier_count", 0),
                    })
        return rows

    def to_dataframe(self, consensus_only: bool = True):
        import pandas as pd
        return pd.DataFrame(self.to_table(consensus_only=consensus_only))

    def to_report_sections(self) -> Dict[str, Any]:
        return {
            "Statistiques descriptives": {
                "N": self._results.get("n"),
                "Moyenne": self._results.get("mean"),
                "Médiane": self._results.get("median"),
                "Écart-type": self._results.get("std"),
                "Q1": self._results.get("q1"),
                "Q3": self._results.get("q3"),
                "IQR": self._results.get("iqr"),
                "Min": self._results.get("min"),
                "Max": self._results.get("max"),
            },
            "Consensus valeurs aberrantes": self.to_table(consensus_only=True),
            "Résultats détaillés par méthode": self.to_table(consensus_only=False),
            "Recommandation qualité": "Une valeur détectée comme aberrante ne doit pas être supprimée automatiquement sans justification technique ou qualité.",
        }

    def get_summary(self) -> str:
        lines = ["=" * 60, "DÉTECTION DE VALEURS ABERRANTES", "=" * 60]
        if self._results.get("error"):
            lines.append(f"\nErreur : {self._results['error']}")
            return "\n".join(lines)

        lines.append("\nStatistiques descriptives:")
        lines.append(f" N = {self._results.get('n')}")
        lines.append(f" Moyenne = {self._results.get('mean'):.6f}")
        lines.append(f" Médiane = {self._results.get('median'):.6f}")
        std = self._results.get("std")
        lines.append(f" Écart-type = {'non calculable' if std is None else f'{std:.6f}'}")
        lines.append(f" Q1 = {self._results.get('q1'):.6f}")
        lines.append(f" Q3 = {self._results.get('q3'):.6f}")
        lines.append(f" IQR = {self._results.get('iqr'):.6f}")
        lines.append(f" Min = {self._results.get('min'):.6f}")
        lines.append(f" Max = {self._results.get('max'):.6f}")
        lines.append(f"\nSeuil de significativité alpha = {self.alpha}")

        if self._results.get("warnings"):
            lines.append("\nAvertissements:")
            for w in self._results["warnings"]:
                lines.append(f" - {w}")

        lines.append("\n" + "-" * 60)
        for name, result in self._results.get("tests", {}).items():
            lines.append(f"\n{name}:")
            lines.append(f" {result.get('description', '')}")
            status = result.get("status", "ok")
            if status != "ok":
                lines.append(f" Statut = {status}")
                if result.get("reason"):
                    lines.append(f" Raison = {result.get('reason')}")
            lines.append(f" Valeurs détectées = {result.get('outlier_count', 0)}")
            for value, line in zip(result.get("outlier_values", []), result.get("outlier_display_rows", [])):
                lines.append(f" Ligne {line} : valeur = {value:.6f}")
            if result.get("outlier_count", 0) == 0:
                lines.append(" Aucune valeur aberrante détectée")

        lines.append("\n" + "-" * 60)
        lines.append("Consensus:")
        if not self._results.get("consensus"):
            lines.append(" Aucune valeur détectée par consensus.")
        else:
            for r in self._results["consensus"]:
                methods = ", ".join(r["detected_by"])
                lines.append(f" Ligne {r['display_row']} : {r['value']:.6f} | score={r['score']} | {r['severity']} | {methods}")
        lines.append("\nRecommandation : investiguer avant toute exclusion ; ne pas supprimer automatiquement.")
        lines.append("=" * 60)
        return "\n".join(lines)
