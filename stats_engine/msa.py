
"""
msa_ameliore.py — Module MSA / Gage R&R amélioré pour StatPro.

Améliorations principales :
- validation robuste des entrées ;
- gestion explicite des erreurs et avertissements ;
- contrôle des plans équilibrés opérateur × pièce × répétition ;
- table ANOVA avec F et p-values ;
- composantes de variance corrigées ;
- %Contribution, %Study Variation et %Tolerance ;
- conclusion basée sur %Tolérance si disponible, sinon %Study Variation ;
- ndc sécurisé ;
- sorties structurées : get_results(), get_summary(), to_table(), to_report_sections().

API compatible :
    GageRRStudy(data, n_operators, n_parts, n_trials, spec_lsl=None, spec_usl=None, tolerance=None)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterable
import numpy as np
from scipy import stats


class GageRRStudy:
    """Étude de capabilité du système de mesure — Gage R&R ANOVA équilibrée."""

    VALID_INTERACTION_POLICIES = {"include", "exclude", "exclude_if_not_significant"}

    def __init__(
        self,
        data: np.ndarray,
        n_operators: int,
        n_parts: int,
        n_trials: int,
        spec_lsl: Optional[float] = None,
        spec_usl: Optional[float] = None,
        tolerance: Optional[float] = None,
        alpha: float = 0.05,
        study_variation_multiplier: float = 6.0,
        interaction_policy: str = "include",
    ):
        self.raw_data = np.asarray(data, dtype=float)
        self.n_input = int(self.raw_data.size)
        self.non_finite_mask = ~np.isfinite(self.raw_data)
        self.n_removed_non_finite = int(np.sum(self.non_finite_mask))
        self.non_finite_positions = np.where(self.non_finite_mask.ravel())[0].astype(int).tolist()
        self.n_operators = int(n_operators)
        self.n_parts = int(n_parts)
        self.n_trials = int(n_trials)
        self.spec_lsl = spec_lsl
        self.spec_usl = spec_usl
        self.tolerance = tolerance
        self.alpha = float(alpha) if 0 < float(alpha) < 1 else 0.05
        self.study_variation_multiplier = float(study_variation_multiplier)
        self.interaction_policy = interaction_policy if interaction_policy in self.VALID_INTERACTION_POLICIES else "include"

        self.data: Optional[np.ndarray] = None
        self._results: Dict[str, Any] = {}
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._compute()

    # ------------------------------------------------------------------
    # Validation / utilitaires
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_div(num, den):
        if den is None or num is None:
            return None
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return None
        return float(num / den)

    @staticmethod
    def _safe_sqrt(value):
        if value is None or not np.isfinite(value):
            return None
        return float(np.sqrt(max(0.0, value)))

    @staticmethod
    def _fmt(value, digits=6):
        if value is None:
            return "non calculable"
        try:
            if not np.isfinite(value):
                return "non calculable"
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    def _validate_inputs(self) -> None:
        if self.n_operators < 2:
            self._errors.append("Au moins 2 opérateurs sont nécessaires pour une étude Gage R&R.")
        if self.n_parts < 2:
            self._errors.append("Au moins 2 pièces sont nécessaires pour une étude Gage R&R.")
        if self.n_trials < 2:
            self._errors.append("Au moins 2 répétitions sont nécessaires pour estimer la répétabilité.")
        if self.study_variation_multiplier <= 0:
            self._warnings.append("Multiplicateur de variation d'étude invalide ; remplacement par 6.")
            self.study_variation_multiplier = 6.0
        if self.spec_lsl is not None and self.spec_usl is not None and self.spec_usl <= self.spec_lsl:
            self._errors.append("USL doit être strictement supérieure à LSL.")
        if self.tolerance is not None and self.tolerance <= 0:
            self._errors.append("La tolérance doit être strictement positive.")
        if self.tolerance is None and self.spec_lsl is not None and self.spec_usl is not None:
            self.tolerance = self.spec_usl - self.spec_lsl
        if self.raw_data.size == 0:
            self._errors.append("Aucune donnée de mesure fournie.")
            return
        if getattr(self, "n_removed_non_finite", 0) > 0:
            self._errors.append(
                f"Valeurs non finies détectées (NaN/Inf) : {self.n_removed_non_finite}. "
                "Le plan MSA équilibré exige des mesures numériques finies."
            )
            return

        if self.raw_data.ndim == 1:
            expected = self.n_operators * self.n_parts * self.n_trials
            if self.raw_data.size != expected:
                self._errors.append(
                    f"Taille des données incohérente : attendu {expected} valeurs "
                    f"({self.n_operators} opérateurs × {self.n_parts} pièces × {self.n_trials} répétitions), "
                    f"reçu {self.raw_data.size}."
                )
                return
            self.data = self.raw_data.reshape(self.n_operators, self.n_parts, self.n_trials)
        elif self.raw_data.ndim == 3:
            expected_shape = (self.n_operators, self.n_parts, self.n_trials)
            if self.raw_data.shape != expected_shape:
                self._errors.append(f"Forme des données incohérente : attendu {expected_shape}, reçu {self.raw_data.shape}.")
                return
            self.data = self.raw_data.copy()
        else:
            self._errors.append("Les données doivent être un tableau 1D ou 3D.")
            return

        if np.isnan(self.data).any():
            self._errors.append("Données manquantes détectées : le plan MSA équilibré exige toutes les mesures.")
        if np.nanmax(self.data) == np.nanmin(self.data):
            self._warnings.append("Toutes les mesures sont identiques : les composantes de variance seront nulles ou non discriminantes.")

    def _init_error_results(self) -> None:
        self._results = {
            "is_valid": False,
            "n_input": getattr(self, "n_input", int(self.raw_data.size) if hasattr(self, "raw_data") else 0),
            "n_removed_non_finite": getattr(self, "n_removed_non_finite", 0),
            "non_finite_positions": getattr(self, "non_finite_positions", []),
            "errors": list(self._errors),
            "warnings": list(self._warnings),
            "n_operators": self.n_operators,
            "n_parts": self.n_parts,
            "n_trials": self.n_trials,
            "spec_lsl": self.spec_lsl,
            "spec_usl": self.spec_usl,
            "tolerance": self.tolerance,
            "alpha": self.alpha,
            "interaction_policy": self.interaction_policy,
            "error": "; ".join(self._errors),
        }

    # ------------------------------------------------------------------
    # Calcul principal
    # ------------------------------------------------------------------
    def _compute(self) -> None:
        self._validate_inputs()
        if self._errors:
            self._init_error_results()
            return

        data = self.data
        n_op, n_part, n_rep = self.n_operators, self.n_parts, self.n_trials
        n_total = n_op * n_part * n_rep

        op_means = np.mean(data, axis=(1, 2))
        part_means = np.mean(data, axis=(0, 2))
        cell_means = np.mean(data, axis=2)
        grand_mean = float(np.mean(data))

        # Sommes de carrés ANOVA équilibrée à deux facteurs avec répétitions
        ss_total = float(np.sum((data - grand_mean) ** 2))
        ss_operators = float(n_part * n_rep * np.sum((op_means - grand_mean) ** 2))
        ss_parts = float(n_op * n_rep * np.sum((part_means - grand_mean) ** 2))
        ss_interaction = float(n_rep * np.sum((cell_means - op_means[:, None] - part_means[None, :] + grand_mean) ** 2))
        ss_repeatability = float(np.sum((data - cell_means[:, :, None]) ** 2))

        df_operators = n_op - 1
        df_parts = n_part - 1
        df_interaction = (n_op - 1) * (n_part - 1)
        df_repeatability = n_op * n_part * (n_rep - 1)
        df_total = n_total - 1

        ms_operators = self._safe_div(ss_operators, df_operators) or 0.0
        ms_parts = self._safe_div(ss_parts, df_parts) or 0.0
        ms_interaction = self._safe_div(ss_interaction, df_interaction) or 0.0
        ms_repeatability = self._safe_div(ss_repeatability, df_repeatability) or 0.0

        f_operators = self._safe_div(ms_operators, ms_interaction) if df_interaction > 0 and ms_interaction > 0 else None
        p_operators = float(stats.f.sf(f_operators, df_operators, df_interaction)) if f_operators is not None else None
        f_parts = self._safe_div(ms_parts, ms_interaction) if df_interaction > 0 and ms_interaction > 0 else None
        p_parts = float(stats.f.sf(f_parts, df_parts, df_interaction)) if f_parts is not None else None
        f_interaction = self._safe_div(ms_interaction, ms_repeatability) if df_repeatability > 0 and ms_repeatability > 0 else None
        p_interaction = float(stats.f.sf(f_interaction, df_interaction, df_repeatability)) if f_interaction is not None else None

        var_repeatability = max(0.0, ms_repeatability)
        var_reproducibility = max(0.0, (ms_operators - ms_interaction) / (n_part * n_rep))
        var_interaction = max(0.0, (ms_interaction - ms_repeatability) / n_rep)
        var_part = max(0.0, (ms_parts - ms_interaction) / (n_op * n_rep))

        interaction_included = False
        if self.interaction_policy == "include":
            interaction_included = True
        elif self.interaction_policy == "exclude":
            interaction_included = False
        elif self.interaction_policy == "exclude_if_not_significant":
            interaction_included = p_interaction is not None and p_interaction < self.alpha

        var_gage_rr = var_repeatability + var_reproducibility + (var_interaction if interaction_included else 0.0)
        var_total = var_gage_rr + var_part

        sigma_repeatability = self._safe_sqrt(var_repeatability)
        sigma_reproducibility = self._safe_sqrt(var_reproducibility)
        sigma_interaction = self._safe_sqrt(var_interaction)
        sigma_gage_rr = self._safe_sqrt(var_gage_rr)
        sigma_part = self._safe_sqrt(var_part)
        sigma_total = self._safe_sqrt(var_total)

        k = self.study_variation_multiplier
        sv_repeatability = k * sigma_repeatability if sigma_repeatability is not None else None
        sv_reproducibility = k * sigma_reproducibility if sigma_reproducibility is not None else None
        sv_interaction = k * sigma_interaction if sigma_interaction is not None else None
        sv_gage_rr = k * sigma_gage_rr if sigma_gage_rr is not None else None
        sv_part = k * sigma_part if sigma_part is not None else None
        sv_total = k * sigma_total if sigma_total is not None else None

        pct_contribution_repeatability = self._safe_div(var_repeatability * 100, var_total)
        pct_contribution_reproducibility = self._safe_div(var_reproducibility * 100, var_total)
        pct_contribution_interaction = self._safe_div(var_interaction * 100, var_total)
        pct_contribution_gage_rr = self._safe_div(var_gage_rr * 100, var_total)
        pct_contribution_part = self._safe_div(var_part * 100, var_total)

        pct_study_variation_repeatability = self._safe_div(sv_repeatability * 100 if sv_repeatability is not None else None, sv_total)
        pct_study_variation_reproducibility = self._safe_div(sv_reproducibility * 100 if sv_reproducibility is not None else None, sv_total)
        pct_study_variation_interaction = self._safe_div(sv_interaction * 100 if sv_interaction is not None else None, sv_total)
        pct_study_variation_gage_rr = self._safe_div(sv_gage_rr * 100 if sv_gage_rr is not None else None, sv_total)
        pct_study_variation_part = self._safe_div(sv_part * 100 if sv_part is not None else None, sv_total)

        pct_tolerance = self._safe_div(sv_gage_rr * 100 if sv_gage_rr is not None else None, self.tolerance)

        ndc = None
        if sigma_gage_rr is not None and sigma_gage_rr > 0 and sigma_part is not None:
            ndc = max(1, int(1.41 * sigma_part / sigma_gage_rr))
        elif sigma_gage_rr == 0 and sigma_part is not None and sigma_part > 0:
            ndc = 999
        else:
            ndc = 1

        acceptability_metric = pct_tolerance if pct_tolerance is not None else pct_study_variation_gage_rr
        acceptability_metric_name = "%Tolérance GRR" if pct_tolerance is not None else "%Study Variation GRR"
        acceptability = self._classify_acceptability(acceptability_metric, ndc)

        anova_table = [
            {"Source": "Pièces", "SS": ss_parts, "df": df_parts, "MS": ms_parts, "F": f_parts, "p-value": p_parts},
            {"Source": "Opérateurs", "SS": ss_operators, "df": df_operators, "MS": ms_operators, "F": f_operators, "p-value": p_operators},
            {"Source": "Interaction opérateur×pièce", "SS": ss_interaction, "df": df_interaction, "MS": ms_interaction, "F": f_interaction, "p-value": p_interaction},
            {"Source": "Répétabilité", "SS": ss_repeatability, "df": df_repeatability, "MS": ms_repeatability, "F": None, "p-value": None},
            {"Source": "Total", "SS": ss_total, "df": df_total, "MS": None, "F": None, "p-value": None},
        ]

        self._results = {
            "is_valid": True,
            "n_input": getattr(self, "n_input", int(self.raw_data.size)),
            "n_removed_non_finite": getattr(self, "n_removed_non_finite", 0),
            "non_finite_positions": getattr(self, "non_finite_positions", []),
            "errors": [],
            "warnings": list(self._warnings),
            "alpha": self.alpha,
            "interaction_policy": self.interaction_policy,
            "interaction_included": interaction_included,
            "study_variation_multiplier": k,
            "n_operators": n_op,
            "n_parts": n_part,
            "n_trials": n_rep,
            "n_total": n_total,
            "spec_lsl": self.spec_lsl,
            "spec_usl": self.spec_usl,
            "tolerance": self.tolerance,
            "grand_mean": grand_mean,
            "op_means": op_means.astype(float).tolist(),
            "part_means": part_means.astype(float).tolist(),
            "cell_means": cell_means.astype(float).tolist(),
            "anova_table": anova_table,
            "ss_operators": ss_operators,
            "ss_parts": ss_parts,
            "ss_interaction": ss_interaction,
            "ss_repeatability": ss_repeatability,
            "ss_total": ss_total,
            "df_operators": df_operators,
            "df_parts": df_parts,
            "df_interaction": df_interaction,
            "df_repeatability": df_repeatability,
            "df_total": df_total,
            "ms_operators": ms_operators,
            "ms_parts": ms_parts,
            "ms_interaction": ms_interaction,
            "ms_repeatability": ms_repeatability,
            "f_operators": f_operators,
            "p_operators": p_operators,
            "f_parts": f_parts,
            "p_parts": p_parts,
            "f_interaction": f_interaction,
            "p_interaction": p_interaction,
            "var_repeatability": var_repeatability,
            "var_reproducibility": var_reproducibility,
            "var_interaction": var_interaction,
            "var_gage_rr": var_gage_rr,
            "var_part": var_part,
            "var_total": var_total,
            "sigma_repeatability": sigma_repeatability,
            "sigma_reproducibility": sigma_reproducibility,
            "sigma_interaction": sigma_interaction,
            "sigma_gage_rr": sigma_gage_rr,
            "sigma_part": sigma_part,
            "sigma_total": sigma_total,
            "sv_repeatability": sv_repeatability,
            "sv_reproducibility": sv_reproducibility,
            "sv_interaction": sv_interaction,
            "sv_gage_rr": sv_gage_rr,
            "sv_part": sv_part,
            "sv_total": sv_total,
            "pct_contribution_repeatability": pct_contribution_repeatability,
            "pct_contribution_reproducibility": pct_contribution_reproducibility,
            "pct_contribution_interaction": pct_contribution_interaction,
            "pct_contribution_gage_rr": pct_contribution_gage_rr,
            "pct_contribution_part": pct_contribution_part,
            "pct_study_variation_repeatability": pct_study_variation_repeatability,
            "pct_study_variation_reproducibility": pct_study_variation_reproducibility,
            "pct_study_variation_interaction": pct_study_variation_interaction,
            "pct_study_variation_gage_rr": pct_study_variation_gage_rr,
            "pct_study_variation_part": pct_study_variation_part,
            "pct_tolerance": pct_tolerance,
            "ndc": ndc,
            "acceptability_metric": acceptability_metric,
            "acceptability_metric_name": acceptability_metric_name,
            "acceptability": acceptability,
        }

    @staticmethod
    def _classify_acceptability(metric: Optional[float], ndc: Optional[int]) -> str:
        if metric is None:
            return "N/A"
        if metric < 10 and (ndc is None or ndc >= 5):
            return "Acceptable"
        if metric < 30 and (ndc is None or ndc >= 2):
            return "Conditionnellement acceptable"
        return "Inacceptable"

    # ------------------------------------------------------------------
    # Sorties
    # ------------------------------------------------------------------
    def get_results(self) -> Dict[str, Any]:
        return self._results

    def to_table(self, section: str = "components") -> List[Dict[str, Any]]:
        r = self._results
        if not r.get("is_valid", False):
            return [{"Erreur": r.get("error", "Analyse non valide")}]

        if section == "anova":
            return r.get("anova_table", [])

        if section == "components":
            return [
                {"Composante": "Répétabilité", "Variance": r.get("var_repeatability"), "Sigma": r.get("sigma_repeatability"), "Study Variation": r.get("sv_repeatability"), "%Contribution": r.get("pct_contribution_repeatability"), "%StudyVar": r.get("pct_study_variation_repeatability")},
                {"Composante": "Reproductibilité", "Variance": r.get("var_reproducibility"), "Sigma": r.get("sigma_reproducibility"), "Study Variation": r.get("sv_reproducibility"), "%Contribution": r.get("pct_contribution_reproducibility"), "%StudyVar": r.get("pct_study_variation_reproducibility")},
                {"Composante": "Interaction", "Variance": r.get("var_interaction"), "Sigma": r.get("sigma_interaction"), "Study Variation": r.get("sv_interaction"), "%Contribution": r.get("pct_contribution_interaction"), "%StudyVar": r.get("pct_study_variation_interaction")},
                {"Composante": "Gage R&R", "Variance": r.get("var_gage_rr"), "Sigma": r.get("sigma_gage_rr"), "Study Variation": r.get("sv_gage_rr"), "%Contribution": r.get("pct_contribution_gage_rr"), "%StudyVar": r.get("pct_study_variation_gage_rr"), "%Tolérance": r.get("pct_tolerance")},
                {"Composante": "Pièce à pièce", "Variance": r.get("var_part"), "Sigma": r.get("sigma_part"), "Study Variation": r.get("sv_part"), "%Contribution": r.get("pct_contribution_part"), "%StudyVar": r.get("pct_study_variation_part")},
                {"Composante": "Total", "Variance": r.get("var_total"), "Sigma": r.get("sigma_total"), "Study Variation": r.get("sv_total"), "%Contribution": 100.0, "%StudyVar": 100.0},
            ]

        if section == "summary":
            return [{
                "Critère": r.get("acceptability_metric_name"),
                "Valeur": r.get("acceptability_metric"),
                "ndc": r.get("ndc"),
                "Conclusion": r.get("acceptability"),
                "Interaction incluse": r.get("interaction_included"),
            }]

        return []

    def to_dataframe(self, section: str = "components"):
        import pandas as pd
        return pd.DataFrame(self.to_table(section=section))

    def to_report_sections(self) -> Dict[str, Any]:
        r = self._results
        if not r.get("is_valid", False):
            return {"Erreurs": r.get("errors", []), "Avertissements": r.get("warnings", [])}
        return {
            "Plan d'étude": {
                "Opérateurs": r.get("n_operators"),
                "Pièces": r.get("n_parts"),
                "Répétitions": r.get("n_trials"),
                "Total mesures": r.get("n_total"),
                "Tolérance": r.get("tolerance"),
                "LSL": r.get("spec_lsl"),
                "USL": r.get("spec_usl"),
            },
            "ANOVA": self.to_table("anova"),
            "Composantes de variance": self.to_table("components"),
            "Conclusion": self.to_table("summary"),
            "Avertissements": r.get("warnings", []),
        }

    def get_summary(self) -> str:
        """Résumé court, factuel et lisible de l'étude Gage R&R."""
        r = self._results
        if not r.get("is_valid", False):
            lines = ["=" * 60, "ÉTUDE GAGE R&R - Synthèse", "=" * 60, "", "Statut : étude non valide"]
            for err in r.get("errors", []):
                lines.append(f"- {err}")
            for warn in r.get("warnings", []):
                lines.append(f"- {warn}")
            return "\n".join(lines)

        def fmt(value, digits=2, suffix=""):
            if value is None:
                return "n/a"
            try:
                if not np.isfinite(value):
                    return "n/a"
                return f"{float(value):.{digits}f}{suffix}"
            except Exception:
                return "n/a"

        def pval(value):
            if value is None:
                return "n/a"
            try:
                if not np.isfinite(value):
                    return "n/a"
                return "< 0.000001" if value < 0.000001 else f"= {value:.6f}"
            except Exception:
                return "n/a"

        def status_pct(value):
            if value is None:
                return "non calculable"
            if value < 10:
                return "favorable"
            if value <= 30:
                return "intermédiaire"
            return "défavorable"

        def status_ndc(value):
            if value is None:
                return "non calculable"
            return "favorable" if value >= 5 else "défavorable"

        sigma_grr = r.get("sigma_gage_rr")
        six_sigma_grr = 6.0 * sigma_grr if sigma_grr is not None and np.isfinite(sigma_grr) else None
        pct_contrib_grr = r.get("pct_contribution_gage_rr")
        pct_contrib_part = r.get("pct_contribution_part")
        pct_sv_grr = r.get("pct_study_variation_gage_rr")
        pct_tol = r.get("pct_tolerance")
        ndc = r.get("ndc")

        lines = ["=" * 60, "ÉTUDE GAGE R&R - Synthèse", "=" * 60]
        if r.get("warnings"):
            lines.append("\nMessages :")
            lines.extend(f"- {w}" for w in r.get("warnings", []))

        lines += [
            "\nPlan d’étude :",
            f"{r.get('n_operators')} opérateurs × {r.get('n_parts')} pièces × {r.get('n_trials')} répétitions",
            f"Nombre total de mesures = {r.get('n_total')}",
            f"Interaction opérateur × pièce : {'incluse' if r.get('interaction_included') else 'non incluse'}",
            "\nRésultats principaux :",
            f"% Contribution Gage R&R = {fmt(pct_contrib_grr, 2, ' %')}",
            f"% Study Variation Gage R&R = {fmt(pct_sv_grr, 2, ' %')}",
            f"% Tolérance Gage R&R = {fmt(pct_tol, 2, ' %')}",
            f"ndc = {fmt(ndc, 0)}",
            "\nLecture des critères :",
            f"% Contribution : {status_pct(pct_contrib_grr)}",
            f"% Study Variation : {status_pct(pct_sv_grr)}",
            f"% Tolérance : {status_pct(pct_tol)}",
            f"ndc : {status_ndc(ndc)}",
            "\nRépartition :",
            "La variation observée est principalement liée aux pièces." if (pct_contrib_part or 0) >= 50 else "La variation observée n’est pas principalement portée par les pièces.",
            "La contribution du système de mesure est faible dans cette étude." if (pct_contrib_grr or 999) < 10 else "La contribution du système de mesure est élevée dans cette étude.",
            f"σ Gage R&R = {fmt(r.get('sigma_gage_rr'), 4)}",
            f"σ Répétabilité = {fmt(r.get('sigma_repeatability'), 4)}",
            f"σ Reproductibilité = {fmt(r.get('sigma_reproducibility'), 4)}",
            f"σ Interaction = {fmt(r.get('sigma_interaction'), 4)}",
            f"σ Pièce = {fmt(r.get('sigma_part'), 4)}",
            f"σ Totale = {fmt(r.get('sigma_total'), 4)}",
            "\nTolérance :",
            f"6 × σ Gage R&R = {fmt(six_sigma_grr, 4)}",
            f"Tolérance renseignée = {fmt(r.get('tolerance'), 4)}",
            f"%Tolérance = {fmt(pct_tol, 2, ' %')}",
            "\nANOVA :",
            f"Pièces : p-value {pval(r.get('p_parts'))}",
            f"Opérateurs : p-value {pval(r.get('p_operators'))}",
            f"Interaction opérateur × pièce : p-value {pval(r.get('p_interaction'))}",
            "\nConclusion :",
            f"Résultat {status_pct(pct_sv_grr)} selon la variation observée.",
            f"Résultat {status_pct(pct_tol)} selon la tolérance renseignée.",
            f"Résultat {status_ndc(ndc)} selon le nombre de catégories distinctes.",
            "\n" + "=" * 60,
            "Détails techniques",
            "=" * 60,
            "\nANOVA :",
        ]

        for source, ss, df, ms, f, p in [
            ("Pièces", "ss_parts", "df_parts", "ms_parts", "f_parts", "p_parts"),
            ("Opérateurs", "ss_operators", "df_operators", "ms_operators", "f_operators", "p_operators"),
            ("Interaction opérateur×pièce", "ss_interaction", "df_interaction", "ms_interaction", "f_interaction", "p_interaction"),
            ("Répétabilité", "ss_repeatability", "df_repeatability", "ms_repeatability", None, None),
            ("Total", "ss_total", "df_total", None, None, None),
        ]:
            lines.append(
                f"- {source}: SS={fmt(r.get(ss), 4)}, df={fmt(r.get(df), 0)}, "
                f"MS={fmt(r.get(ms), 4)}, F={fmt(r.get(f), 4)}, p={pval(r.get(p))}"
            )

        lines.append("\nComposantes de variance :")
        for label, key in [
            ("Répétabilité", "sigma_repeatability"),
            ("Reproductibilité", "sigma_reproducibility"),
            ("Interaction", "sigma_interaction"),
            ("Gage R&R", "sigma_gage_rr"),
            ("Pièce", "sigma_part"),
            ("Totale", "sigma_total"),
        ]:
            lines.append(f"- {label}: σ={fmt(r.get(key), 6)}")

        lines.append("\n% Contribution :")
        for label, key in [
            ("Répétabilité", "pct_contribution_repeatability"),
            ("Reproductibilité", "pct_contribution_reproducibility"),
            ("Interaction", "pct_contribution_interaction"),
            ("Gage R&R", "pct_contribution_gage_rr"),
            ("Pièce", "pct_contribution_part"),
        ]:
            lines.append(f"- {label}: {fmt(r.get(key), 2, ' %')}")

        lines.append("\n% Study Variation :")
        for label, key in [
            ("Répétabilité", "pct_study_variation_repeatability"),
            ("Reproductibilité", "pct_study_variation_reproducibility"),
            ("Interaction", "pct_study_variation_interaction"),
            ("Gage R&R", "pct_study_variation_gage_rr"),
            ("Pièce", "pct_study_variation_part"),
        ]:
            lines.append(f"- {label}: {fmt(r.get(key), 2, ' %')}")

        return "\n".join(lines)
