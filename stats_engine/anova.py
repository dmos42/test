import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional


class ANOVA_N_Factor:
    """ANOVA à un facteur pour comparer N groupes.

    API compatible avec l'ancien module : ANOVA_N_Factor(groups, labels=None, factor_labels=None).
    Améliorations : filtrage NaN/Inf, erreurs structurées, post-hoc Welch + Bonferroni.
    """

    def __init__(self, groups: List[np.ndarray], labels: List[str] = None, factor_labels: List[str] = None, alpha: float = 0.05):
        self.raw_groups = [np.asarray(g, dtype=float) for g in groups]
        self.labels = labels or [f"Groupe {i+1}" for i in range(len(groups))]
        self.factor_labels = factor_labels or []
        self.alpha = float(alpha) if 0 < float(alpha) < 1 else 0.05
        self.groups: List[np.ndarray] = []
        self.n_input_each: List[int] = []
        self.n_removed_non_finite_each: List[int] = []
        self._results: Dict[str, Any] = {}
        self._compute()

    def _clean_groups(self):
        self.groups = []
        self.n_input_each = []
        self.n_removed_non_finite_each = []
        for g in self.raw_groups:
            self.n_input_each.append(int(g.size))
            mask = np.isfinite(g)
            self.n_removed_non_finite_each.append(int(g.size - np.sum(mask)))
            self.groups.append(g[mask])

    def _invalid(self, errors: List[str], warnings: List[str] = None):
        self._results = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings or [],
            "k": len(self.raw_groups),
            "labels": self.labels,
            "n_input_each": self.n_input_each,
            "n_removed_non_finite_each": self.n_removed_non_finite_each,
            "n_total": int(sum(len(g) for g in self.groups)) if self.groups else 0,
        }

    def _compute(self):
        self._clean_groups()
        warnings = []
        errors = []
        k = len(self.groups)
        if k < 2:
            errors.append("Au moins deux groupes sont nécessaires pour une ANOVA.")
        if len(self.labels) != k:
            warnings.append("Nombre de libellés différent du nombre de groupes ; libellés automatiques utilisés.")
            self.labels = [f"Groupe {i+1}" for i in range(k)]
        if any(n > 0 for n in self.n_removed_non_finite_each):
            warnings.append(f"Valeurs non finies exclues par groupe : {self.n_removed_non_finite_each}.")
        too_small = [self.labels[i] for i, g in enumerate(self.groups) if len(g) < 2]
        if too_small:
            errors.append("Chaque groupe doit contenir au moins 2 valeurs numériques finies. Groupes concernés : " + ", ".join(too_small))
        if errors:
            self._invalid(errors, warnings)
            return

        n_each = [len(g) for g in self.groups]
        n_total = int(sum(n_each))
        all_data = np.concatenate(self.groups)
        grand_mean = float(np.mean(all_data))
        ss_between = float(sum(n * (np.mean(g) - grand_mean) ** 2 for n, g in zip(n_each, self.groups)))
        ss_within = float(sum(np.sum((g - np.mean(g)) ** 2) for g in self.groups))
        ss_total = float(np.sum((all_data - grand_mean) ** 2))
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1
        ms_between = ss_between / df_between if df_between > 0 else np.nan
        ms_within = ss_within / df_within if df_within > 0 else np.nan
        f_stat = ms_between / ms_within if np.isfinite(ms_within) and ms_within > 0 else np.inf
        p_value = float(stats.f.sf(f_stat, df_between, df_within)) if df_within > 0 and np.isfinite(f_stat) else 0.0
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        pairwise_results = self._pairwise_welch_bonferroni()
        normality_results = []
        for i, g in enumerate(self.groups):
            if len(g) >= 3 and len(np.unique(g)) > 1:
                sw_stat, sw_p = stats.shapiro(g)
                status = "ok"
                normal = bool(sw_p > self.alpha)
            else:
                sw_stat, sw_p = np.nan, np.nan
                status = "non_applicable"
                normal = None
            normality_results.append({"group": self.labels[i], "stat": float(sw_stat) if np.isfinite(sw_stat) else None, "p": float(sw_p) if np.isfinite(sw_p) else None, "normal": normal, "status": status})
        try:
            levene_stat, levene_p = stats.levene(*self.groups)
            levene_stat = float(levene_stat) if np.isfinite(levene_stat) else None
            levene_p = float(levene_p) if np.isfinite(levene_p) else None
            homogeneity = bool(levene_p is not None and levene_p > self.alpha)
        except Exception as e:
            levene_stat, levene_p, homogeneity = None, None, None
            warnings.append(f"Test de Levene non calculable : {e}")
        descriptives = []
        for i, g in enumerate(self.groups):
            descriptives.append({"group": self.labels[i], "n": len(g), "mean": float(np.mean(g)), "std": float(np.std(g, ddof=1)), "min": float(np.min(g)), "max": float(np.max(g)), "median": float(np.median(g))})
        self._results = {
            "is_valid": True,
            "errors": [],
            "warnings": warnings,
            "analysis_type": "one_way_anova",
            "k": k,
            "labels": self.labels,
            "n_input_each": self.n_input_each,
            "n_removed_non_finite_each": self.n_removed_non_finite_each,
            "n_total": n_total,
            "n_each": n_each,
            "grand_mean": grand_mean,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ss_total": ss_total,
            "df_between": df_between,
            "df_within": df_within,
            "df_total": df_total,
            "ms_between": ms_between,
            "ms_within": ms_within,
            "f_stat": float(f_stat),
            "p_value": p_value,
            "eta_squared": float(eta_squared),
            "pairwise_method": "Welch t-test + Bonferroni",
            "pairwise_results": pairwise_results,
            "tukey_results": pairwise_results,  # compatibilité historique, mais méthode renommée dans pairwise_method
            "normality_results": normality_results,
            "levene_stat": levene_stat,
            "levene_p": levene_p,
            "homogeneity": homogeneity,
            "descriptives": descriptives,
        }

    def _pairwise_welch_bonferroni(self):
        results = []
        k = len(self.groups)
        m = max(k * (k - 1) // 2, 1)
        for i in range(k):
            for j in range(i + 1, k):
                mean_diff = float(np.mean(self.groups[i]) - np.mean(self.groups[j]))
                stat, p_raw = stats.ttest_ind(self.groups[i], self.groups[j], equal_var=False)
                p_adj = min(float(p_raw) * m, 1.0) if np.isfinite(p_raw) else None
                results.append({
                    "comparison": f"{self.labels[i]} vs {self.labels[j]}",
                    "mean_diff": mean_diff,
                    "statistic": float(stat) if np.isfinite(stat) else None,
                    "p_raw": float(p_raw) if np.isfinite(p_raw) else None,
                    "p_value": p_adj,
                    "p_adjusted": p_adj,
                    "method": "Welch t-test + Bonferroni",
                    "significant": bool(p_adj is not None and p_adj < self.alpha),
                })
        return results

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        if not r.get("is_valid", False):
            return "\n".join(["=" * 60, "ANOVA À UN FACTEUR - étude non valide", "=" * 60] + [f"- {e}" for e in r.get("errors", [])])
        lines = ["=" * 60, f"ANOVA À UN FACTEUR - comparaison de {r['k']} groupes", "=" * 60]
        lines.append(f"\nNombre de groupes : {r['k']}")
        lines.append(f"Total observations : {r['n_total']}")
        if r.get("warnings"):
            lines.append("\nAvertissements :")
            lines.extend([f" - {w}" for w in r["warnings"]])
        lines.append("\n--- Statistiques descriptives ---")
        for d in r["descriptives"]:
            lines.append(f" {d['group']}: n={d['n']}, moy={d['mean']:.4f}, σ={d['std']:.4f}, méd={d['median']:.4f}")
        lines.append("\n--- Résultats ANOVA ---")
        lines.append(" Source              SS        df        MS         F        p-value")
        lines.append(" " + "-" * 70)
        lines.append(f" Inter-groupes {r['ss_between']:10.4f} {r['df_between']:5d} {r['ms_between']:10.4f} {r['f_stat']:9.4f} {r['p_value']:.6f}")
        lines.append(f" Intra-groupe  {r['ss_within']:10.4f} {r['df_within']:5d} {r['ms_within']:10.4f}")
        lines.append(f" Total         {r['ss_total']:10.4f} {r['df_total']:5d}")
        lines.append(f"\nTaille d'effet (η²) = {r['eta_squared']:.4f}")
        lines.append("\n⚠ Différence significative détectée (p < 0.05)" if r['p_value'] < self.alpha else "\n✓ Aucune différence significative (p ≥ 0.05)")
        if r.get("pairwise_results"):
            lines.append("\n--- Comparaisons multiples — Welch + Bonferroni ---")
            for t in r["pairwise_results"]:
                sig = "*" if t["significant"] else " "
                ptxt = "n/a" if t["p_value"] is None else f"{t['p_value']:.4f}"
                lines.append(f" {t['comparison']}: diff={t['mean_diff']:.4f}, p-ajustée={ptxt} {sig}")
        lines.append("=" * 60)
        return "\n".join(lines)


class ANOVA_Two_Factor:
    """ANOVA à deux facteurs avec interaction, plan équilibré."""

    def __init__(self, data: np.ndarray, n_factor_a: int, n_factor_b: int, n_replicates: int, factor_a_labels: List[str] = None, factor_b_labels: List[str] = None, alpha: float = 0.05):
        self.raw_data = np.asarray(data, dtype=float)
        self.n_factor_a = int(n_factor_a)
        self.n_factor_b = int(n_factor_b)
        self.n_replicates = int(n_replicates)
        self.factor_a_labels = factor_a_labels or [f"A{i}" for i in range(self.n_factor_a)]
        self.factor_b_labels = factor_b_labels or [f"B{j}" for j in range(self.n_factor_b)]
        self.alpha = float(alpha) if 0 < float(alpha) < 1 else 0.05
        self._results: Dict[str, Any] = {}
        self.data = None
        self._compute()

    def _invalid(self, errors: List[str], warnings: List[str] = None):
        self._results = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings or [],
            "n_factor_a": self.n_factor_a,
            "n_factor_b": self.n_factor_b,
            "n_replicates": self.n_replicates,
            "n_input": int(self.raw_data.size),
        }

    def _compute(self):
        errors, warnings = [], []
        if self.n_factor_a < 2 or self.n_factor_b < 2 or self.n_replicates < 2:
            errors.append("Deux facteurs avec au moins 2 niveaux chacun et au moins 2 répétitions sont requis.")
        expected = self.n_factor_a * self.n_factor_b * self.n_replicates
        if self.raw_data.size != expected:
            errors.append(f"Taille des données incohérente : attendu {expected} valeurs, reçu {self.raw_data.size}.")
        if not np.all(np.isfinite(self.raw_data)):
            errors.append("Valeurs non finies détectées (NaN/Inf) : l'ANOVA deux facteurs exige des données numériques finies.")
        if errors:
            self._invalid(errors, warnings)
            return
        self.data = self.raw_data.reshape(self.n_factor_a, self.n_factor_b, self.n_replicates)
        data = self.data
        a, b, r = self.n_factor_a, self.n_factor_b, self.n_replicates
        n_total = a * b * r
        grand_mean = float(np.mean(data))
        a_means = np.mean(data, axis=(1, 2))
        b_means = np.mean(data, axis=(0, 2))
        cell_means = np.mean(data, axis=2)
        ss_total = float(np.sum((data - grand_mean) ** 2))
        ss_a = float(b * r * np.sum((a_means - grand_mean) ** 2))
        ss_b = float(a * r * np.sum((b_means - grand_mean) ** 2))
        ss_ab = float(r * np.sum((cell_means - a_means[:, np.newaxis] - b_means[np.newaxis, :] + grand_mean) ** 2))
        ss_error = float(np.sum((data - cell_means[:, :, np.newaxis]) ** 2))
        df_a, df_b, df_ab = a - 1, b - 1, (a - 1) * (b - 1)
        df_error, df_total = a * b * (r - 1), n_total - 1
        ms_a = ss_a / df_a if df_a > 0 else np.nan
        ms_b = ss_b / df_b if df_b > 0 else np.nan
        ms_ab = ss_ab / df_ab if df_ab > 0 else np.nan
        ms_error = ss_error / df_error if df_error > 0 else np.nan
        f_a = ms_a / ms_error if np.isfinite(ms_error) and ms_error > 0 else np.inf
        f_b = ms_b / ms_error if np.isfinite(ms_error) and ms_error > 0 else np.inf
        f_ab = ms_ab / ms_error if np.isfinite(ms_error) and ms_error > 0 else np.inf
        p_a = float(stats.f.sf(f_a, df_a, df_error)) if df_error > 0 and np.isfinite(f_a) else 0.0
        p_b = float(stats.f.sf(f_b, df_b, df_error)) if df_error > 0 and np.isfinite(f_b) else 0.0
        p_ab = float(stats.f.sf(f_ab, df_ab, df_error)) if df_error > 0 and np.isfinite(f_ab) else 0.0
        eta_a = ss_a / ss_total if ss_total > 0 else 0.0
        eta_b = ss_b / ss_total if ss_total > 0 else 0.0
        eta_ab = ss_ab / ss_total if ss_total > 0 else 0.0
        self._results = {
            "is_valid": True,
            "errors": [],
            "warnings": warnings,
            "n_total": n_total,
            "n_input": int(self.raw_data.size),
            "n_factor_a": a,
            "n_factor_b": b,
            "n_replicates": r,
            "grand_mean": grand_mean,
            "ss_a": ss_a, "ss_b": ss_b, "ss_ab": ss_ab, "ss_error": ss_error, "ss_total": ss_total,
            "df_a": df_a, "df_b": df_b, "df_ab": df_ab, "df_error": df_error, "df_total": df_total,
            "ms_a": float(ms_a), "ms_b": float(ms_b), "ms_ab": float(ms_ab), "ms_error": float(ms_error),
            "f_a": float(f_a), "f_b": float(f_b), "f_ab": float(f_ab),
            "p_a": p_a, "p_b": p_b, "p_ab": p_ab,
            "eta_a": float(eta_a), "eta_b": float(eta_b), "eta_ab": float(eta_ab),
            "a_means": a_means.tolist(), "b_means": b_means.tolist(), "cell_means": cell_means.tolist(),
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        if not r.get("is_valid", False):
            return "\n".join(["=" * 60, "ANOVA À DEUX FACTEURS - étude non valide", "=" * 60] + [f"- {e}" for e in r.get("errors", [])])
        lines = ["=" * 60, "ANALYSE DE VARIANCE À DEUX FACTEURS", "=" * 60]
        lines.append(f"\nTotal observations : {r['n_total']}")
        lines.append(f"Facteur A : {r['n_factor_a']} niveaux, Facteur B : {r['n_factor_b']} niveaux, Répétitions : {r['n_replicates']}")
        lines.append("\n--- Résultats ---")
        lines.append(" Source              SS        df        MS         F        p-value      η²")
        lines.append(" " + "-" * 80)
        lines.append(f" Facteur A    {r['ss_a']:10.4f} {r['df_a']:5d} {r['ms_a']:10.4f} {r['f_a']:9.4f} {r['p_a']:.6f} {r['eta_a']:.4f}")
        lines.append(f" Facteur B    {r['ss_b']:10.4f} {r['df_b']:5d} {r['ms_b']:10.4f} {r['f_b']:9.4f} {r['p_b']:.6f} {r['eta_b']:.4f}")
        lines.append(f" Interaction  {r['ss_ab']:10.4f} {r['df_ab']:5d} {r['ms_ab']:10.4f} {r['f_ab']:9.4f} {r['p_ab']:.6f} {r['eta_ab']:.4f}")
        lines.append(f" Erreur       {r['ss_error']:10.4f} {r['df_error']:5d} {r['ms_error']:10.4f}")
        lines.append(f" Total        {r['ss_total']:10.4f} {r['df_total']:5d}")
        lines.append("\n--- Interprétation ---")
        lines.append(f" {'⚠' if r['p_a'] < self.alpha else '✓'} Facteur A {'significatif' if r['p_a'] < self.alpha else 'non significatif'} (p = {r['p_a']:.4f})")
        lines.append(f" {'⚠' if r['p_b'] < self.alpha else '✓'} Facteur B {'significatif' if r['p_b'] < self.alpha else 'non significatif'} (p = {r['p_b']:.4f})")
        lines.append(f" {'⚠' if r['p_ab'] < self.alpha else '✓'} Interaction A×B {'significative' if r['p_ab'] < self.alpha else 'non significative'} (p = {r['p_ab']:.4f})")
        lines.append("=" * 60)
        return "\n".join(lines)
