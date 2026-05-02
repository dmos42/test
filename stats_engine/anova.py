import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


class ANOVA_N_Factor:
    """ANOVA à N facteurs (sélectionnable)"""

    def __init__(self, groups: List[np.ndarray], labels: List[str] = None,
                 factor_labels: List[str] = None):
        """
        groups: liste de arrays, chaque array représentant un groupe
        labels: noms des groupes
        factor_labels: noms des facteurs (pour ANOVA à 2+ facteurs)
        """
        self.groups = [np.array(g, dtype=float) for g in groups]
        self.labels = labels or [f"Groupe {i+1}" for i in range(len(groups))]
        self.factor_labels = factor_labels or []
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        k = len(self.groups)
        n_each = [len(g) for g in self.groups]
        n_total = sum(n_each)

        # Données combinées
        all_data = np.concatenate(self.groups)
        grand_mean = np.mean(all_data)

        # Sum of Squares Between (SSB)
        ss_between = sum(n * (np.mean(g) - grand_mean) ** 2 for n, g in zip(n_each, self.groups))

        # Sum of Squares Within (SSW)
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in self.groups)

        # Sum of Squares Total (SST)
        ss_total = np.sum((all_data - grand_mean) ** 2)

        # Degrees of freedom
        df_between = k - 1
        df_within = n_total - k
        df_total = n_total - 1

        # Mean Squares
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0

        # F-statistic
        f_stat = ms_between / ms_within if ms_within > 0 else float('inf')
        p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)

        # Eta-squared (effect size)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Post-hoc Tukey HSD
        tukey_results = []
        if k > 2:
            tukey = self._tukey_hsd(all_data, self.groups, self.labels, ms_within, df_within)
            tukey_results = tukey

        # Vérification des hypothèses
        # Normalité (Shapiro-Wilk) pour chaque groupe
        normality_results = []
        for i, g in enumerate(self.groups):
            if len(g) >= 3:
                sw_stat, sw_p = stats.shapiro(g)
            else:
                sw_stat, sw_p = float('nan'), float('nan')
            normality_results.append({
                "group": self.labels[i],
                "stat": sw_stat,
                "p": sw_p,
                "normal": sw_p > 0.05 if not np.isnan(sw_p) else False,
            })

        # Homogénéité des variances (Levene)
        levene_stat, levene_p = stats.levene(*self.groups)

        # Statistiques descriptives
        descriptives = []
        for i, g in enumerate(self.groups):
            descriptives.append({
                "group": self.labels[i],
                "n": len(g),
                "mean": np.mean(g),
                "std": np.std(g, ddof=1),
                "min": np.min(g),
                "max": np.max(g),
                "median": np.median(g),
            })

        self._results = {
            "k": k,
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
            "f_stat": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "tukey_results": tukey_results,
            "normality_results": normality_results,
            "levene_stat": levene_stat,
            "levene_p": levene_p,
            "homogeneity": levene_p > 0.05,
            "descriptives": descriptives,
        }

    def _tukey_hsd(self, all_data, groups, labels, ms_within, df_within):
        results = []
        k = len(groups)
        for i in range(k):
            for j in range(i + 1, k):
                mean_diff = np.mean(groups[i]) - np.mean(groups[j])
                n_i = len(groups[i])
                n_j = len(groups[j])
                se = np.sqrt(ms_within * (1 / n_i + 1 / n_j) / 2)
                q_stat = abs(mean_diff) / se if se > 0 else 0
                p_value = 1 - stats.t.cdf(q_stat, df_within) * 2
                p_value = max(0, min(1, p_value))
                results.append({
                    "comparison": f"{labels[i]} vs {labels[j]}",
                    "mean_diff": mean_diff,
                    "se": se,
                    "q_stat": q_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                })
        return results

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, f"ANALYSE DE VARIANCE (ANOVA {r['k']} facteurs)", "=" * 60]
        lines.append(f"\nNombre de groupes : {r['k']}")
        lines.append(f"Total observations : {r['n_total']}")
        lines.append(f"\n--- Statistiques descriptives ---")
        for d in r['descriptives']:
            lines.append(f"  {d['group']}: n={d['n']}, moy={d['mean']:.4f}, σ={d['std']:.4f}, méd={d['median']:.4f}")

        lines.append(f"\n--- Test d'hypothèses ---")
        normality_ok = all(n['normal'] for n in r['normality_results'])
        lines.append(f"  Normalité : {'✓ Acceptée' if normality_ok else '⚠ Rejetée'}")
        lines.append(f"  Homogénéité des variances (Levene) : {'✓ Acceptée' if r['homogeneity'] else '⚠ Rejetée'} (p={r['levene_p']:.4f})")

        lines.append(f"\n--- Résultats ANOVA ---")
        lines.append(f"  Source        | SS         | df | MS         | F         | p-value")
        lines.append(f"  {'-'*60}")
        lines.append(f"  Inter-groupes | {r['ss_between']:10.4f} | {r['df_between']:2d} | {r['ms_between']:10.4f} | {r['f_stat']:9.4f} | {r['p_value']:.6f}")
        lines.append(f"  Intra-groupe  | {r['ss_within']:10.4f} | {r['df_within']:2d} | {r['ms_within']:10.4f}")
        lines.append(f"  Total         | {r['ss_total']:10.4f} | {r['df_total']:2d}")

        lines.append(f"\nTaille d'effet (η²) = {r['eta_squared']:.4f}")
        if r['eta_squared'] > 0.14:
            lines.append("  Effet large")
        elif r['eta_squared'] > 0.06:
            lines.append("  Effet moyen")
        else:
            lines.append("  Effet petit")

        if r['p_value'] < 0.05:
            lines.append(f"\n⚠ Différence significative détectée (p < 0.05)")
        else:
            lines.append(f"\n✓ Aucune différence significative (p ≥ 0.05)")

        if r['tukey_results']:
            lines.append(f"\n--- Post-hoc Tukey HSD ---")
            for t in r['tukey_results']:
                sig = "*" if t['significant'] else " "
                lines.append(f"  {t['comparison']}: diff={t['mean_diff']:.4f}, p={t['p_value']:.4f} {sig}")

        lines.append("=" * 60)
        return "\n".join(lines)


class ANOVA_Two_Factor:
    """ANOVA à deux facteurs avec interaction"""

    def __init__(self, data: np.ndarray, n_factor_a: int, n_factor_b: int, n_replicates: int,
                 factor_a_labels: List[str] = None, factor_b_labels: List[str] = None):
        """
        data: array 1D ou 3D (factor_a, factor_b, replicates)
        """
        if isinstance(data, np.ndarray) and data.ndim == 1:
            self.data = data.reshape(n_factor_a, n_factor_b, n_replicates)
        else:
            self.data = np.array(data, dtype=float)

        self.n_factor_a = n_factor_a
        self.n_factor_b = n_factor_b
        self.n_replicates = n_replicates
        self.factor_a_labels = factor_a_labels or [f"A{i}" for i in range(n_factor_a)]
        self.factor_b_labels = factor_b_labels or [f"B{j}" for j in range(n_factor_b)]
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        data = self.data
        a, b, r = self.n_factor_a, self.n_factor_b, self.n_replicates
        n_total = a * b * r
        grand_mean = np.mean(data)

        # Means
        a_means = np.mean(data, axis=(1, 2))  # Moyenne par niveau A
        b_means = np.mean(data, axis=(0, 2))  # Moyenne par niveau B
        cell_means = np.mean(data, axis=2)  # Moyenne par cellule

        # Sum of Squares
        ss_total = np.sum((data - grand_mean) ** 2)
        ss_a = b * r * np.sum((a_means - grand_mean) ** 2)
        ss_b = a * r * np.sum((b_means - grand_mean) ** 2)
        ss_ab = r * np.sum((cell_means - a_means[:, np.newaxis] - b_means[np.newaxis, :] + grand_mean) ** 2)
        ss_error = np.sum((data - cell_means[:, :, np.newaxis]) ** 2)

        # Degrees of freedom
        df_a = a - 1
        df_b = b - 1
        df_ab = (a - 1) * (b - 1)
        df_error = a * b * (r - 1)
        df_total = n_total - 1

        # Mean Squares
        ms_a = ss_a / df_a
        ms_b = ss_b / df_b
        ms_ab = ss_ab / df_ab
        ms_error = ss_error / df_error

        # F-statistics
        f_a = ms_a / ms_error
        f_b = ms_b / ms_error
        f_ab = ms_ab / ms_error

        # P-values
        p_a = 1 - stats.f.cdf(f_a, df_a, df_error)
        p_b = 1 - stats.f.cdf(f_b, df_b, df_error)
        p_ab = 1 - stats.f.cdf(f_ab, df_ab, df_error)

        # Eta-squared
        eta_a = ss_a / ss_total
        eta_b = ss_b / ss_total
        eta_ab = ss_ab / ss_total

        self._results = {
            "n_total": n_total,
            "n_factor_a": a,
            "n_factor_b": b,
            "n_replicates": r,
            "grand_mean": grand_mean,
            "ss_a": ss_a, "ss_b": ss_b, "ss_ab": ss_ab, "ss_error": ss_error, "ss_total": ss_total,
            "df_a": df_a, "df_b": df_b, "df_ab": df_ab, "df_error": df_error, "df_total": df_total,
            "ms_a": ms_a, "ms_b": ms_b, "ms_ab": ms_ab, "ms_error": ms_error,
            "f_a": f_a, "f_b": f_b, "f_ab": f_ab,
            "p_a": p_a, "p_b": p_b, "p_ab": p_ab,
            "eta_a": eta_a, "eta_b": eta_b, "eta_ab": eta_ab,
            "a_means": a_means.tolist(),
            "b_means": b_means.tolist(),
            "cell_means": cell_means.tolist(),
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "ANALYSE DE VARIANCE À DEUX FACTEURS", "=" * 60]
        lines.append(f"\nTotal observations : {r['n_total']}")
        lines.append(f"Facteur A : {r['n_factor_a']} niveaux, Facteur B : {r['n_factor_b']} niveaux, Répétitions : {r['n_replicates']}")

        lines.append(f"\n--- Résultats ---")
        lines.append(f"  Source     | SS         | df | MS         | F         | p-value   | η²")
        lines.append(f"  {'-'*70}")
        lines.append(f"  Facteur A  | {r['ss_a']:10.4f} | {r['df_a']:2d} | {r['ms_a']:10.4f} | {r['f_a']:9.4f} | {r['p_a']:.6f} | {r['eta_a']:.4f}")
        lines.append(f"  Facteur B  | {r['ss_b']:10.4f} | {r['df_b']:2d} | {r['ms_b']:10.4f} | {r['f_b']:9.4f} | {r['p_b']:.6f} | {r['eta_b']:.4f}")
        lines.append(f"  Interact.  | {r['ss_ab']:10.4f} | {r['df_ab']:2d} | {r['ms_ab']:10.4f} | {r['f_ab']:9.4f} | {r['p_ab']:.6f} | {r['eta_ab']:.4f}")
        lines.append(f"  Erreur     | {r['ss_error']:10.4f} | {r['df_error']:2d} | {r['ms_error']:10.4f}")
        lines.append(f"  Total      | {r['ss_total']:10.4f} | {r['df_total']:2d}")

        lines.append(f"\n--- Interprétation ---")
        if r['p_a'] < 0.05:
            lines.append(f"  ⚠ Facteur A significatif (p = {r['p_a']:.4f})")
        else:
            lines.append(f"  ✓ Facteur A non significatif (p = {r['p_a']:.4f})")

        if r['p_b'] < 0.05:
            lines.append(f"  ⚠ Facteur B significatif (p = {r['p_b']:.4f})")
        else:
            lines.append(f"  ✓ Facteur B non significatif (p = {r['p_b']:.4f})")

        if r['p_ab'] < 0.05:
            lines.append(f"  ⚠ Interaction A×B significative (p = {r['p_ab']:.4f})")
        else:
            lines.append(f"  ✓ Interaction A×B non significative (p = {r['p_ab']:.4f})")

        lines.append("=" * 60)
        return "\n".join(lines)
