import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


class GageRRStudy:
    """Étude de capabilité du système de mesure (Gage R&R)"""

    def __init__(self, data: np.ndarray, n_operators: int, n_parts: int, n_trials: int,
                 spec_lsl: float = None, spec_usl: float = None,
                 tolerance: float = None):
        """
        data: array 1D avec les mesures dans l'ordre:
              [op1_part1_trial1, op1_part1_trial2, ..., op1_part2_trial1, ...]
              Ou array 3D (operators, parts, trials)
        """
        if isinstance(data, np.ndarray) and data.ndim == 1:
            self.data = data.reshape(n_operators, n_parts, n_trials)
        else:
            self.data = np.array(data, dtype=float)

        self.n_operators = n_operators
        self.n_parts = n_parts
        self.n_trials = n_trials
        self.spec_lsl = spec_lsl
        self.spec_usl = spec_usl
        self.tolerance = tolerance
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        data = self.data
        n_op = self.n_operators
        n_part = self.n_parts
        n_rep = self.n_trials

        # Moyennes par opérateur
        op_means = np.mean(data, axis=(1, 2))  # Moyenne par opérateur
        part_means = np.mean(data, axis=(0, 2))  # Moyenne par pièce
        cell_means = np.mean(data, axis=2)  # Moyenne par cellule (opérateur, pièce)

        grand_mean = np.mean(data)

        # ANOVA pour décomposer la variance
        # SS_Total
        ss_total = np.sum((data - grand_mean) ** 2)

        # SS_Operators
        ss_operators = n_part * n_rep * np.sum((op_means - grand_mean) ** 2)

        # SS_Parts
        ss_parts = n_op * n_rep * np.sum((part_means - grand_mean) ** 2)

        # SS_Cells (interaction opérateur × pièce + répétabilité)
        ss_cells = n_rep * np.sum((cell_means - op_means[:, np.newaxis] - part_means[np.newaxis, :] + grand_mean) ** 2)

        # SS_Repeatability (EV - Equipment Variation)
        ss_repeatability = ss_total - ss_operators - ss_parts - ss_cells
        # Alternative: somme des écarts par rapport à la moyenne de cellule
        ss_repeatability_alt = np.sum((data - cell_means[:, :, np.newaxis]) ** 2)
        ss_repeatability = ss_repeatability_alt

        # Degrés de liberté
        df_operators = n_op - 1
        df_parts = n_part - 1
        df_interaction = (n_op - 1) * (n_part - 1)
        df_repeatability = n_op * n_part * (n_rep - 1)
        df_total = n_op * n_part * n_rep - 1

        # Carrés moyens
        ms_operators = ss_operators / df_operators if df_operators > 0 else 0
        ms_parts = ss_parts / df_parts if df_parts > 0 else 0
        ms_repeatability = ss_repeatability / df_repeatability if df_repeatability > 0 else 0

        # Variance components (méthode ANOVA)
        ms_interaction = ss_cells / df_interaction if df_interaction > 0 else 0

        # sigma_repeatability (EV)
        sigma_repeatability = np.sqrt(ms_repeatability)

        # sigma_reproducibility (AV)
        # AV = sqrt(MS_Operators - MS_Interaction) / (n_parts * n_rep)
        var_reproducibility = max(0, (ms_operators - ms_interaction) / (n_part * n_rep))
        sigma_reproducibility = np.sqrt(var_reproducibility)

        # sigma_interaction
        var_interaction = max(0, (ms_interaction - ms_repeatability) / n_rep)
        sigma_interaction = np.sqrt(var_interaction)

        # Gage R&R total
        var_gage_rr = ms_repeatability + var_reproducibility
        sigma_gage_rr = np.sqrt(var_gage_rr)

        # sigma_part-to-part
        sigma_part = np.sqrt(max(0, (ms_parts - ms_interaction) / (n_op * n_rep)))

        # sigma_total
        sigma_total = np.sqrt(var_gage_rr + sigma_part ** 2)

        # %Study Variation (6 sigma)
        k = 6.0  # Study variation multiplier
        sv_repeatability = k * sigma_repeatability
        sv_reproducibility = k * sigma_reproducibility
        sv_gage_rr = k * sigma_gage_rr
        sv_part = k * sigma_part
        sv_total = k * sigma_total

        # %Contribution
        pct_contribution_repeatability = (var_gage_rr / (var_gage_rr + sigma_part ** 2) * 100) if (var_gage_rr + sigma_part ** 2) > 0 else 0
        pct_contribution_reproducibility = (var_reproducibility / (var_gage_rr + sigma_part ** 2) * 100) if (var_gage_rr + sigma_part ** 2) > 0 else 0
        pct_contribution_gage_rr = (var_gage_rr / (var_gage_rr + sigma_part ** 2) * 100) if (var_gage_rr + sigma_part ** 2) > 0 else 0
        pct_contribution_part = (sigma_part ** 2 / (var_gage_rr + sigma_part ** 2) * 100) if (var_gage_rr + sigma_part ** 2) > 0 else 0

        # %Tolerance (si spécifications fournies)
        if self.tolerance is not None:
            pct_tolerance = (sv_gage_rr / self.tolerance) * 100
        elif self.spec_lsl is not None and self.spec_usl is not None:
            pct_tolerance = (sv_gage_rr / (self.spec_usl - self.spec_lsl)) * 100
        else:
            pct_tolerance = None

        # ndc (Number of Distinct Categories)
        ndc = max(1, int(sigma_part / sigma_gage_rr * 1.41))

        # Détermination de l'acceptabilité
        if pct_contribution_gage_rr is not None:
            if pct_contribution_gage_rr < 1:
                acceptability = "Acceptable"
            elif pct_contribution_gage_rr < 9:
                acceptability = "Conditionnellement acceptable"
            else:
                acceptability = "Inacceptable"
        else:
            acceptability = "N/A"

        self._results = {
            "grand_mean": grand_mean,
            "ss_operators": ss_operators,
            "ss_parts": ss_parts,
            "ss_repeatability": ss_repeatability,
            "df_operators": df_operators,
            "df_parts": df_parts,
            "df_repeatability": df_repeatability,
            "ms_operators": ms_operators,
            "ms_parts": ms_parts,
            "ms_repeatability": ms_repeatability,
            "sigma_repeatability": sigma_repeatability,
            "sigma_reproducibility": sigma_reproducibility,
            "sigma_interaction": sigma_interaction,
            "sigma_gage_rr": sigma_gage_rr,
            "sigma_part": sigma_part,
            "sigma_total": sigma_total,
            "sv_repeatability": sv_repeatability,
            "sv_reproducibility": sv_reproducibility,
            "sv_gage_rr": sv_gage_rr,
            "sv_part": sv_part,
            "sv_total": sv_total,
            "pct_contribution_repeatability": pct_contribution_repeatability,
            "pct_contribution_reproducibility": pct_contribution_reproducibility,
            "pct_contribution_gage_rr": pct_contribution_gage_rr,
            "pct_contribution_part": pct_contribution_part,
            "pct_tolerance": pct_tolerance,
            "ndc": ndc,
            "acceptability": acceptability,
            "n_operators": n_op,
            "n_parts": n_part,
            "n_trials": n_rep,
            "op_means": op_means.tolist(),
            "part_means": part_means.tolist(),
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = ["=" * 60, "ÉTUDE GAGE R&R", "=" * 60]
        lines.append(f"\nOpérateurs : {r['n_operators']}, Pièces : {r['n_parts']}, Répétitions : {r['n_trials']}")
        lines.append(f"\n--- Composantes de variance ---")
        lines.append(f"  σ_répétabilité   = {r['sigma_repeatability']:.6f}")
        lines.append(f"  σ_reproductibilité = {r['sigma_reproducibility']:.6f}")
        lines.append(f"  σ_Gage R&R       = {r['sigma_gage_rr']:.6f}")
        lines.append(f"  σ_Pièce          = {r['sigma_part']:.6f}")
        lines.append(f"  σ_Totale         = {r['sigma_total']:.6f}")
        lines.append(f"\n--- % Contribution ---")
        lines.append(f"  Répétabilité     = {r['pct_contribution_repeatability']:.2f}%")
        lines.append(f"  Reproductibilité = {r['pct_contribution_reproducibility']:.2f}%")
        lines.append(f"  Gage R&R         = {r['pct_contribution_gage_rr']:.2f}%")
        lines.append(f"  Pièce            = {r['pct_contribution_part']:.2f}%")
        lines.append(f"\n--- Étude de variation (6σ) ---")
        lines.append(f"  SV Répétabilité  = {r['sv_repeatability']:.6f}")
        lines.append(f"  SV Reproductibilité = {r['sv_reproducibility']:.6f}")
        lines.append(f"  SV Gage R&R      = {r['sv_gage_rr']:.6f}")
        lines.append(f"  SV Pièce         = {r['sv_part']:.6f}")
        lines.append(f"  SV Totale        = {r['sv_total']:.6f}")

        if r['pct_tolerance'] is not None:
            lines.append(f"\n--- % Tolérance ---")
            lines.append(f"  %Tol = {r['pct_tolerance']:.2f}%")

        lines.append(f"\n--- Nombre de catégories distinctes ---")
        lines.append(f"  ndc = {r['ndc']}")

        lines.append(f"\n--- Conclusion ---")
        lines.append(f"  {r['acceptability']}")

        if r['pct_contribution_gage_rr'] < 10:
            lines.append(f"  ✓ Système de mesure acceptable")
        else:
            lines.append(f"  ⚠ Système de mesure à améliorer")

        lines.append("=" * 60)
        return "\n".join(lines)
