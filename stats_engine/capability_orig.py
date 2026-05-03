import numpy as np
from scipy import stats
from typing import Optional, Dict, Any, Tuple


class CapabilityAnalysis:
    """Statistical process capability analysis (Cp, Cpk, Pp, Ppk, Cpm, etc.)"""

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
        self.subgroup_size = subgroup_size
        self.estimation_method = estimation_method
        self.confidence_level = confidence_level
        self.ppk_method = ppk_method
        self._results: Dict[str, Any] = {}
        self._compute()

    def _compute(self):
        n = len(self.data)
        mean = np.mean(self.data)
        std_overall = np.std(self.data, ddof=1)

        sigma_within = self._estimate_sigma_within()
        sigma_overall = std_overall

        self._results = {
            "n": n,
            "mean": mean,
            "median": np.median(self.data),
            "std_within": sigma_within,
            "std_overall": sigma_overall,
            "min": np.min(self.data),
            "max": np.max(self.data),
            "range": np.ptp(self.data),
            "usl": self.usl,
            "lsl": self.lsl,
            "target": self.target,
        }

        if self.usl is not None and self.lsl is not None:
            self._compute_two_sided(mean, sigma_within, sigma_overall)
        elif self.usl is not None:
            self._compute_upper_only(mean, sigma_within, sigma_overall)
        elif self.lsl is not None:
            self._compute_lower_only(mean, sigma_within, sigma_overall)
        else:
            self._results["error"] = "Au moins une limite (USL ou LSL) doit être spécifiée"

        if self.target is not None:
            self._compute_cpm(mean, sigma_within)

        self._compute_confidence_intervals(sigma_within, sigma_overall, n)
        self._compute_ppm(mean, sigma_within, sigma_overall)

    def _estimate_sigma_within(self) -> float:
        n = len(self.data)

        if self.subgroup_size <= 1:
            if self.estimation_method == "rbar":
                return self._sigma_from_moving_range()
            elif self.estimation_method == "sbar":
                return self._sigma_from_moving_std()
            elif self.estimation_method == "pooled":
                return self._sigma_pooled()
            else:
                return self._sigma_from_moving_range()
        else:
            subgroups = self._create_subgroups()
            if self.estimation_method == "rbar":
                return self._sigma_from_rbar(subgroups)
            elif self.estimation_method == "sbar":
                return self._sigma_from_sbar(subgroups)
            elif self.estimation_method == "pooled":
                return self._sigma_pooled(subgroups)
            else:
                return self._sigma_from_rbar(subgroups)

    def _sigma_from_moving_range(self) -> float:
        mr = np.abs(np.diff(self.data))
        d2 = 1.128
        return np.mean(mr) / d2

    def _sigma_from_moving_std(self) -> float:
        if len(self.data) < 2:
            return np.std(self.data, ddof=1)
        n_window = min(2, len(self.data) - 1)
        c4_values = {1: 0.7979, 2: 0.8862}
        c4 = c4_values.get(n_window, 0.9213)
        stds = []
        for i in range(len(self.data) - n_window):
            window = self.data[i:i + n_window + 1]
            stds.append(np.std(window, ddof=1))
        return np.mean(stds) / c4

    def _create_subgroups(self) -> list:
        n = len(self.data)
        n_subgroups = n // self.subgroup_size
        subgroups = []
        for i in range(n_subgroups):
            start = i * self.subgroup_size
            end = start + self.subgroup_size
            subgroups.append(self.data[start:end])
        return subgroups

    def _sigma_from_rbar(self, subgroups: list) -> float:
        d2_table = {
            2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
            7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078, 11: 3.173,
            12: 3.258, 13: 3.336, 14: 3.407, 15: 3.472,
        }
        ranges = [np.max(sg) - np.min(sg) for sg in subgroups]
        rbar = np.mean(ranges)
        d2 = d2_table.get(self.subgroup_size, 3.078)
        return rbar / d2

    def _sigma_from_sbar(self, subgroups: list) -> float:
        c4_table = {
            2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515,
            7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727, 11: 0.9754,
            12: 0.9776, 13: 0.9794, 14: 0.9810, 15: 0.9823,
        }
        stds = [np.std(sg, ddof=1) for sg in subgroups]
        sbar = np.mean(stds)
        c4 = c4_table.get(self.subgroup_size, 0.9727)
        return sbar / c4

    def _sigma_pooled(self, subgroups: list = None) -> float:
        if subgroups is None:
            return np.std(self.data, ddof=1)
        variances = [np.var(sg, ddof=1) for sg in subgroups]
        n_i = [len(sg) for sg in subgroups]
        pooled_var = sum((n - 1) * v for n, v in zip(n_i, variances)) / sum(n - 1 for n in n_i)
        return np.sqrt(pooled_var)

    def _compute_two_sided(self, mean, sigma_within, sigma_overall):
        cp = (self.usl - self.lsl) / (6 * sigma_within)
        cpu = (self.usl - mean) / (3 * sigma_within)
        cpl = (mean - self.lsl) / (3 * sigma_within)
        cpk = min(cpu, cpl)

        pp = (self.usl - self.lsl) / (6 * sigma_overall)
        ppu = (self.usl - mean) / (3 * sigma_overall)
        ppl = (mean - self.lsl) / (3 * sigma_overall)
        ppk = min(ppu, ppl)

        self._results.update({
            "cp": cp, "cpu": cpu, "cpl": cpl, "cpk": cpk,
            "pp": pp, "ppu": ppu, "ppl": ppl, "ppk": ppk,
            "min_index": "USL" if cpu < cpl else "LSL",
        })

    def _compute_upper_only(self, mean, sigma_within, sigma_overall):
        cpu = (self.usl - mean) / (3 * sigma_within)
        ppu = (self.usl - mean) / (3 * sigma_overall)

        self._results.update({
            "cpu": cpu, "cpk": cpu,
            "ppu": ppu, "ppk": ppu,
        })

    def _compute_lower_only(self, mean, sigma_within, sigma_overall):
        cpl = (mean - self.lsl) / (3 * sigma_within)
        ppl = (mean - self.lsl) / (3 * sigma_overall)

        self._results.update({
            "cpl": cpl, "cpk": cpl,
            "ppl": ppl, "ppk": ppl,
        })

    def _compute_cpm(self, mean, sigma_within):
        if self.target is None:
            return
        diff = mean - self.target
        sigma_prime = np.sqrt(sigma_within**2 + diff**2)
        cpm = (self.usl - self.lsl) / (6 * sigma_prime) if (self.usl is not None and self.lsl is not None) else None

        if self.usl is not None and self.lsl is not None:
            self._results["cpm"] = cpm
        elif self.usl is not None:
            self._results["cpm"] = (self.usl - self.target) / (3 * sigma_prime)
        elif self.lsl is not None:
            self._results["cpm"] = (self.target - self.lsl) / (3 * sigma_prime)

    def _compute_confidence_intervals(self, sigma_within, sigma_overall, n):
        alpha = 1 - self.confidence_level
        z = stats.norm.ppf(1 - alpha / 2)

        if "cpk" in self._results:
            cpk = self._results["cpk"]
            if cpk is not None and np.isfinite(cpk):
                se_cpk = np.sqrt(cpk**2 / (2 * (n - 1)) + 1 / (9 * n))
                self._results["cpk_ci_lower"] = max(0, cpk - z * se_cpk)
                self._results["cpk_ci_upper"] = cpk + z * se_cpk

        if "ppk" in self._results:
            ppk = self._results["ppk"]
            if ppk is not None and np.isfinite(ppk):
                se_ppk = np.sqrt(ppk**2 / (2 * (n - 1)) + 1 / (9 * n))
                self._results["ppk_ci_lower"] = max(0, ppk - z * se_ppk)
                self._results["ppk_ci_upper"] = ppk + z * se_ppk

        if "cp" in self._results:
            cp = self._results["cp"]
            if cp is not None and cp > 0 and np.isfinite(cp):
                se_cp = cp * np.sqrt(1 / (2 * (n - 1)))
                denom_lo = 1 + z * se_cp / cp
                denom_hi = 1 - z * se_cp / cp
                if denom_lo > 0:
                    self._results["cp_ci_lower"] = cp / denom_lo
                if denom_hi > 0:
                    self._results["cp_ci_upper"] = cp / denom_hi

    def _compute_ppm(self, mean, sigma_within, sigma_overall):
        if self.usl is not None:
            z_usl_within = (self.usl - mean) / sigma_within
            z_usl_overall = (self.usl - mean) / sigma_overall
            self._results["ppm_above_usl_within"] = (1 - stats.norm.cdf(z_usl_within)) * 1e6
            self._results["ppm_above_usl_overall"] = (1 - stats.norm.cdf(z_usl_overall)) * 1e6

        if self.lsl is not None:
            z_lsl_within = (self.lsl - mean) / sigma_within
            z_lsl_overall = (self.lsl - mean) / sigma_overall
            self._results["ppm_below_lsl_within"] = stats.norm.cdf(z_lsl_within) * 1e6
            self._results["ppm_below_lsl_overall"] = stats.norm.cdf(z_lsl_overall) * 1e6

        if self.usl is not None and self.lsl is not None:
            self._results["ppm_total_within"] = (
                self._results["ppm_below_lsl_within"] + self._results["ppm_above_usl_within"]
            )
            self._results["ppm_total_overall"] = (
                self._results["ppm_below_lsl_overall"] + self._results["ppm_above_usl_overall"]
            )

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        r = self._results
        lines = []
        lines.append("=" * 60)
        lines.append("ANALYSE DE CAPABILITÉ")
        lines.append("=" * 60)
        lines.append(f"\nStatistiques descriptives:")
        lines.append(f"  N = {r['n']}")
        lines.append(f"  Moyenne = {r['mean']:.6f}")
        lines.append(f"  Médiane = {r['median']:.6f}")
        lines.append(f"  Écart-type (intra) = {r['std_within']:.6f}")
        lines.append(f"  Écart-type (global) = {r['std_overall']:.6f}")
        lines.append(f"  Min = {r['min']:.6f}")
        lines.append(f"  Max = {r['max']:.6f}")

        lines.append(f"\nLimites:")
        if r.get("lsl") is not None:
            lines.append(f"  LSL = {r['lsl']:.6f}")
        if r.get("usl") is not None:
            lines.append(f"  USL = {r['usl']:.6f}")
        if r.get("target") is not None:
            lines.append(f"  Cible = {r['target']:.6f}")

        lines.append(f"\nIndices de capabilité:")
        if "cp" in r:
            lines.append(f"  Cp  = {r['cp']:.4f}")
            lines.append(f"  CPL = {r['cpl']:.4f}")
            lines.append(f"  CPU = {r['cpu']:.4f}")
            lines.append(f"  Cpk = {r['cpk']:.4f}")
            if "cpm" in r:
                lines.append(f"  Cpm = {r['cpm']:.4f}")
            lines.append(f"\n  IC Cpk ({self.confidence_level*100:.0f}%): [{r['cpk_ci_lower']:.4f}, {r['cpk_ci_upper']:.4f}]")

        if "pp" in r:
            lines.append(f"\n  Pp  = {r['pp']:.4f}")
            lines.append(f"  PPL = {r['ppl']:.4f}")
            lines.append(f"  PPU = {r['ppu']:.4f}")
            lines.append(f"  Ppk = {r['ppk']:.4f}")
            lines.append(f"  IC Ppk ({self.confidence_level*100:.0f}%): [{r['ppk_ci_lower']:.4f}, {r['ppk_ci_upper']:.4f}]")

        if "ppm_total_within" in r:
            lines.append(f"\nPPM (pièces par million):")
            lines.append(f"  < LSL (intra):  {r['ppm_below_lsl_within']:.2f}")
            lines.append(f"  > USL (intra):  {r['ppm_above_usl_within']:.2f}")
            lines.append(f"  Total (intra):  {r['ppm_total_within']:.2f}")
            lines.append(f"  < LSL (global): {r['ppm_below_lsl_overall']:.2f}")
            lines.append(f"  > USL (global): {r['ppm_above_usl_overall']:.2f}")
            lines.append(f"  Total (global): {r['ppm_total_overall']:.2f}")
        elif "ppm_above_usl_within" in r:
            lines.append(f"\nPPM (pièces par million):")
            lines.append(f"  > USL (intra):  {r['ppm_above_usl_within']:.2f}")
            lines.append(f"  > USL (global): {r['ppm_above_usl_overall']:.2f}")
        elif "ppm_below_lsl_within" in r:
            lines.append(f"\nPPM (pièces par million):")
            lines.append(f"  < LSL (intra):  {r['ppm_below_lsl_within']:.2f}")
            lines.append(f"  < LSL (global): {r['ppm_below_lsl_overall']:.2f}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
