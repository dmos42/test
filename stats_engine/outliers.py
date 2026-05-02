import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional


class OutlierDetection:
    """Tests de détection de valeurs aberrantes:
    Grubbs, Dixon, IQR, Z-score, Modified Z-score, Chauvenet, Peirce, Tukey."""

    def __init__(self, data: np.ndarray, alpha: float = 0.05):
        self.data = np.array(data, dtype=float)
        self.data = self.data[~np.isnan(self.data)]
        self.alpha = alpha
        self._results: Dict[str, Any] = {}
        self._run_all_tests()

    def _run_all_tests(self):
        n = len(self.data)
        self._results = {
            "n": n,
            "mean": np.mean(self.data),
            "std": np.std(self.data, ddof=1),
            "median": np.median(self.data),
            "q1": np.percentile(self.data, 25),
            "q3": np.percentile(self.data, 75),
            "iqr": np.percentile(self.data, 75) - np.percentile(self.data, 25),
            "min": np.min(self.data),
            "max": np.max(self.data),
            "alpha": self.alpha,
            "tests": {},
        }

        self._iqr_method()
        self._z_score_method()
        self._modified_z_score_method()

        if n >= 3:
            self._grubbs_test()

        if 3 <= n <= 30:
            self._dixon_test()

        self._chauvenet_criterion()

        if n >= 3:
            self._tukey_fences()

    def _iqr_method(self):
        q1, q3 = np.percentile(self.data, 25), np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = self.data[(self.data < lower_bound) | (self.data > upper_bound)]
        outlier_indices = np.where((self.data < lower_bound) | (self.data > upper_bound))[0].tolist()

        self._results["tests"]["IQR (1.5×IQR)"] = {
            "method": "IQR",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_values": outliers.tolist(),
            "outlier_indices": outlier_indices,
            "outlier_count": len(outliers),
            "description": "Méthode de l'écart interquartile (boîte à moustaches)",
        }

    def _z_score_method(self):
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        if std == 0:
            return

        z_scores = np.abs((self.data - mean) / std)
        threshold = stats.norm.ppf(1 - self.alpha / 2)

        outlier_mask = z_scores > threshold
        outliers = self.data[outlier_mask]
        outlier_indices = np.where(outlier_mask)[0].tolist()

        self._results["tests"]["Z-score"] = {
            "method": "Z-score",
            "threshold": threshold,
            "outlier_values": outliers.tolist(),
            "outlier_indices": outlier_indices,
            "outlier_count": len(outliers),
            "z_scores": z_scores.tolist(),
            "description": f"Valeurs avec |Z| > {threshold:.3f}",
        }

    def _modified_z_score_method(self):
        median = np.median(self.data)
        mad = np.median(np.abs(self.data - median))
        if mad == 0:
            return

        modified_z_scores = 0.6745 * (self.data - median) / mad
        threshold = 3.5

        outlier_mask = np.abs(modified_z_scores) > threshold
        outliers = self.data[outlier_mask]
        outlier_indices = np.where(outlier_mask)[0].tolist()

        self._results["tests"]["Z-score modifié (MAD)"] = {
            "method": "Modified Z-score",
            "threshold": threshold,
            "median": median,
            "mad": mad,
            "outlier_values": outliers.tolist(),
            "outlier_indices": outlier_indices,
            "outlier_count": len(outliers),
            "description": "Z-score modifié basé sur la médiane et MAD (robuste)",
        }

    def _grubbs_test(self):
        n = len(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        if std == 0:
            return

        sorted_indices = np.argsort(self.data)
        min_val = self.data[sorted_indices[0]]
        max_val = self.data[sorted_indices[-1]]

        g_min = (mean - min_val) / std
        g_max = (max_val - mean) / std

        t_crit = stats.t.ppf(1 - self.alpha / (2 * n), n - 2)
        g_crit = ((n - 1) * t_crit) / np.sqrt(n * (n - 2 + t_crit**2))

        outliers = []
        outlier_indices = []

        if g_max > g_crit:
            outliers.append(max_val)
            outlier_indices.append(sorted_indices[-1])
        if g_min > g_crit:
            outliers.append(min_val)
            outlier_indices.append(sorted_indices[0])

        self._results["tests"]["Grubbs"] = {
            "method": "Grubbs",
            "g_min": g_min,
            "g_max": g_max,
            "critical_value": g_crit,
            "outlier_values": outliers,
            "outlier_indices": outlier_indices,
            "outlier_count": len(outliers),
            "description": "Test de Grubbs (détection d'une valeur aberrante)",
        }

    def _dixon_test(self):
        n = len(self.data)
        sorted_data = np.sort(self.data)
        sorted_indices = np.argsort(self.data)

        range_val = sorted_data[-1] - sorted_data[0]
        if range_val == 0:
            return

        q_values = self._get_dixon_q_values(n)
        critical_values = self._get_dixon_critical_values(n, self.alpha)

        results = []

        if n in q_values:
            q_formula = q_values[n]
            q_low, q_high = self._compute_dixon_q(sorted_data, n)

            q_crit = critical_values.get(n, 0.5)

            outliers = []
            outlier_indices = []

            if q_high > q_crit:
                outliers.append(float(sorted_data[-1]))
                outlier_indices.append(int(sorted_indices[-1]))
            if q_low > q_crit:
                outliers.append(float(sorted_data[0]))
                outlier_indices.append(int(sorted_indices[0]))

            self._results["tests"]["Dixon"] = {
                "method": "Dixon",
                "q_low": q_low,
                "q_high": q_high,
                "critical_value": q_crit,
                "outlier_values": outliers,
                "outlier_indices": outlier_indices,
                "outlier_count": len(outliers),
                "description": "Test de Dixon (petits échantillons, 3 ≤ n ≤ 30)",
            }

    def _get_dixon_q_values(self, n) -> Dict[int, str]:
        return {
            3: "r10", 4: "r10", 5: "r10",
            6: "r10", 7: "r10", 8: "r10", 9: "r10", 10: "r10",
            11: "r11", 12: "r11", 13: "r11", 14: "r11", 15: "r11",
            16: "r11", 17: "r11", 18: "r11", 19: "r11", 20: "r11",
            21: "r21", 22: "r21", 23: "r21", 24: "r21", 25: "r21",
            26: "r21", 27: "r21", 28: "r21", 29: "r21", 30: "r21",
        }

    def _compute_dixon_q(self, sorted_data, n):
        range_val = sorted_data[-1] - sorted_data[0]
        if range_val == 0:
            return 0.0, 0.0

        if n <= 7:
            q_low = (sorted_data[1] - sorted_data[0]) / range_val
            q_high = (sorted_data[-1] - sorted_data[-2]) / range_val
        elif n <= 10:
            q_low = (sorted_data[1] - sorted_data[0]) / range_val
            q_high = (sorted_data[-1] - sorted_data[-2]) / range_val
        elif n <= 13:
            range_high = sorted_data[-1] - sorted_data[1]
            range_low = sorted_data[-2] - sorted_data[0]
            if range_high != 0:
                q_high = (sorted_data[-1] - sorted_data[-2]) / range_high
            else:
                q_high = 0
            if range_low != 0:
                q_low = (sorted_data[1] - sorted_data[0]) / range_low
            else:
                q_low = 0
        else:
            range_high = sorted_data[-1] - sorted_data[2]
            range_low = sorted_data[-3] - sorted_data[0]
            if range_high != 0:
                q_high = (sorted_data[-1] - sorted_data[-3]) / range_high
            else:
                q_high = 0
            if range_low != 0:
                q_low = (sorted_data[2] - sorted_data[0]) / range_low
            else:
                q_low = 0

        return q_low, q_high

    def _get_dixon_critical_values(self, n, alpha) -> Dict[int, float]:
        critical_05 = {
            3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568,
            8: 0.526, 9: 0.493, 10: 0.466, 11: 0.517, 12: 0.490,
            13: 0.467, 14: 0.446, 15: 0.428, 16: 0.412, 17: 0.397,
            18: 0.384, 19: 0.372, 20: 0.361, 21: 0.386, 22: 0.376,
            23: 0.367, 24: 0.358, 25: 0.350, 26: 0.343, 27: 0.337,
            28: 0.331, 29: 0.325, 30: 0.320,
        }
        return critical_05

    def _chauvenet_criterion(self):
        n = len(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        if std == 0:
            return

        z_scores = np.abs((self.data - mean) / std)
        threshold = 0.5 + stats.norm.ppf(1 - 1 / (4 * n))

        outlier_mask = z_scores > threshold
        outliers = self.data[outlier_mask]
        outlier_indices = np.where(outlier_mask)[0].tolist()

        self._results["tests"]["Chauvenet"] = {
            "method": "Chauvenet",
            "threshold": threshold,
            "outlier_values": outliers.tolist(),
            "outlier_indices": outlier_indices,
            "outlier_count": len(outliers),
            "description": "Critère de Chauvenet (probabilité < 1/(2n))",
        }

    def _tukey_fences(self):
        q1, q3 = np.percentile(self.data, 25), np.percentile(self.data, 75)
        iqr = q3 - q1

        lower_fence = q1 - 3 * iqr
        upper_fence = q3 + 3 * iqr

        outliers = self.data[(self.data < lower_fence) | (self.data > upper_fence)]
        outlier_indices = np.where((self.data < lower_fence) | (self.data > upper_fence))[0].tolist()

        self._results["tests"]["Tukey (3×IQR, extrêmes)"] = {
            "method": "Tukey fences",
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outlier_values": outliers.tolist(),
            "outlier_indices": outlier_indices,
            "outlier_count": len(outliers),
            "description": "Valeurs aberrantes extrêmes (3×IQR, critère strict)",
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("DÉTECTION DE VALEURS ABERRANTES")
        lines.append("=" * 60)
        lines.append(f"\nStatistiques descriptives:")
        lines.append(f"  N = {self._results['n']}")
        lines.append(f"  Moyenne = {self._results['mean']:.6f}")
        lines.append(f"  Médiane = {self._results['median']:.6f}")
        lines.append(f"  Écart-type = {self._results['std']:.6f}")
        lines.append(f"  Q1 = {self._results['q1']:.6f}")
        lines.append(f"  Q3 = {self._results['q3']:.6f}")
        lines.append(f"  IQR = {self._results['iqr']:.6f}")
        lines.append(f"  Min = {self._results['min']:.6f}")
        lines.append(f"  Max = {self._results['max']:.6f}")
        lines.append(f"\nSeuil de significativité (α) = {self._results['alpha']}")
        lines.append("\n" + "-" * 60)

        for name, result in self._results["tests"].items():
            lines.append(f"\n{name}:")
            lines.append(f"  {result['description']}")
            lines.append(f"  Valeurs détectées: {result['outlier_count']}")

            if result["outlier_count"] > 0:
                for i, (val, idx) in enumerate(zip(result["outlier_values"], result["outlier_indices"])):
                    lines.append(f"    [{idx}] Valeur = {val:.6f}")
            else:
                lines.append(f"    Aucune valeur aberrante détectée")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def get_outlier_indices(self) -> set:
        all_indices = set()
        for test_result in self._results["tests"].values():
            all_indices.update(test_result["outlier_indices"])
        return all_indices
