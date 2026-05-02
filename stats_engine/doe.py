import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
from itertools import product


class FullFactorialDOE:
    """Plan d'expériences factoriel complet"""

    def __init__(self, factors: Dict[str, List[float]], responses: np.ndarray = None):
        """
        factors: dict {nom_facteur: [niveau_bas, niveau_haut]} ou {nom_facteur: [liste_niveaux]}
        responses: array de réponses observées (optionnel)
        """
        self.factors = factors
        self.factor_names = list(factors.keys())
        self.n_factors = len(factors)
        self.responses = np.array(responses, dtype=float) if responses is not None else None
        self._design_matrix = None
        self._results: Dict[str, Any] = {}
        self._generate_design()
        if self.responses is not None:
            self._analyze()

    def _generate_design(self):
        """Génère la matrice du plan d'expériences"""
        levels = [self.factors[name] for name in self.factor_names]
        self._design_matrix = np.array(list(product(*levels)))

    def get_design_matrix(self) -> np.ndarray:
        return self._design_matrix

    def set_responses(self, responses: np.ndarray):
        self.responses = np.array(responses, dtype=float)
        self._analyze()

    def _analyze(self):
        if self.responses is None:
            return

        X = self._design_matrix
        y = self.responses

        # Ajouter l'intercept
        X_design = np.column_stack([np.ones(len(X)), X])

        # Modèle linéaire: y = b0 + b1*x1 + b2*x2 + ...
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
            y_pred = X_design @ coeffs

            # Statistiques
            n = len(y)
            p = len(coeffs)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)

            # Erreur standard des coefficients
            if n > p:
                mse = ss_res / (n - p)
                var_coeffs = mse * np.linalg.inv(X_design.T @ X_design).diagonal()
                se_coeffs = np.sqrt(var_coeffs)
                t_values = coeffs / se_coeffs
                p_values = [2 * (1 - stats.t.cdf(abs(t), n - p)) for t in t_values]
            else:
                se_coeffs = np.zeros(p)
                t_values = np.zeros(p)
                p_values = np.ones(p)

            # ANOVA du modèle
            ss_model = ss_tot - ss_res
            df_model = p - 1
            df_error = n - p
            ms_model = ss_model / df_model if df_model > 0 else 0
            ms_error = ss_res / df_error if df_error > 0 else 0
            f_model = ms_model / ms_error if ms_error > 0 else float('inf')
            p_model = 1 - stats.f.cdf(f_model, df_model, df_error)

            # Effets principaux (codés -1, +1)
            effects = {}
            for i, name in enumerate(self.factor_names):
                col = X[:, i]
                unique_vals = np.unique(col)
                if len(unique_vals) == 2:
                    # Effet = (moyenne haute - moyenne basse)
                    high_mask = col == unique_vals[1]
                    low_mask = col == unique_vals[0]
                    effect = np.mean(y[high_mask]) - np.mean(y[low_mask])
                    effects[name] = {
                        "effect": effect,
                        "coefficient": effect / 2,  # coefficient du modèle codé
                        "low_mean": np.mean(y[low_mask]),
                        "high_mean": np.mean(y[high_mask]),
                    }

            self._results = {
                "coefficients": coeffs.tolist(),
                "se_coefficients": se_coeffs.tolist(),
                "t_values": t_values.tolist(),
                "p_values": p_values,
                "r_squared": r_squared,
                "adj_r_squared": adj_r_squared,
                "ss_res": ss_res,
                "ss_tot": ss_tot,
                "ss_model": ss_model,
                "df_model": df_model,
                "df_error": df_error,
                "ms_model": ms_model,
                "ms_error": ms_error,
                "f_model": f_model,
                "p_model": p_model,
                "effects": effects,
                "n": n,
                "n_predictors": self.n_factors,
                "y_pred": y_pred.tolist(),
                "residuals": (y - y_pred).tolist(),
            }
        except Exception as e:
            self._results = {"error": str(e)}

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        if not self._results or "error" in self._results:
            return "Pas d'analyse disponible. Fournissez des réponses."

        r = self._results
        lines = ["=" * 60, "PLAN D'EXPÉRIENCES FACTORIEL COMPLET", "=" * 60]
        lines.append(f"\nNombre de facteurs : {r['n_predictors']}")
        lines.append(f"Nombre d'expériences : {r['n']}")
        lines.append(f"Facteurs : {', '.join(self.factor_names)}")

        lines.append(f"\n--- Modèle ---")
        terms = [f"{r['coefficients'][0]:.4f}"]
        for i, name in enumerate(self.factor_names):
            idx = i + 1
            terms.append(f"{r['coefficients'][idx]:+.4f}*{name}")
        lines.append(f"y = {' '.join(terms)}")

        lines.append(f"\n--- Qualité du modèle ---")
        lines.append(f"  R² = {r['r_squared']:.4f}")
        lines.append(f"  R² ajusté = {r['adj_r_squared']:.4f}")
        lines.append(f"  F = {r['f_model']:.4f} (p = {r['p_model']:.6f})")
        if r['p_model'] < 0.05:
            lines.append(f"  ✓ Modèle significatif")
        else:
            lines.append(f"  ⚠ Modèle non significatif")

        lines.append(f"\n--- Effets principaux ---")
        for name, eff in r['effects'].items():
            lines.append(f"  {name}: effet = {eff['effect']:.4f} (coef = {eff['coefficient']:.4f})")
            lines.append(f"    Moy. basse = {eff['low_mean']:.4f}, Moy. haute = {eff['high_mean']:.4f}")

        lines.append(f"\n--- Analyse des coefficients ---")
        lines.append(f"  {'Terme':<15s} | {'Coef':>10s} | {'SE':>10s} | {'t':>10s} | {'p':>10s}")
        lines.append(f"  {'-'*60}")
        for i, name in enumerate(["Intercept"] + self.factor_names):
            lines.append(f"  {name:<15s} | {r['coefficients'][i]:10.4f} | {r['se_coefficients'][i]:10.4f} | {r['t_values'][i]:10.4f} | {r['p_values'][i]:10.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class TaguchiOA:
    """Plan de Taguchi (tableau orthogonal)"""

    def __init__(self, n_factors: int, n_levels: int = 2):
        self.n_factors = n_factors
        self.n_levels = n_levels
        self._design_matrix = None
        self._generate()

    def _generate(self):
        """Génère un tableau orthogonal L4, L8, L9, L16, L27 selon le cas"""
        if self.n_levels == 2:
            if self.n_factors <= 3:
                self._design_matrix = self._l4()
            elif self.n_factors <= 7:
                self._design_matrix = self._l8()
            elif self.n_factors <= 15:
                self._design_matrix = self._l16()
            else:
                self._design_matrix = self._full_factorial_2k()
        elif self.n_levels == 3:
            if self.n_factors <= 4:
                self._design_matrix = self._l9()
            elif self.n_factors <= 13:
                self._design_matrix = self._l27()
            else:
                self._design_matrix = self._full_factorial_3k()
        else:
            self._design_matrix = self._full_factorial()

    def _l4(self):
        return np.array([
            [-1, -1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1],
        ])[:self.n_factors]

    def _l8(self):
        return np.array([
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1],
            [-1, 1, 1, -1, -1, 1, 1],
            [-1, 1, 1, 1, 1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1],
            [1, -1, 1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1],
            [1, 1, -1, 1, -1, -1, 1],
        ])[:self.n_factors]

    def _l9(self):
        return np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 2, 2, 2],
            [1, 0, 1, 2],
            [1, 1, 2, 0],
            [1, 2, 0, 1],
            [2, 0, 2, 1],
            [2, 1, 0, 2],
            [2, 2, 1, 0],
        ])[:self.n_factors]

    def _l16(self):
        from itertools import product
        return np.array(list(product([-1, 1], repeat=4)))

    def _l27(self):
        from itertools import product
        return np.array(list(product([0, 1, 2], repeat=3)))

    def _full_factorial_2k(self):
        from itertools import product
        return np.array(list(product([-1, 1], repeat=self.n_factors)))

    def _full_factorial_3k(self):
        from itertools import product
        return np.array(list(product([0, 1, 2], repeat=self.n_factors)))

    def _full_factorial(self):
        from itertools import product
        levels = list(range(self.n_levels))
        return np.array(list(product(levels, repeat=self.n_factors)))

    def get_design_matrix(self) -> np.ndarray:
        return self._design_matrix

    def get_summary(self) -> str:
        lines = ["=" * 60, "TABLEAU ORTHOGONAL DE TAGUCHI", "=" * 60]
        lines.append(f"\nNombre de facteurs : {self.n_factors}")
        lines.append(f"Nombre de niveaux : {self.n_levels}")
        lines.append(f"Nombre d'expériences : {self._design_matrix.shape[0]}")
        lines.append(f"\nMatrice du plan :")
        for i, row in enumerate(self._design_matrix):
            lines.append(f"  Exp {i+1:2d}: {row}")
        lines.append("=" * 60)
        return "\n".join(lines)
