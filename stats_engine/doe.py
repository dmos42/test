
"""DOE / Plans d'expériences pour StatPro.

Objectif : fournir une base proche des usages Minitab pour :
- génération de plans factoriels complets 2 niveaux et multi-niveaux ;
- plans de Taguchi L4/L8/L9/L16/L27 ;
- plans de surface de réponse : CCD et Box-Behnken ;
- analyse DOE par modèle linéaire avec effets principaux, interactions et termes quadratiques ;
- rapports texte clairs, ANOVA modèle, coefficients, effets standardisés, diagnostics.
"""
from __future__ import annotations

from itertools import product, combinations
from typing import Dict, Any, List, Optional, Sequence, Tuple
import math
import numpy as np
import pandas as pd
from scipy import stats


def _as_finite_vector(values, name="réponse"):
    arr = np.asarray(values, dtype=float).reshape(-1)
    mask = np.isfinite(arr)
    return arr[mask], int(np.sum(~mask))


def _fmt(value, digits=4):
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _fmt_p(value):
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        value = float(value)
        return "<0.000001" if value < 1e-6 else f"{value:.6f}"
    except Exception:
        return "n/a"


def _safe_colname(name):
    return str(name).strip() or "Facteur"


class DOEAnalyzer:
    """Analyse générique de plan d'expériences.

    Parameters
    ----------
    design:
        Matrice des facteurs, une ligne par essai.
    response:
        Réponse numérique, alignée sur les lignes du design.
    factor_names:
        Noms des facteurs.
    model:
        "main", "2fi" ou "quadratic".
    alpha:
        Seuil de significativité.
    """

    VALID_MODELS = {"main", "2fi", "quadratic"}

    def __init__(self, design, response, factor_names=None, model="2fi", alpha=0.05, coded=False):
        self.design = np.asarray(design, dtype=float)
        if self.design.ndim == 1:
            self.design = self.design.reshape(-1, 1)
        self.response = np.asarray(response, dtype=float).reshape(-1)
        self.factor_names = factor_names or [f"X{i+1}" for i in range(self.design.shape[1])]
        self.factor_names = [_safe_colname(x) for x in self.factor_names]
        self.model = str(model).lower()
        if self.model not in self.VALID_MODELS:
            self.model = "2fi"
        self.alpha = float(alpha) if 0 < float(alpha) < 1 else 0.05
        self.coded = bool(coded)
        self._results: Dict[str, Any] = {}
        self._compute()

    def _invalid(self, errors, warnings=None):
        self._results = {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings or [],
            "model": self.model,
            "factor_names": self.factor_names,
            "n": int(min(len(self.response), len(self.design))),
        }

    def _build_terms(self, X):
        n, k = X.shape
        columns = [np.ones(n)]
        terms = ["Constante"]
        term_type = ["intercept"]
        # effets principaux
        for i, name in enumerate(self.factor_names):
            columns.append(X[:, i])
            terms.append(name)
            term_type.append("main")
        # interactions 2 facteurs
        if self.model in {"2fi", "quadratic"}:
            for i, j in combinations(range(k), 2):
                columns.append(X[:, i] * X[:, j])
                terms.append(f"{self.factor_names[i]}*{self.factor_names[j]}")
                term_type.append("interaction")
        # quadratiques
        if self.model == "quadratic":
            for i, name in enumerate(self.factor_names):
                columns.append(X[:, i] ** 2)
                terms.append(f"{name}²")
                term_type.append("quadratic")
        return np.column_stack(columns), terms, term_type

    def _compute(self):
        errors, warnings = [], []
        if self.design.ndim != 2 or self.design.shape[0] < 2:
            errors.append("Le plan doit contenir au moins 2 essais.")
        if self.design.shape[1] < 1:
            errors.append("Le plan doit contenir au moins 1 facteur.")
        if len(self.factor_names) != self.design.shape[1]:
            warnings.append("Nombre de noms de facteurs différent du nombre de colonnes ; noms automatiques utilisés.")
            self.factor_names = [f"X{i+1}" for i in range(self.design.shape[1])]
        if len(self.response) != self.design.shape[0]:
            errors.append(f"Nombre de réponses incohérent : attendu {self.design.shape[0]}, reçu {len(self.response)}.")
        finite_rows = np.all(np.isfinite(self.design), axis=1) & np.isfinite(self.response[:self.design.shape[0]]) if len(self.response) >= self.design.shape[0] else np.array([], dtype=bool)
        if len(finite_rows) and not np.all(finite_rows):
            warnings.append(f"{int(np.sum(~finite_rows))} ligne(s) avec NaN/Inf exclue(s).")
        if errors:
            self._invalid(errors, warnings)
            return
        X_raw = self.design[finite_rows]
        y = self.response[:self.design.shape[0]][finite_rows]
        n = len(y)
        if n < 3:
            self._invalid(["Au moins 3 essais valides sont nécessaires pour l'analyse DOE."], warnings)
            return
        if np.std(y, ddof=1) <= 0:
            self._invalid(["La réponse est constante : modèle DOE non informatif."], warnings)
            return
        X_model, terms, term_type = self._build_terms(X_raw)
        p = X_model.shape[1]
        if n <= 1:
            self._invalid(["Nombre d'essais insuffisant."], warnings)
            return
        coeffs, residuals_lstsq, rank, singular_values = np.linalg.lstsq(X_model, y, rcond=None)
        y_pred = X_model @ coeffs
        resid = y - y_pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_model = max(0.0, ss_tot - ss_res)
        df_model = max(rank - 1, 0)
        df_error = max(n - rank, 0)
        df_total = n - 1
        ms_model = ss_model / df_model if df_model > 0 else np.nan
        ms_error = ss_res / df_error if df_error > 0 else np.nan
        f_model = ms_model / ms_error if df_model > 0 and df_error > 0 and ms_error > 0 else np.nan
        p_model = float(stats.f.sf(f_model, df_model, df_error)) if np.isfinite(f_model) else None
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        adj_r2 = 1 - (1 - r2) * (n - 1) / df_error if df_error > 0 and np.isfinite(r2) else np.nan
        pred_r2 = np.nan
        # PRESS approximé par la diagonale du hat matrix
        try:
            xtx_inv = np.linalg.pinv(X_model.T @ X_model)
            hat = np.sum((X_model @ xtx_inv) * X_model, axis=1)
            press = float(np.sum((resid / np.clip(1 - hat, 1e-12, None)) ** 2))
            pred_r2 = 1 - press / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            press = None
        if df_error > 0 and np.isfinite(ms_error):
            try:
                cov = ms_error * np.linalg.pinv(X_model.T @ X_model)
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                t_values = np.divide(coeffs, se, out=np.zeros_like(coeffs), where=se > 0)
                p_values = np.array([2 * stats.t.sf(abs(t), df_error) if np.isfinite(t) else np.nan for t in t_values])
            except Exception:
                se = np.full_like(coeffs, np.nan)
                t_values = np.full_like(coeffs, np.nan)
                p_values = np.full_like(coeffs, np.nan)
        else:
            se = np.full_like(coeffs, np.nan)
            t_values = np.full_like(coeffs, np.nan)
            p_values = np.full_like(coeffs, np.nan)
        # Effets standardisés hors intercept
        effects = []
        for idx in range(1, len(terms)):
            effects.append({
                "term": terms[idx],
                "type": term_type[idx],
                "coefficient": float(coeffs[idx]),
                "effect_estimate": float(2 * coeffs[idx]) if term_type[idx] in ("main", "interaction") else float(coeffs[idx]),
                "se": float(se[idx]) if np.isfinite(se[idx]) else None,
                "t": float(t_values[idx]) if np.isfinite(t_values[idx]) else None,
                "p": float(p_values[idx]) if np.isfinite(p_values[idx]) else None,
                "significant": bool(np.isfinite(p_values[idx]) and p_values[idx] < self.alpha),
                "abs_standardized_effect": float(abs(t_values[idx])) if np.isfinite(t_values[idx]) else None,
            })
        effects_sorted = sorted(effects, key=lambda d: -1 if d["abs_standardized_effect"] is None else -d["abs_standardized_effect"])
        coeff_table = []
        for i, term in enumerate(terms):
            coeff_table.append({
                "term": term,
                "type": term_type[i],
                "coef": float(coeffs[i]),
                "se": float(se[i]) if np.isfinite(se[i]) else None,
                "t": float(t_values[i]) if np.isfinite(t_values[i]) else None,
                "p": float(p_values[i]) if np.isfinite(p_values[i]) else None,
            })
        self._results = {
            "is_valid": True,
            "errors": [],
            "warnings": warnings,
            "model": self.model,
            "n": int(n),
            "n_factors": int(X_raw.shape[1]),
            "factor_names": self.factor_names,
            "terms": terms,
            "term_type": term_type,
            "coefficients": coeffs.tolist(),
            "coefficient_table": coeff_table,
            "effects": effects,
            "effects_sorted": effects_sorted,
            "rank": int(rank),
            "df_model": int(df_model),
            "df_error": int(df_error),
            "df_total": int(df_total),
            "ss_model": ss_model,
            "ss_error": ss_res,
            "ss_total": ss_tot,
            "ms_model": ms_model,
            "ms_error": ms_error,
            "f_model": f_model,
            "p_model": p_model,
            "r_squared": float(r2),
            "adj_r_squared": float(adj_r2) if np.isfinite(adj_r2) else None,
            "pred_r_squared": float(pred_r2) if np.isfinite(pred_r2) else None,
            "s": float(np.sqrt(ms_error)) if np.isfinite(ms_error) and ms_error >= 0 else None,
            "y_pred": y_pred.tolist(),
            "residuals": resid.tolist(),
            "response": y.tolist(),
            "design": X_raw.tolist(),
            "model_matrix_rank": int(rank),
            "singular_values": singular_values.tolist(),
        }

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def predict(self, design):
        r = self._results
        if not r.get("is_valid"):
            raise ValueError("Modèle non valide")
        X = np.asarray(design, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_model, _, _ = self._build_terms(X)
        return X_model @ np.asarray(r["coefficients"], dtype=float)

    def get_summary(self) -> str:
        r = self._results
        if not r.get("is_valid", False):
            lines = ["="*72, "PLAN D'EXPÉRIENCES - ANALYSE NON VALIDE", "="*72]
            lines.extend([f"- {e}" for e in r.get("errors", [])])
            return "\n".join(lines)
        lines = ["="*72, "PLAN D'EXPÉRIENCES - ANALYSE DU MODÈLE", "="*72]
        if r.get("warnings"):
            lines.append("\nAvertissements :")
            lines.extend([f"- {w}" for w in r["warnings"]])
        lines += [
            "\nPlan d'analyse :",
            f"Nombre d'essais = {r['n']}",
            f"Nombre de facteurs = {r['n_factors']}",
            f"Modèle = {r['model']} (main = effets principaux, 2fi = interactions, quadratic = courbure)",
            f"Facteurs = {', '.join(r['factor_names'])}",
            f"Seuil alpha = {self.alpha:.4f}",
            "\n" + "="*72,
            "SYNTHÈSE DU MODÈLE",
            "="*72,
            f"\nS = {_fmt(r.get('s'), 4)}",
            f"R² = {_fmt(100*r['r_squared'], 2)} %",
            f"R² ajusté = {_fmt(None if r.get('adj_r_squared') is None else 100*r['adj_r_squared'], 2)} %",
            f"R² prédit = {_fmt(None if r.get('pred_r_squared') is None else 100*r['pred_r_squared'], 2)} %",
            f"F modèle = {_fmt(r.get('f_model'), 4)} ; p-value = {_fmt_p(r.get('p_model'))}",
            "Conclusion globale = " + ("modèle significatif" if r.get("p_model") is not None and r["p_model"] < self.alpha else "modèle non significatif ou non testable"),
            "\n" + "="*72,
            "ANOVA DU MODÈLE",
            "="*72,
            f"{'Source':<22}{'ddl':>7}{'SS':>14}{'MS':>14}{'F':>12}{'p-value':>14}",
            "-"*83,
            f"{'Modèle':<22}{r['df_model']:>7}{_fmt(r['ss_model'],4):>14}{_fmt(r['ms_model'],4):>14}{_fmt(r['f_model'],4):>12}{_fmt_p(r['p_model']):>14}",
            f"{'Erreur':<22}{r['df_error']:>7}{_fmt(r['ss_error'],4):>14}{_fmt(r['ms_error'],4):>14}{'n/a':>12}{'n/a':>14}",
            f"{'Total':<22}{r['df_total']:>7}{_fmt(r['ss_total'],4):>14}{'n/a':>14}{'n/a':>12}{'n/a':>14}",
            "\n" + "="*72,
            "COEFFICIENTS DU MODÈLE",
            "="*72,
            f"{'Terme':<28}{'Coef':>12}{'SE':>12}{'t':>12}{'p-value':>14}",
            "-"*83,
        ]
        for row in r["coefficient_table"]:
            lines.append(f"{row['term']:<28}{_fmt(row['coef'],5):>12}{_fmt(row['se'],5):>12}{_fmt(row['t'],4):>12}{_fmt_p(row['p']):>14}")
        lines += ["\n" + "="*72, "EFFETS CLASSÉS PAR IMPORTANCE", "="*72]
        if r["effects_sorted"]:
            lines.append(f"{'Rang':<6}{'Terme':<28}{'Effet':>12}{'t abs.':>12}{'p-value':>14}{'Sig.':>8}")
            lines.append("-"*83)
            for i, eff in enumerate(r["effects_sorted"], start=1):
                sig = "Oui" if eff["significant"] else "Non"
                lines.append(f"{i:<6}{eff['term']:<28}{_fmt(eff['effect_estimate'],5):>12}{_fmt(eff['abs_standardized_effect'],4):>12}{_fmt_p(eff['p']):>14}{sig:>8}")
        else:
            lines.append("Aucun effet calculable.")
        sig_terms = [e["term"] for e in r["effects_sorted"] if e["significant"]]
        lines += ["\n" + "="*72, "LECTURE FACTUELLE", "="*72]
        if sig_terms:
            lines.append("Effets significatifs détectés : " + ", ".join(sig_terms))
            lines.append("Ces facteurs/termes doivent être étudiés en priorité pour optimiser la réponse.")
        else:
            lines.append("Aucun effet individuel significatif n'est détecté au seuil alpha choisi.")
        lines.append("="*72)
        return "\n".join(lines)


class FullFactorialDOE:
    """Plan factoriel complet général."""

    def __init__(self, factors: Dict[str, List[float]], responses: np.ndarray = None, model="2fi", alpha=0.05):
        if not isinstance(factors, dict) or not factors:
            raise ValueError("factors doit être un dictionnaire non vide {nom: niveaux}.")
        self.factors = {str(k): list(v) for k, v in factors.items()}
        self.factor_names = list(self.factors.keys())
        self.n_factors = len(self.factors)
        self.responses = np.asarray(responses, dtype=float) if responses is not None else None
        self.model = model
        self.alpha = alpha
        self._design_matrix = None
        self._results: Dict[str, Any] = {}
        self._generate_design()
        if self.responses is not None:
            self._analyze()

    def _generate_design(self):
        levels = []
        for name in self.factor_names:
            vals = np.asarray(self.factors[name], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                raise ValueError(f"Le facteur {name} doit contenir au moins 2 niveaux numériques finis.")
            levels.append(list(vals))
        self._design_matrix = np.asarray(list(product(*levels)), dtype=float)

    def get_design_matrix(self) -> np.ndarray:
        return self._design_matrix.copy()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._design_matrix, columns=self.factor_names)
        df.insert(0, "RunOrder", np.arange(1, len(df)+1))
        return df

    def randomize(self, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self._design_matrix))
        self._design_matrix = self._design_matrix[idx]
        return self

    def replicate(self, n_replicates=1):
        n_replicates = int(n_replicates)
        if n_replicates < 1:
            raise ValueError("n_replicates doit être >= 1")
        self._design_matrix = np.repeat(self._design_matrix, n_replicates, axis=0)
        return self

    def set_responses(self, responses: np.ndarray):
        self.responses = np.asarray(responses, dtype=float)
        self._analyze()

    def _analyze(self):
        analyzer = DOEAnalyzer(self._design_matrix, self.responses, self.factor_names, model=self.model, alpha=self.alpha)
        self._results = analyzer.get_results()

    def get_results(self) -> Dict[str, Any]:
        return self._results

    def get_summary(self) -> str:
        if self.responses is None:
            lines = ["="*72, "PLAN FACTORIEL COMPLET", "="*72]
            lines.append(f"Nombre de facteurs = {self.n_factors}")
            lines.append(f"Nombre d'essais = {len(self._design_matrix)}")
            lines.append("Facteurs = " + ", ".join(self.factor_names))
            return "\n".join(lines)
        return DOEAnalyzer(self._design_matrix, self.responses, self.factor_names, model=self.model, alpha=self.alpha).get_summary()


class ResponseSurfaceDOE:
    """Plans de surface de réponse : CCD et Box-Behnken."""

    @staticmethod
    def central_composite(factor_names: Sequence[str], alpha="rotatable", center_points=5, face_centered=False):
        names = [_safe_colname(n) for n in factor_names]
        k = len(names)
        if k < 2:
            raise ValueError("Un plan CCD nécessite au moins 2 facteurs.")
        factorial = np.asarray(list(product([-1.0, 1.0], repeat=k)), dtype=float)
        if face_centered:
            a = 1.0
        elif alpha == "rotatable":
            a = float((2 ** k) ** 0.25)
        else:
            a = float(alpha)
        axial = []
        for i in range(k):
            row = np.zeros(k); row[i] = a; axial.append(row.copy())
            row = np.zeros(k); row[i] = -a; axial.append(row.copy())
        center = np.zeros((int(center_points), k))
        design = np.vstack([factorial, np.asarray(axial), center])
        return design, names, {"type": "CCD", "alpha": a, "center_points": int(center_points)}

    @staticmethod
    def box_behnken(factor_names: Sequence[str], center_points=3):
        names = [_safe_colname(n) for n in factor_names]
        k = len(names)
        if k < 3:
            raise ValueError("Un plan Box-Behnken nécessite au moins 3 facteurs.")
        rows = []
        for i, j in combinations(range(k), 2):
            for levels in product([-1.0, 1.0], repeat=2):
                row = np.zeros(k)
                row[i], row[j] = levels
                rows.append(row)
        rows.extend([np.zeros(k) for _ in range(int(center_points))])
        return np.asarray(rows, dtype=float), names, {"type": "Box-Behnken", "center_points": int(center_points)}


class TaguchiOA:
    """Plans de Taguchi et analyse signal/bruit."""

    def __init__(self, n_factors: int, n_levels: int = 2):
        self.n_factors = int(n_factors)
        self.n_levels = int(n_levels)
        self._design_matrix = None
        self.design_name = None
        self._generate()

    def _generate(self):
        if self.n_factors < 1:
            raise ValueError("n_factors doit être >= 1")
        if self.n_levels == 2:
            if self.n_factors <= 3:
                base = self._l4(); self.design_name = "L4"
            elif self.n_factors <= 7:
                base = self._l8(); self.design_name = "L8"
            elif self.n_factors <= 15:
                base = self._l16(); self.design_name = "L16"
            else:
                base = np.asarray(list(product([-1, 1], repeat=self.n_factors)), dtype=float); self.design_name = "Factoriel complet 2^k"
            self._design_matrix = base[:, :self.n_factors]
        elif self.n_levels == 3:
            if self.n_factors <= 4:
                base = self._l9(); self.design_name = "L9"
            elif self.n_factors <= 13:
                base = self._l27(); self.design_name = "L27"
            else:
                base = np.asarray(list(product([0, 1, 2], repeat=self.n_factors)), dtype=float); self.design_name = "Factoriel complet 3^k"
            self._design_matrix = base[:, :self.n_factors]
        else:
            levels = list(range(self.n_levels))
            self._design_matrix = np.asarray(list(product(levels, repeat=self.n_factors)), dtype=float)
            self.design_name = f"Factoriel complet {self.n_levels}^{self.n_factors}"

    def _l4(self):
        return np.array([[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)

    def _l8(self):
        return np.array([
            [-1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1],
            [-1, 1, 1, -1, -1, 1, 1], [-1, 1, 1, 1, 1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1], [1, -1, 1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1], [1, 1, -1, 1, -1, -1, 1]], dtype=float)

    def _l9(self):
        return np.array([[0,0,0,0], [0,1,1,1], [0,2,2,2], [1,0,1,2], [1,1,2,0], [1,2,0,1], [2,0,2,1], [2,1,0,2], [2,2,1,0]], dtype=float)

    def _l16(self):
        # Construction 16 lignes × 15 colonnes via colonnes de base et produits modulaires pour niveaux ±1.
        base4 = np.asarray(list(product([-1, 1], repeat=4)), dtype=float)
        cols = [base4[:, i] for i in range(4)]
        for r in range(2, 5):
            for comb in combinations(range(4), r):
                prod_col = np.prod(base4[:, comb], axis=1)
                cols.append(prod_col)
        return np.column_stack(cols[:15])

    def _l27(self):
        base = np.asarray(list(product([0, 1, 2], repeat=3)), dtype=float)
        cols = [base[:, 0], base[:, 1], base[:, 2]]
        # Colonnes supplémentaires orthogonales approximées modulo 3.
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    col = (a*base[:,0] + b*base[:,1] + c*base[:,2]) % 3
                    if not any(np.array_equal(col, existing) for existing in cols):
                        cols.append(col)
                    if len(cols) >= 13:
                        return np.column_stack(cols[:13])
        return np.column_stack(cols[:13])

    def get_design_matrix(self) -> np.ndarray:
        return self._design_matrix.copy()

    def to_dataframe(self, factor_names=None) -> pd.DataFrame:
        names = factor_names or [f"F{i+1}" for i in range(self.n_factors)]
        df = pd.DataFrame(self._design_matrix, columns=names)
        df.insert(0, "RunOrder", np.arange(1, len(df)+1))
        return df

    @staticmethod
    def signal_to_noise(values, goal="larger_better"):
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            return np.nan
        goal = str(goal).lower()
        if goal == "larger_better":
            if np.any(y == 0):
                return np.nan
            return float(-10*np.log10(np.mean(1/(y**2))))
        if goal == "smaller_better":
            return float(-10*np.log10(np.mean(y**2)))
        # nominal_better
        mean = np.mean(y); var = np.var(y, ddof=1) if len(y) > 1 else 0
        return float(10*np.log10((mean**2)/var)) if var > 0 else np.inf

    def analyze(self, responses, factor_names=None, goal="larger_better"):
        y = np.asarray(responses, dtype=float).reshape(-1)
        if len(y) != len(self._design_matrix):
            return {"is_valid": False, "errors": [f"Nombre de réponses incohérent : attendu {len(self._design_matrix)}, reçu {len(y)}."]}
        names = factor_names or [f"F{i+1}" for i in range(self.n_factors)]
        rows = []
        for j, name in enumerate(names):
            for level in np.unique(self._design_matrix[:, j]):
                vals = y[self._design_matrix[:, j] == level]
                vals = vals[np.isfinite(vals)]
                rows.append({"factor": name, "level": level, "n": len(vals), "mean": float(np.mean(vals)) if len(vals) else None, "sn": self.signal_to_noise(vals, goal)})
        df = pd.DataFrame(rows)
        summary = []
        for name in names:
            sub = df[df["factor"] == name]
            delta_mean = sub["mean"].max() - sub["mean"].min()
            delta_sn = sub["sn"].max() - sub["sn"].min()
            best_row = sub.loc[sub["sn"].idxmax()] if sub["sn"].notna().any() else None
            summary.append({"factor": name, "delta_mean": float(delta_mean), "delta_sn": float(delta_sn), "best_level": None if best_row is None else best_row["level"]})
        return {"is_valid": True, "design_name": self.design_name, "goal": goal, "level_table": df.to_dict(orient="records"), "factor_importance": summary}

    def get_summary(self) -> str:
        lines = ["="*72, "TABLEAU ORTHOGONAL DE TAGUCHI", "="*72]
        lines.append(f"Nom du plan = {self.design_name}")
        lines.append(f"Nombre de facteurs = {self.n_factors}")
        lines.append(f"Nombre de niveaux = {self.n_levels}")
        lines.append(f"Nombre d'essais = {self._design_matrix.shape[0]}")
        lines.append("\nMatrice du plan :")
        for i, row in enumerate(self._design_matrix, start=1):
            lines.append(f"Essai {i:>2d}: {row}")
        return "\n".join(lines)
