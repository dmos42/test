"""MiniQual - Fonctions d'analyse de capabilité (extrait de main.py, reformatté)."""

from pathlib import Path
from datetime import datetime
import json
import math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt

SUPPORTED_DISTRIBUTIONS = [
    'normal', 'lognormal', 'weibull', 'exponential', 'gamma',
    'logistic', 'gumbel', 'cauchy', 'rayleigh', 'uniform', 'student_t', 'laplace'
]

LABEL = {
    'normal': 'normale',
    'lognormal': 'lognormale',
    'weibull': 'Weibull',
    'exponential': 'exponentielle 2P',
    'gamma': 'gamma',
    'logistic': 'logistique',
    'gumbel': 'Gumbel max',
    'cauchy': 'Cauchy',
    'rayleigh': 'Rayleigh',
    'uniform': 'uniforme',
    'student_t': 'Student-t',
    'laplace': 'Laplace',
}


def read_table(path, sheet_name=0):
    """Lit un fichier CSV ou Excel et retourne un DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(p, sep=None, engine='python', encoding='utf-8-sig')
    elif suffix in ['.xlsx', '.xlsm']:
        df = pd.read_excel(p, sheet_name=sheet_name, engine='openpyxl')
    elif suffix == '.xls':
        df = pd.read_excel(p, sheet_name=sheet_name)
    else:
        raise ValueError(f'Format non supporté: {suffix}')
    df.columns = [str(c).lstrip('\ufeff').strip() for c in df.columns]
    return df


def numeric_series(df, col):
    """Extrait une série numérique finie d'une colonne, en ignorant NaN et Inf."""
    if col not in df.columns:
        raise KeyError(f'Colonne introuvable: {col}. Colonnes: {list(df.columns)}')
    s = pd.to_numeric(df[col], errors='coerce')
    s = s[np.isfinite(s)]
    if s.empty:
        raise ValueError('Aucune valeur numérique finie exploitable')
    return s


def descriptive(s):
    """Statistiques descriptives d'une série numérique finie."""
    s = pd.to_numeric(s, errors='coerce')
    s = s[np.isfinite(s)]
    if s.empty:
        raise ValueError('Aucune valeur numérique finie exploitable')
    return {
        'n': len(s),
        'moyenne': float(s.mean()),
        'mediane': float(s.median()),
        'ecart_type_echantillon': float(s.std(ddof=1)) if len(s) > 1 else None,
        'minimum': float(s.min()),
        'q1': float(s.quantile(0.25)),
        'q3': float(s.quantile(0.75)),
        'maximum': float(s.max()),
    }


def _pos(x, d):
    """Filtre les valeurs selon le support de la distribution."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if d in ['lognormal', 'weibull', 'gamma', 'rayleigh']:
        return x[x > 0]
    # Exponentielle 2P : loc libre, les valeurs négatives sont possibles si le procédé est décalé.
    return x

def _dist(d):
    """Retourne l'objet scipy.stats pour une distribution."""
    return {
        'normal': scipy_stats.norm,
        'lognormal': scipy_stats.lognorm,
        'weibull': scipy_stats.weibull_min,
        'exponential': scipy_stats.expon,
        'gamma': scipy_stats.gamma,
        'logistic': scipy_stats.logistic,
        'gumbel': scipy_stats.gumbel_r,
        'cauchy': scipy_stats.cauchy,
        'rayleigh': scipy_stats.rayleigh,
        'uniform': scipy_stats.uniform,
        'student_t': scipy_stats.t,
        'laplace': scipy_stats.laplace,
    }[d]

def _name(d):
    """Nom scipy pour le test KS."""
    return {
        'normal': 'norm',
        'lognormal': 'lognorm',
        'weibull': 'weibull_min',
        'exponential': 'expon',
        'gamma': 'gamma',
        'logistic': 'logistic',
        'gumbel': 'gumbel_r',
        'cauchy': 'cauchy',
        'rayleigh': 'rayleigh',
        'uniform': 'uniform',
        'student_t': 't',
        'laplace': 'laplace',
    }[d]

def fit_distribution(series, d):
    """Ajuste une distribution aux données et retourne les résultats du test KS."""
    x = pd.Series(series).dropna().astype(float).to_numpy()
    xp = _pos(x, d)
    if len(xp) < 3:
        return {
            'loi': d,
            'p_value': np.nan,
            'statistique': np.nan,
            'paramètres': 'Données incompatibles',
            'params': None,
        }
    try:
        if d == 'normal':
            params = scipy_stats.norm.fit(xp)
        elif d in ['lognormal', 'weibull', 'gamma', 'rayleigh']:
            params = _dist(d).fit(xp, floc=0)
        else:
            # Exponentielle 2P, logistique, Gumbel, Cauchy, uniforme, Student-t, Laplace : loc libre.
            params = _dist(d).fit(xp)
        D, p = scipy_stats.kstest(xp, _name(d), args=params)
        return {
            'loi': d,
            'p_value': float(p),
            'statistique': float(D),
            'paramètres': str(tuple(round(float(v), 6) for v in params)),
            'params': params,
        }
    except Exception as e:
        return {
            'loi': d,
            'p_value': np.nan,
            'statistique': np.nan,
            'paramètres': f'Erreur : {e}',
            'params': None,
        }

def distribution_pvalues(s):
    """Retourne les p-values de toutes les distributions triées (meilleure en premier)."""
    results = []
    for d in SUPPORTED_DISTRIBUTIONS:
        r = fit_distribution(s, d)
        results.append({k: v for k, v in r.items() if k != 'params'})
    results.sort(key=lambda r: (-1 if r['p_value'] != r['p_value'] else -r['p_value']))
    return results


def chi_square_gof(s, distribution='normal', bins='auto'):
    """Test d'ajustement du Khi-deux."""
    x = pd.Series(s).dropna().astype(float).to_numpy()
    n = len(x)
    if n < 10:
        return {
            'Test': 'Khi²',
            'Loi testée': distribution,
            'Statistique': np.nan,
            'ddl': np.nan,
            'p-value': np.nan,
            'Lecture': 'Effectif insuffisant (<10)',
            'Classes': 'n/a',
        }
    k = max(4, min(10, int(np.sqrt(n)))) if str(bins) == 'auto' else int(bins)
    fit = fit_distribution(x, distribution)
    p = fit.get('params')
    if p is None:
        return {
            'Test': 'Khi²',
            'Loi testée': distribution,
            'Statistique': np.nan,
            'ddl': np.nan,
            'p-value': np.nan,
            'Lecture': 'Paramètres non estimables',
            'Classes': 'n/a',
        }
    edges = _dist(distribution).ppf(np.linspace(0, 1, k + 1), *p)
    edges[0], edges[-1] = -np.inf, np.inf
    obs, _ = np.histogram(x, bins=edges)
    exp = np.ones(k) * n / k
    chi = float(((obs - exp) ** 2 / exp).sum())
    ddl = max(1, k - 1 - len(p))
    pv = float(scipy_stats.chi2.sf(chi, ddl))
    lecture = 'Compatible avec la loi testée' if pv > 0.05 else 'Écart possible avec la loi testée'
    return {
        'Test': 'Khi²',
        'Loi testée': distribution,
        'Statistique': chi,
        'ddl': ddl,
        'p-value': pv,
        'Lecture': lecture,
        'Classes': k,
    }


def boxcox_transform(s, lsl=None, usl=None, target=None):
    """Transformation de Box-Cox pour normaliser les données."""
    x = pd.Series(s).dropna().astype(float)
    shift = 0.0
    if x.min() <= 0:
        shift = float(abs(x.min()) + 1e-6)
    y, lam = scipy_stats.boxcox(x + shift)

    def tr(v):
        if v is None:
            return None
        vv = float(v) + shift
        return float(scipy_stats.boxcox([vv], lmbda=lam)[0]) if vv > 0 else None

    return pd.Series(y, index=x.index), tr(lsl), tr(usl), tr(target), {
        'Box-Cox lambda': float(lam),
        'Décalage appliqué': shift,
    }


def validate_capability(df, col, lsl=None, usl=None, target=None, subgroup=None, distribution='normal'):
    """Valide les paramètres avant une analyse de capabilité."""
    rows = []
    def add(niveau, message):
        rows.append({'Niveau': niveau, 'Message': message})
    if col not in df.columns:
        add('ERREUR', f'Colonne mesure introuvable: {col}')
        return rows
    s = pd.to_numeric(df[col], errors='coerce')
    n = s.notna().sum()
    bad = len(df) - n
    add('OK', f'Colonne mesure trouvée: {col}')
    add('OK', f'{int(n)} valeurs numériques exploitables sur {len(df)} lignes')
    if bad:
        add('AVERTISSEMENT', f'{int(bad)} valeurs vides ou non numériques seront ignorées')
    if n < 10:
        add('ERREUR', 'Moins de 10 valeurs numériques: analyse non robuste')
    elif n < 30:
        add('AVERTISSEMENT', 'Moins de 30 valeurs: interprétation prudente')
    if lsl is None and usl is None:
        add('ERREUR', 'Aucune limite de spécification renseignée')
    if lsl is not None and usl is not None and usl <= lsl:
        add('ERREUR', 'USL doit être strictement supérieure à LSL')
    if target is not None and lsl is not None and usl is not None and not (lsl <= target <= usl):
        add('AVERTISSEMENT', 'La cible est hors intervalle [LSL;USL]')
    if subgroup and subgroup not in df.columns:
        add('AVERTISSEMENT', f'Sous-groupe introuvable: {subgroup}. Calcul sans sous-groupe')
    vals = s.dropna()
    if distribution in ['lognormal', 'weibull', 'gamma', 'rayleigh'] and (vals <= 0).any():
        add('AVERTISSEMENT', f'Loi {distribution}: des valeurs <=0 sont incompatibles avec le support forcé à 0')
    return rows

def status(rows):
    """Retourne le statut global à partir des lignes de validation."""
    if any(r['Niveau'] == 'ERREUR' for r in rows):
        return 'BLOQUÉ'
    if any(r['Niveau'] == 'AVERTISSEMENT' for r in rows):
        return 'OK AVEC AVERTISSEMENT'
    return 'OK'


def capability(s, lsl=None, usl=None, target=None, subgroup=None):
    """Calcule les indices de capabilité (Cp, Cpk, Pp, Ppk, etc.) de manière robuste."""
    x = pd.to_numeric(pd.Series(s), errors='coerce')
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise ValueError('Au moins 2 valeurs numériques finies sont nécessaires')
    m = float(x.mean())
    so = float(x.std(ddof=1))
    mr = x.diff().abs().dropna()
    sw = float(mr.mean() / 1.128) if len(mr) and mr.mean() > 0 else so
    if lsl is None and usl is None:
        raise ValueError('Renseigner LSL ou USL')
    if lsl is not None and usl is not None and usl <= lsl:
        raise ValueError('USL doit être strictement supérieure à LSL')
    if lsl is not None and usl is not None and target is None:
        target = (lsl + usl) / 2
    cu = (usl - m) / (3 * sw) if usl is not None and sw and sw > 0 else None
    cl = (m - lsl) / (3 * sw) if lsl is not None and sw and sw > 0 else None
    pu = (usl - m) / (3 * so) if usl is not None and so and so > 0 else None
    pl = (m - lsl) / (3 * so) if lsl is not None and so and so > 0 else None
    cp = (usl - lsl) / (6 * sw) if lsl is not None and usl is not None and sw and sw > 0 else None
    pp = (usl - lsl) / (6 * so) if lsl is not None and usl is not None and so and so > 0 else None
    def valid(v):
        return v is not None and np.isfinite(v)
    mask = pd.Series(False, index=x.index)
    if lsl is not None:
        mask |= x < lsl
    if usl is not None:
        mask |= x > usl
    cpk_candidates = [v for v in [cu, cl] if valid(v)]
    ppk_candidates = [v for v in [pu, pl] if valid(v)]
    return {
        'n': len(x), 'mean': m, 'stdev_within': sw, 'stdev_overall': so,
        'lsl': lsl, 'usl': usl, 'target': target,
        'cp': cp, 'cpk': min(cpk_candidates) if cpk_candidates else None,
        'cpk_upper': cu, 'cpk_lower': cl,
        'pp': pp, 'ppk': min(ppk_candidates) if ppk_candidates else None,
        'ppk_upper': pu, 'ppk_lower': pl,
        'ppm_below_lsl': float(scipy_stats.norm.cdf((lsl - m) / so) * 1e6) if lsl is not None and so and so > 0 else None,
        'ppm_above_usl': float((1 - scipy_stats.norm.cdf((usl - m) / so)) * 1e6) if usl is not None and so and so > 0 else None,
        'observed_nc_count': int(mask.sum()),
        'observed_nc_percent': float(mask.mean() * 100),
        'spec_mode': (
            'Bilatéral LSL+USL' if lsl is not None and usl is not None
            else 'Unilatéral supérieur USL' if usl is not None
            else 'Unilatéral inférieur LSL'
        ),
    }

    if lsl is None and usl is None:
        raise ValueError('Renseigner LSL ou USL')

    if lsl is not None and usl is not None and target is None:
        target = (lsl + usl) / 2

    cu = (usl - m) / (3 * sw) if usl is not None and sw > 0 else None
    cl = (m - lsl) / (3 * sw) if lsl is not None and sw > 0 else None
    pu = (usl - m) / (3 * so) if usl is not None and so > 0 else None
    pl = (m - lsl) / (3 * so) if lsl is not None and so > 0 else None
    cp = (usl - lsl) / (6 * sw) if lsl is not None and usl is not None and sw > 0 else None
    pp = (usl - lsl) / (6 * so) if lsl is not None and usl is not None and so > 0 else None

    def valid(v):
        return v is not None and not np.isnan(v)

    mask = pd.Series(False, index=x.index)
    if lsl is not None:
        mask |= x < lsl
    if usl is not None:
        mask |= x > usl

    cpk_candidates = [v for v in [cu, cl] if valid(v)]
    ppk_candidates = [v for v in [pu, pl] if valid(v)]

    return {
        'n': len(x),
        'mean': m,
        'stdev_within': sw,
        'stdev_overall': so,
        'lsl': lsl,
        'usl': usl,
        'target': target,
        'cp': cp,
        'cpk': min(cpk_candidates) if cpk_candidates else None,
        'cpk_upper': cu,
        'cpk_lower': cl,
        'pp': pp,
        'ppk': min(ppk_candidates) if ppk_candidates else None,
        'ppk_upper': pu,
        'ppk_lower': pl,
        'ppm_below_lsl': float(scipy_stats.norm.cdf((lsl - m) / so) * 1e6) if lsl is not None and so > 0 else None,
        'ppm_above_usl': float((1 - scipy_stats.norm.cdf((usl - m) / so)) * 1e6) if usl is not None and so > 0 else None,
        'observed_nc_count': int(mask.sum()),
        'observed_nc_percent': float(mask.mean() * 100),
        'spec_mode': (
            'Bilatéral LSL+USL' if lsl is not None and usl is not None
            else 'Unilatéral supérieur USL' if usl is not None
            else 'Unilatéral inférieur LSL'
        ),
    }


def outlier_tests(s, alpha=0.05, z_threshold=3.0):
    """Tests de détection de valeurs aberrantes (IQR et Z-score)."""
    x = pd.Series(s).dropna().astype(float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    oi = x[(x < lo) | (x > hi)]

    z = (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else x * 0
    oz = x[z.abs() > z_threshold]

    def fmt_values(series, limit=12):
        vals = series.tolist()[:limit]
        return ', '.join(map(str, vals)) if vals else 'Aucune'

    return {
        'table': [
            {
                'Méthode': 'IQR',
                'Seuil / statistique': f'[{lo:.6g}; {hi:.6g}]',
                'Valeurs détectées': fmt_values(oi),
                'Nombre': len(oi),
            },
            {
                'Méthode': f'Z-score absolu > {z_threshold:g}',
                'Seuil / statistique': f'|Z| > {z_threshold:g}',
                'Valeurs détectées': fmt_values(oz),
                'Nombre': len(oz),
            },
        ],
        'values_to_exclude': list(set(map(float, oi)).union(set(map(float, oz)))),
    }


def normality_tests(s, alpha=0.05):
    """Tests de normalité sécurisés : Shapiro-Wilk, KS indicatif, Anderson-Darling."""
    x = pd.to_numeric(pd.Series(s), errors='coerce').to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {
            'table': [],
            'favorable_count': 0,
            'global_ok': False,
            'error': 'Au moins 3 valeurs numériques finies sont nécessaires pour les tests de normalité.',
            'n': int(len(x)),
        }
    if len(np.unique(x)) <= 1:
        return {
            'table': [],
            'favorable_count': 0,
            'global_ok': False,
            'error': 'Données constantes : tests de normalité non applicables.',
            'n': int(len(x)),
        }
    sample = x if len(x) <= 5000 else pd.Series(x).sample(5000, random_state=42).to_numpy()
    sh, shp = scipy_stats.shapiro(sample)
    sd = np.std(x, ddof=1)
    if sd > 0:
        ks, ksp = scipy_stats.kstest(x, 'norm', args=(np.mean(x), sd))
    else:
        ks, ksp = np.nan, np.nan
    ad = scipy_stats.anderson(x, 'norm')
    crit = float(ad.critical_values[2])
    ok = [
        shp > alpha,
        ksp > alpha if np.isfinite(ksp) else False,
        float(ad.statistic) < crit,
    ]
    return {
        'table': [
            {'Test': 'Shapiro-Wilk', 'Statistique': float(sh), 'p-value / seuil': float(shp), 'Lecture': 'Compatible avec une loi normale' if ok[0] else 'Écart possible à la normalité'},
            {'Test': 'Kolmogorov-Smirnov indicatif', 'Statistique': float(ks) if np.isfinite(ks) else np.nan, 'p-value / seuil': float(ksp) if np.isfinite(ksp) else np.nan, 'Lecture': 'Compatible avec une loi normale' if ok[1] else 'Écart possible à la normalité'},
            {'Test': 'Anderson-Darling', 'Statistique': float(ad.statistic), 'p-value / seuil': f'Critique 5% : {crit:.6g}', 'Lecture': 'Compatible avec une loi normale' if ok[2] else 'Écart possible à la normalité'},
        ],
        'favorable_count': int(sum(ok)),
        'global_ok': sum(ok) >= 2,
        'n': int(len(x)),
    }

    ok = [
        shp > alpha,
        ksp > alpha,
        float(ad.statistic) < crit,
    ]

    return {
        'table': [
            {
                'Test': 'Shapiro-Wilk',
                'Statistique': float(sh),
                'p-value / seuil': float(shp),
                'Lecture': 'Compatible avec une loi normale' if ok[0] else 'Écart possible à la normalité',
            },
            {
                'Test': 'Kolmogorov-Smirnov',
                'Statistique': float(ks),
                'p-value / seuil': float(ksp),
                'Lecture': 'Compatible avec une loi normale' if ok[1] else 'Écart possible à la normalité',
            },
            {
                'Test': 'Anderson-Darling',
                'Statistique': float(ad.statistic),
                'p-value / seuil': f'Critique 5% : {crit:.6g}',
                'Lecture': 'Compatible avec une loi normale' if ok[2] else 'Écart possible à la normalité',
            },
        ],
        'favorable_count': int(sum(ok)),
        'global_ok': sum(ok) >= 2,
    }


def nonnormal_capability(s, lsl=None, usl=None, distribution='empirical'):
    """Capabilité non normale.

    - distribution='empirical' : conserve l'ancien comportement par percentiles empiriques.
    - autre distribution supportée : percentiles de la loi ajustée + ppm prédits.
    """
    x = pd.Series(s).dropna().astype(float)
    if distribution in (None, 'empirical'):
        lo = x.quantile(0.00135)
        hi = x.quantile(0.99865)
        med = x.median()
        out = {
            'méthode': 'Percentiles empiriques 0,135% / 99,865%',
            'q0_135': float(lo),
            'q99_865': float(hi),
            'mediane': float(med),
        }
    else:
        fit = fit_distribution(x, distribution)
        p = fit.get('params')
        if p is None:
            return {
                'méthode': f'Percentiles de loi ajustée ({distribution})',
                'erreur': fit.get('paramètres', 'Paramètres non estimables'),
            }
        dist = _dist(distribution)
        lo = float(dist.ppf(0.00135, *p))
        med = float(dist.ppf(0.5, *p))
        hi = float(dist.ppf(0.99865, *p))
        out = {
            'méthode': f"Percentiles de loi ajustée ({LABEL.get(distribution, distribution)})",
            'loi': distribution,
            'p_value': fit.get('p_value'),
            'paramètres': fit.get('paramètres'),
            'q0_135': float(lo),
            'q99_865': float(hi),
            'mediane': float(med),
        }
        p_below = max(0.0, min(1.0, float(dist.cdf(lsl, *p)))) if lsl is not None else None
        p_above = max(0.0, 1.0 - max(0.0, min(1.0, float(dist.cdf(usl, *p))))) if usl is not None else None
        out['prob_below_lsl'] = p_below
        out['prob_above_usl'] = p_above
        out['prob_total_oos'] = (p_below or 0.0) + (p_above or 0.0)
        out['ppm_below_lsl'] = p_below * 1e6 if p_below is not None else None
        out['ppm_above_usl'] = p_above * 1e6 if p_above is not None else None
        out['ppm_total_oos'] = out['prob_total_oos'] * 1e6

    def safe_ratio(num, den):
        if den is None or abs(float(den)) <= np.finfo(float).eps:
            return None
        if not np.isfinite(num) or not np.isfinite(den):
            return None
        return float(num) / float(den)

    if lsl is not None and usl is not None and hi > lo:
        out['CNp'] = safe_ratio(usl - lsl, hi - lo)
        out['pp_non_normal'] = out['CNp']
    ppl = safe_ratio(med - lsl, med - lo) if lsl is not None else None
    ppu = safe_ratio(usl - med, hi - med) if usl is not None else None
    vals = [v for v in (ppl, ppu) if v is not None and np.isfinite(v)]
    out['ppl_non_normal'] = ppl
    out['ppu_non_normal'] = ppu
    out['CNpk'] = min(vals) if vals else None
    out['ppk_non_normal'] = out['CNpk']
    return out

def dashboard(s, res, out_png, normality=None, distribution='normal', accept=1.33, excellent=1.67):
    """Génère un dashboard de capabilité en 4 panneaux et le sauve en PNG."""
    data = pd.Series(s).dropna().astype(float).to_numpy()
    m = np.mean(data)
    sd = np.std(data, ddof=1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analyse de capabilité du procédé', fontsize=18, fontweight='bold')

    # Histogramme + courbe
    a = ax[0, 0]
    a.hist(data, bins=min(15, max(5, int(np.sqrt(len(data))))), density=True, alpha=0.75, edgecolor='black')
    xs = [data.min(), data.max(), m - 4 * sd, m + 4 * sd]
    xs += [v for v in [res.get('lsl'), res.get('usl'), res.get('target')] if v is not None]
    xx = np.linspace(min(xs), max(xs), 300)
    fit = fit_distribution(data, distribution)
    p = fit.get('params')
    if p is not None:
        a.plot(xx, _dist(distribution).pdf(xx, *p), 'r-', label=f"Courbe {LABEL.get(distribution, distribution)} estimée")
    for key, c, lab, ls in [('lsl', 'green', 'LSL', '--'), ('usl', 'red', 'USL', '--'), ('target', 'purple', 'Cible', ':')]:
        if res.get(key) is not None:
            a.axvline(res[key], color=c, ls=ls, label=f'{lab}={res[key]:.3g}')
    a.axvline(m, color='orange', label=f'Moyenne={m:.3g}')
    a.legend(fontsize=9)
    a.set_title('Distribution')

    # Boxplot
    ax[0, 1].boxplot(data)
    ax[0, 1].set_title('Boxplot')

    # Q-Q plot
    qfit = fit_distribution(data, distribution)
    p = qfit.get('params')
    if p is not None:
        probs = (np.arange(1, len(data) + 1) - 0.5) / len(data)
        q = _dist(distribution).ppf(probs, *p)
        ordered = np.sort(_pos(data, distribution))
        ax[1, 0].scatter(q, ordered)
        lo_min = min(q.min(), ordered.min())
        hi_max = max(q.max(), ordered.max())
        ax[1, 0].plot([lo_min, hi_max], [lo_min, hi_max], 'r-')
    ax[1, 0].set_title(f"Q-Q plot — {LABEL.get(distribution, distribution)}")

    # Barres indices — si la capabilité non normale est disponible, elle est retenue.
    if res.get('ppk_non_normal') is not None:
        index_specs = [
            ('Pp', 'pp_non_normal'),
            ('Ppl', 'ppl_non_normal'),
            ('Ppu', 'ppu_non_normal'),
            ('Ppk', 'ppk_non_normal'),
        ]
        chart_title = f"Indices non normaux — {LABEL.get(distribution, distribution)}"
    else:
        index_specs = [
            ('Pp', 'pp'),
            ('Ppl', 'ppk_lower'),
            ('Ppu', 'ppk_upper'),
            ('Ppk', 'ppk'),
        ]
        chart_title = 'Indices normaux/classiques'
    labels = [lab for lab, _ in index_specs]
    raw_vals = [res.get(key) for _, key in index_specs]
    vals = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in raw_vals]
    colors = ['#1f77b4' if v is not None and np.isfinite(v) else '#bdbdbd' for v in raw_vals]
    bars = ax[1, 1].bar(labels, vals, color=colors)
    ax[1, 1].axhline(accept, color='orange', ls='--', label=f'Accept. {accept:g}')
    ax[1, 1].axhline(excellent, color='green', ls='--', label=f'Excellent {excellent:g}')
    y_candidates = [v for v in vals if np.isfinite(v)] + [accept, excellent, 1.0]
    ax[1, 1].set_ylim(0, max(y_candidates) * 1.25 if max(y_candidates) > 0 else 1)
    for bar, raw in zip(bars, raw_vals):
        if raw is None or not np.isfinite(raw):
            ax[1, 1].text(bar.get_x() + bar.get_width() / 2, 0.02, 'N/A',
                          ha='center', va='bottom', fontsize=9)
        else:
            ax[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{float(raw):.2f}',
                          ha='center', va='bottom', fontsize=9)
    ax[1, 1].set_title(chart_title)
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_excel(path, sheets):
    """Écrit un fichier Excel avec plusieurs feuilles."""
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        for n, d in sheets.items():
            (pd.DataFrame([d]) if isinstance(d, dict) else pd.DataFrame(d)).to_excel(
                w, sheet_name=str(n)[:31], index=False
            )


def write_docx(title, sections, path, signature=None):
    """Génère un rapport Word (.docx)."""
    from docx import Document
    from docx.shared import Inches

    doc = Document()
    doc.add_heading(title, 0)
    doc.add_paragraph('MiniQual Python v7.5.2 — ' + datetime.now().strftime('%d/%m/%Y %H:%M'))

    for name, content in sections:
        doc.add_heading(name, 1)
        if isinstance(content, dict):
            t = doc.add_table(rows=1, cols=2)
            t.style = 'Table Grid'
            t.rows[0].cells[0].text = 'Indicateur'
            t.rows[0].cells[1].text = 'Valeur'
            for k, v in content.items():
                c = t.add_row().cells
                c[0].text = str(k)
                c[1].text = '' if v is None else str(v)
        elif isinstance(content, list) and content:
            cols = list(content[0].keys())
            t = doc.add_table(rows=1, cols=len(cols))
            t.style = 'Table Grid'
            for i, c0 in enumerate(cols):
                t.rows[0].cells[i].text = str(c0)
            for row in content:
                c = t.add_row().cells
                for i, c0 in enumerate(cols):
                    c[i].text = str(row.get(c0, ''))
        elif str(content).endswith('.png') and Path(content).exists():
            doc.add_picture(str(content), width=Inches(6.7))
        else:
            doc.add_paragraph(str(content))

    if signature:
        doc.add_paragraph(f"Signature : {signature}")

    doc.save(path)



def _json_safe(obj):
    """Convertit récursivement les objets NumPy/Pandas/Path pour une sauvegarde JSON robuste."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.to_dict(orient='list'))
    if isinstance(obj, pd.Series):
        return _json_safe(obj.tolist())
    return str(obj)

def save_project(path, settings):
    """Sauvegarde un projet JSON de manière robuste et atomique."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        k: _json_safe(v)
        for k, v in settings.items()
        if k != 'func'
    }
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def load_project(path):
    """Charge un projet depuis un fichier JSON."""
    return json.loads(Path(path).read_text(encoding='utf-8'))
