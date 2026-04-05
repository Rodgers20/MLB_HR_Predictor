"""
SHAP-based per-prediction explainer for MLB HR Predictor.
Generates human-readable insight text explaining WHY the model favors each pick.

Usage:
    from utils.explainer import explain_prediction
    insight_text, factors = explain_prediction(model, feat_scaled, MODEL_FEATURES, feat_values_dict)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Human-readable display metadata per feature: (label, direction_hint, unit)
FEATURE_DISPLAY = {
    "barrel_pct_w3yr":     ("Barrel Rate",          "high = elite contact",        "%"),
    "exit_velo_w3yr":      ("Exit Velocity",         "high = hard hitter",          "mph"),
    "hard_hit_pct_w3yr":   ("Hard Hit %",            "high = consistent power",     "%"),
    "xwoba_w3yr":          ("xwOBA",                 "high = elite hitter",         ""),
    "xslg_w3yr":           ("xSLG",                  "high = extra-base threat",    ""),
    "launch_angle_w3yr":   ("Launch Angle",          "optimal 10–30°",              "°"),
    "iso":                 ("ISO Power",             "above .200 = slugger",        ""),
    "hr_fb_rate":          ("HR/FB Rate",            "high = converts fly balls",   "%"),
    "fb_pct":              ("Fly Ball %",            "high = more HR chances",      "%"),
    "pull_pct":            ("Pull Rate",             "high = pull-side power",      "%"),
    "wrc_plus":            ("wRC+",                  "above 100 = above average",   ""),
    "sweet_spot_pct":      ("Sweet Spot %",          "8–32° = line drives/HR",      "%"),
    "max_ev_95th":         ("Peak Exit Velo",        "95th pct — ceiling speed",    "mph"),
    "platoon_adj":         ("Platoon Matchup",       "L/R splits",                  ""),
    "pitcher_hr9":         ("Pitcher HR/9",          "high = gives up HRs",         ""),
    "pitcher_hr_fb":       ("Pitcher HR/FB",         "high = fly balls leave park", "%"),
    "pitcher_fip":         ("Pitcher FIP",           "high = less reliable",        ""),
    "pitcher_xfip":        ("Pitcher xFIP",          "high = volatile",             ""),
    "pitcher_gb_pct":      ("Pitcher GB%",           "high = fewer fly balls",      "%"),
    "pitcher_k_pct":       ("Pitcher K%",            "high = fewer balls in play",  "%"),
    "pitcher_barrel_pct":  ("Pitcher Barrel%",       "high = allows barrels",       "%"),
    "pitcher_hard_hit_pct":("Pitcher Hard Hit%",     "high = allows hard contact",  "%"),
    "park_hr_factor":      ("Park Factor",           "above 100 = HR-friendly",     ""),
    "wind_direction_enc":  ("Wind Direction",        "1.0=out, -1.0=in",            ""),
    "wind_speed_mph":      ("Wind Speed",            "high+out = HR boost",         "mph"),
    "temp_f":              ("Temperature",           "hot = ball carries farther",  "°F"),
    "is_day_game":         ("Day Game",              "slight day/night variance",   ""),
}

# Module-level cache: avoid re-creating TreeExplainer on every prediction
_explainer_cache: dict = {}


def load_explainer(model):
    """
    Return a cached SHAP TreeExplainer for the given CalibratedXGB model.
    Accesses model.model (the raw XGBClassifier) so SHAP can traverse the trees.
    """
    try:
        import shap
        key = id(model)
        if key not in _explainer_cache:
            base = model.model if hasattr(model, "model") else model
            _explainer_cache[key] = shap.TreeExplainer(base)
            logger.debug("Initialized SHAP TreeExplainer for model id=%d", key)
        return _explainer_cache[key]
    except Exception as exc:
        logger.warning("Could not initialise SHAP explainer: %s", exc)
        return None


def explain_prediction(
    model,
    feat_scaled_row: np.ndarray,
    feature_names: list,
    feat_values_dict: dict,
) -> tuple:
    """
    Compute SHAP values for a single scaled feature row and produce insight text.

    Args:
        model: CalibratedXGB wrapper (has .model = XGBClassifier)
        feat_scaled_row: (1, n_features) or (n_features,) numpy array after scaler.transform()
        feature_names: ordered list of feature names matching feat_scaled_row columns
        feat_values_dict: raw (unscaled) feature values for display formatting

    Returns:
        insight_text: str — "high barrel rate (11.2%) · pitcher gives up HRs (HR/FB: 22%) · hitter-friendly park (+15%)"
        top_factors:  list[(label, is_positive, description)]
    """
    fallback = ("Model-selected pick", [])
    try:
        explainer = load_explainer(model)
        if explainer is None:
            return fallback

        row = np.array(feat_scaled_row)
        if row.ndim == 1:
            row = row.reshape(1, -1)

        shap_vals = explainer.shap_values(row)

        # XGBoost binary classifier returns a 2D array (n_samples, n_features)
        if isinstance(shap_vals, list):
            # Some versions return [neg_class, pos_class]
            sv = shap_vals[1][0] if len(shap_vals) == 2 else shap_vals[0][0]
        elif hasattr(shap_vals, "ndim"):
            sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
        else:
            return fallback

        if len(sv) != len(feature_names):
            logger.debug("SHAP length mismatch: %d values vs %d features", len(sv), len(feature_names))
            return fallback

        # Pair (feature, shap_value, raw_value), sort by |shap|
        paired = sorted(
            [(f, float(s), feat_values_dict.get(f, np.nan)) for f, s in zip(feature_names, sv)],
            key=lambda t: abs(t[1]),
            reverse=True,
        )

        factors = []
        for feat, shap_val, raw_val in paired:
            if len(factors) >= 3:
                break
            desc = _describe_factor(feat, shap_val, raw_val)
            if desc:
                label = FEATURE_DISPLAY.get(feat, (feat,))[0]
                factors.append((label, shap_val > 0, desc))

        if not factors:
            return fallback

        insight_text = " · ".join(f[2] for f in factors)
        return insight_text, factors

    except Exception as exc:
        logger.debug("explain_prediction failed: %s", exc)
        return fallback


def _nan_safe(val) -> float | None:
    """Return float or None; treats NaN/None as None."""
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _describe_factor(feat: str, shap_val: float, raw_val) -> str | None:
    """
    Turn a (feature, shap_value, raw_value) triple into a short English phrase.
    Returns None if the factor is not interesting enough to display.
    """
    pos = shap_val > 0
    v = _nan_safe(raw_val)

    match feat:
        case "barrel_pct_w3yr":
            if v is not None:
                pct = v * 100
                if pos and pct >= 8.0:
                    return f"high barrel rate ({pct:.1f}%)"
                elif not pos and pct < 5.5:
                    return f"low barrel rate ({pct:.1f}%)"

        case "exit_velo_w3yr":
            if v is not None:
                if pos and v >= 90.5:
                    return f"{v:.1f} mph avg exit velocity"
                elif not pos and v < 87.5:
                    return f"soft contact ({v:.1f} mph avg EV)"

        case "hard_hit_pct_w3yr":
            if v is not None and pos and v >= 0.42:
                return f"hard contact rate {v*100:.0f}%"

        case "xwoba_w3yr":
            if v is not None and pos and v >= 0.330:
                return f"xwOBA {v:.3f} (elite hitter)" if v >= 0.370 else f"xwOBA {v:.3f} (above-avg)"

        case "xslg_w3yr":
            if v is not None and pos and v >= 0.450:
                return f"xSLG {v:.3f} (extra-base threat)"

        case "iso":
            if v is not None:
                if pos and v >= 0.200:
                    return f"ISO {v:.3f} (power hitter)"
                elif pos and v >= 0.150:
                    return f"ISO {v:.3f} (above-avg power)"

        case "hr_fb_rate":
            if v is not None and pos and v >= 0.15:
                return f"HR/FB rate {v*100:.0f}% (converts fly balls)"

        case "fb_pct":
            if v is not None and pos and v >= 0.38:
                return f"fly-ball hitter ({v*100:.0f}% FB rate)"

        case "pitcher_hr_fb":
            if v is not None and pos and v >= 0.12:
                return f"pitcher gives up HRs (HR/FB: {v*100:.0f}%)"

        case "pitcher_hr9":
            if v is not None and pos and v >= 1.2:
                return f"pitcher allows {v:.1f} HR/9"

        case "pitcher_gb_pct":
            if v is not None:
                # Low GB% = fly-ball pitcher = more HRs → positive SHAP
                if pos and v <= 0.42:
                    return "fly-ball pitcher (HR-vulnerable)"
                # High GB% = groundball pitcher = fewer HRs → negative SHAP
                elif not pos and v >= 0.54:
                    return "groundball pitcher (suppresses HRs)"

        case "pitcher_fip":
            if v is not None and pos and v >= 4.50:
                return f"shaky pitcher (FIP {v:.2f})"

        case "pitcher_xfip":
            if v is not None and pos and v >= 4.50:
                return f"volatile pitcher (xFIP {v:.2f})"

        case "park_hr_factor":
            if v is not None:
                delta = v - 100
                if delta >= 8:
                    return f"hitter-friendly park (+{delta:.0f}% vs avg)"
                elif delta <= -8:
                    return f"pitcher-friendly park ({delta:.0f}% vs avg)"

        case "wind_direction_enc":
            if v is not None:
                if v >= 0.7:
                    return "wind blowing out (HR-boost)"
                elif v <= -0.7:
                    return "wind blowing in (suppressor)"

        case "temp_f":
            if v is not None and pos and v >= 82:
                return f"{v:.0f}°F (hot — ball carries farther)"

        case "platoon_adj":
            if v is not None:
                if v >= 1.10:
                    return "platoon advantage"
                elif v <= 0.82:
                    return "platoon disadvantage"

        case "wrc_plus":
            if v is not None and pos and v >= 125:
                return f"wRC+ {int(v)} (elite hitter)"

        case "pull_pct":
            if v is not None and pos and v >= 0.44:
                return f"pull-side power ({v*100:.0f}% pull rate)"

        case _:
            # Generic fallback — only emit when value is genuinely above baseline
            # (v = 0 means feature was unset/missing → skip to avoid noise)
            if v is None or v == 0.0:
                return None
            label = FEATURE_DISPLAY.get(feat, (feat,))[0]
            if feat.startswith("pitcher_") and pos and v > 0:
                return f"favorable pitcher matchup ({label.lower()})"
            if pos and v > 0:
                return f"strong {label.lower()}"

    return None
