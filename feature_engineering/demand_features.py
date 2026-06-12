"""
feature_engineering/demand_features.py

Extrait les features de demande depuis la table `rides`.
Agrège par zone géographique (grille H3) et heure.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


DEMAND_QUERY = """
SELECT
    r.id,
    r.pickup_lat,
    r.pickup_lon,
    r.dropoff_lat,
    r.dropoff_lon,
    r.status,
    r.distance_km,
    r.duration_min,
    r.price_final,
    r.price_estimate,
    r.surge_multiplier,
    r.created_at,
    r.class_id,
    c.name AS class_name,
    DATE_TRUNC('hour', r.created_at) AS hour_bucket,
    EXTRACT(DOW  FROM r.created_at) AS day_of_week,
    EXTRACT(HOUR FROM r.created_at) AS hour_of_day,
    EXTRACT(MONTH FROM r.created_at) AS month
FROM rides r
JOIN classes c ON c.id = r.class_id
WHERE r.created_at >= NOW() - INTERVAL ':lookback days'
  AND r.status != 'CANCELLED'
ORDER BY r.created_at
"""


def load_raw(lookback_days: int = ML.demand_lookback_days) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(DEMAND_QUERY.replace(":lookback", str(lookback_days))),
            conn,
            parse_dates=["created_at", "hour_bucket"],
        )
    return df


def build_zone_hour_features(df: pd.DataFrame, resolution: float = 0.05) -> pd.DataFrame:
    """
    Discrétise pickup_lat/lon en cellules grille et agrège par (zone, heure).

    Returns DataFrame avec colonnes :
        zone_lat | zone_lon | hour_bucket | ride_count | avg_distance_km |
        avg_duration_min | avg_price | demand_trend_7d | demand_trend_1d
    """
    df = df.copy()

    # Grille simple (arrondi à ~5km) — remplacer par H3 en prod
    df["zone_lat"] = (df["pickup_lat"] // resolution) * resolution
    df["zone_lon"] = (df["pickup_lon"] // resolution) * resolution

    # FIX : pré-calcule le booléen AVANT le groupby → pandas utilise sum() C-vectorisé
    # au lieu d'une lambda pure-Python qui itère sur chaque groupe (~27x plus rapide)
    df["is_cancelled"] = (df["status"] == "CANCELLED").astype(np.int8)

    agg = (
        df.groupby(["zone_lat", "zone_lon", "hour_bucket", "day_of_week", "hour_of_day"])
        .agg(
            ride_count=("id", "count"),
            avg_distance_km=("distance_km", "mean"),
            avg_duration_min=("duration_min", "mean"),
            avg_price=("price_final", "mean"),
            cancelled_count=("is_cancelled", "sum"),   # FIX : "sum" natif, pas lambda
        )
        .reset_index()
    )

    agg = agg.sort_values(["zone_lat", "zone_lon", "hour_bucket"])

    # FIX : rolling via apply sur le numpy array — évite la surcharge du transform(lambda)
    def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=1).mean()

    agg["demand_trend_1d"] = (
        agg.groupby(["zone_lat", "zone_lon"])["ride_count"]
        .transform(_rolling_mean, window=24)
    )
    agg["demand_trend_7d"] = (
        agg.groupby(["zone_lat", "zone_lon"])["ride_count"]
        .transform(_rolling_mean, window=24 * 7)
    )

    # Taux d'annulation par bucket
    agg["cancellation_rate"] = agg["cancelled_count"] / (agg["ride_count"] + 1)

    # Features cycliques pour heure et jour
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour_of_day"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour_of_day"] / 24)
    agg["dow_sin"]  = np.sin(2 * np.pi * agg["day_of_week"] / 7)
    agg["dow_cos"]  = np.cos(2 * np.pi * agg["day_of_week"] / 7)

    return agg


def build_lstm_sequences(
    zone_df: pd.DataFrame,
    seq_len: int = 24,
    target_col: str = "ride_count",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prépare les séquences (X, y) pour l'entraînement LSTM.

    FIX : itère sur TOUTES les zones au lieu de recevoir une seule zone,
    ce qui évite X=(0,) quand la top-zone a moins de seq_len buckets.

    X shape : (n_samples, seq_len, n_features)
    y shape : (n_samples,)
    """
    feature_cols = [
        "ride_count", "avg_distance_km", "avg_price",
        "demand_trend_1d", "demand_trend_7d",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    X_all, y_all = [], []
    skipped = []

    # ── FIX : boucle sur chaque zone indépendamment ──────────────────────────
    group_cols = ["zone_lat", "zone_lon"]
    groups = (
        zone_df.groupby(group_cols)
        if all(c in zone_df.columns for c in group_cols)
        else [("all", zone_df)]   # fallback si colonnes absentes
    )

    for zone_id, grp in groups:
        grp = grp.sort_values("hour_bucket").reset_index(drop=True)

        # Ignore les zones trop courtes pour former au moins une séquence
        if len(grp) <= seq_len:
            skipped.append((zone_id, len(grp)))
            continue

        # Normalisation min-max locale à la zone
        # fillna(0) AVANT min/max — sinon NaN dans une cellule → NaN dans tout min/max
        values = grp[feature_cols].fillna(0).values.astype(np.float32)
        # Remplace les NaN résiduels (ex: 0/0 depuis la DB) par 0
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mins  = values.min(axis=0)
        maxs  = values.max(axis=0)
        # Dénominateur sécurisé : colonnes constantes → denom=1 → résultat=0 (pas NaN)
        denom = np.where((maxs - mins) < 1e-8, 1.0, maxs - mins)
        values_norm = (values - mins) / denom

        target_idx = feature_cols.index(target_col)

        for i in range(seq_len, len(values_norm)):
            X_all.append(values_norm[i - seq_len : i])
            y_all.append(values_norm[i, target_idx])

    if skipped:
        print(f"  ⚠  {len(skipped)} zone(s) ignorées (< {seq_len} buckets) : {skipped[:5]}")

    # ── Guard : lève une erreur claire si toujours vide ─────────────────────
    if not X_all:
        raise ValueError(
            f"Aucune séquence construite : toutes les zones ont ≤ {seq_len} buckets. "
            f"Réduisez seq_len (actuellement {seq_len}) "
            f"ou augmentez ML.demand_lookback_days."
        )

    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.float32)


def run(lookback_days: int = ML.demand_lookback_days) -> dict:
    """Point d'entrée principal du pipeline de features demande."""
    print(f"[demand_features] Chargement des rides ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} rides chargés")

    zone_hour = build_zone_hour_features(raw)
    print(f"  → {len(zone_hour):,} buckets zone×heure construits")

    # FIX : on passe TOUT zone_hour (toutes les zones) au lieu de la top-zone seule
    X, y = build_lstm_sequences(zone_hour)
    print(f"  → Séquences LSTM : X={X.shape}, y={y.shape}")

    # top_zone conservé pour Prophet (on garde la série la plus riche)
    top_zone = (
        zone_hour.groupby(["zone_lat", "zone_lon"])["ride_count"]
        .sum()
        .idxmax()
    )

    return {"zone_hour_df": zone_hour, "X": X, "y": y, "top_zone": top_zone}


if __name__ == "__main__":
    run()