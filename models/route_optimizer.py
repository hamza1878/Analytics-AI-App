"""
models/route_optimizer.py

Agent RL (DQN simplifié) pour optimiser le dispatch des chauffeurs.
Source : dispatch_offers (score, distance_to_pickup_km, statuts).
Cible : taux d'acceptation > 90%, temps de prise en charge < 5 min

Enums réels PostgreSQL :
  ride_status_enum           : PENDING, SEARCHING_DRIVER, ASSIGNED,
                               EN_ROUTE_TO_PICKUP, ARRIVED, IN_TRIP,
                               COMPLETED, CANCELLED
  driver_availability_status : pending, setup_required, offline,
                               online, on_trip
"""
import os
import numpy as np
import pandas as pd
import mlflow
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from config import DB, ML

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────
# Enum value sets  (single source of truth)
# ─────────────────────────────────────────────

# ride_status_enum values that mean the driver accepted the dispatch
DISPATCH_ACCEPTED_STATUSES = {
    "ASSIGNED",
    "EN_ROUTE_TO_PICKUP",
    "ARRIVED",
    "IN_TRIP",
    "COMPLETED",
}

# ride_status_enum values that mean no driver was found / offer failed
DISPATCH_REJECTED_STATUSES = {
    "SEARCHING_DRIVER",   # offer broadcast but nobody took it yet
}

# ride_status_enum values that mean the ride was abandoned
DISPATCH_CANCELLED_STATUSES = {
    "CANCELLED",
}

# driver_availability_status values considered "available for dispatch"
DRIVER_AVAILABLE_STATUSES = {
    "online",
}

# driver_availability_status values that should be excluded from queries
DRIVER_EXCLUDED_STATUSES = {
    "pending",
    "setup_required",
    "offline",
    "on_trip",
}


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

DISPATCH_QUERY = """
SELECT
    d_o.id                                                      AS offer_id,
    d_o.ride_id,
    d_o.driver_id,
    d_o.status,
    d_o.distance_to_pickup_km,
    d_o.score                                                   AS dispatch_score,

    EXTRACT(EPOCH FROM (d_o.expires_at - d_o.offered_at)) / 60
                                                                AS offer_window_min,

    r.pickup_lat,
    r.pickup_lon,
    r.class_id,
    r.distance_km                                               AS trip_distance_km,
    r.surge_multiplier,

    d.rating_average                                            AS driver_rating,
    d.total_trips                                               AS driver_total_trips,
    d.availability_status                                       AS driver_availability,

    dl.latitude                                                 AS driver_lat,
    dl.longitude                                                AS driver_lon,
    dl.speed_kmh                                                AS driver_speed

FROM dispatch_offers d_o
JOIN rides   r  ON r.id      = d_o.ride_id
JOIN drivers d  ON d.user_id = d_o.driver_id
LEFT JOIN driver_locations dl ON dl.driver_id = d_o.driver_id

WHERE d_o.offered_at >= NOW() - INTERVAL '{lookback} days'
  AND r.status = ANY(ARRAY[
        'PENDING',
        'SEARCHING_DRIVER',
        'ASSIGNED',
        'EN_ROUTE_TO_PICKUP',
        'ARRIVED',
        'IN_TRIP',
        'COMPLETED',
        'CANCELLED'
      ]::ride_status_enum[])
ORDER BY d_o.offered_at;
"""


def load_raw(lookback_days: int = 30) -> pd.DataFrame:
    engine = create_engine(DB.url)

    def _fetch(days: int) -> pd.DataFrame:
        query = DISPATCH_QUERY.format(lookback=days)
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    # ── Essai avec la fenêtre demandée ────────────────────────────────────
    df = _fetch(lookback_days)
    print(f"[route_optimizer] {len(df)} lignes chargées (fenêtre {lookback_days}j)")

    # ── Fallback automatique vers 1 an si vide ────────────────────────────
    if len(df) == 0 and lookback_days < 365:
        print(
            f"[route_optimizer] ⚠️  Aucune donnée sur {lookback_days}j — "
            f"fallback sur 365 jours..."
        )
        df = _fetch(365)
        print(f"[route_optimizer] {len(df)} lignes chargées (fenêtre 365j)")

    if len(df) == 0:
        print(
            "[route_optimizer] ❌ Aucune donnée même sur 365j. "
            "Vérifiez la base de données."
        )
    else:
        print(
            f"[route_optimizer] Distribution statuts rides :\n"
            f"{df['status'].value_counts().to_string()}"
        )
        if "driver_availability" in df.columns:
            print(
                f"[route_optimizer] Distribution statuts drivers :\n"
                f"{df['driver_availability'].value_counts().to_string()}"
            )
        missing = [c for c in STATE_COLS if c not in df.columns]
        if missing:
            print(f"[route_optimizer] ⚠️  Colonnes manquantes : {missing}")

    return df


# ─────────────────────────────────────────────
# State / Action space
# ─────────────────────────────────────────────

STATE_COLS = [
    "distance_to_pickup_km",
    "trip_distance_km",
    "surge_multiplier",
    "driver_rating",
    "driver_total_trips",
    "driver_speed",
    "offer_window_min",
    "dispatch_score",
]

# Actions : 0 = ne pas dispatcher, 1 = dispatcher
N_ACTIONS = 2
STATE_DIM  = len(STATE_COLS)   # 8


@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


def compute_reward(row: pd.Series) -> float:
    """
    Reward function basée sur le statut réel de la course :

      ASSIGNED / EN_ROUTE_TO_PICKUP / ARRIVED / IN_TRIP / COMPLETED
          → chauffeur a accepté  (+1.0 + bonus pickup rapide)
      CANCELLED
          → course annulée       (-0.8)
      SEARCHING_DRIVER
          → aucun chauffeur trouvé / offre expirée (-0.3)
      PENDING
          → offre encore en attente (neutre, 0.0)
    """
    # Normalise en majuscules pour comparer de façon fiable
    status = str(row["status"]).upper()

    if status in DISPATCH_ACCEPTED_STATUSES:
        # Bonus si le chauffeur est proche (distance < 10 km)
        pickup_bonus = max(0.0, 1.0 - row["distance_to_pickup_km"] / 10.0)
        return 1.0 + pickup_bonus

    if status in DISPATCH_CANCELLED_STATUSES:
        return -0.8

    if status in DISPATCH_REJECTED_STATUSES:
        return -0.3

    # PENDING ou valeur inconnue → neutre
    return 0.0


def build_transitions(df: pd.DataFrame) -> list[Transition]:
    """Convertit les offres historiques en transitions (s, a, r, s', done)."""
    if len(df) < 2:
        print("[route_optimizer] ⚠️  DataFrame trop petit (besoin ≥ 2 lignes).")
        return []

    # S'assurer que toutes les colonnes d'état existent
    for col in STATE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    df = df.copy().fillna(0)
    transitions = []

    for i in range(len(df) - 1):
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]

        state      = np.array([row[c]      for c in STATE_COLS], dtype=np.float32)
        next_state = np.array([next_row[c] for c in STATE_COLS], dtype=np.float32)

        # Action = 1 si le dispatch a abouti (statut "accepté"), 0 sinon
        action = 1 if str(row["status"]).upper() in DISPATCH_ACCEPTED_STATUSES else 0
        reward = compute_reward(row)
        done   = (i == len(df) - 2)

        transitions.append(Transition(state, action, reward, next_state, done))

    return transitions


# ─────────────────────────────────────────────
# DQN Network
# ─────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer:   list[Transition] = []
        self.capacity: int              = capacity

    def push(self, t: Transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self) -> int:
        return len(self.buffer)


def train_dqn(
    transitions: list[Transition],
    episodes:    int   = 10,
    batch_size:  int   = 64,
    gamma:       float = 0.99,
    lr:          float = 1e-3,
) -> dict:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch non disponible — pip install torch")

    if len(transitions) == 0:
        raise ValueError("[train_dqn] Aucune transition fournie.")

    # Vérification dimension état
    if transitions[0].state.shape[0] != STATE_DIM:
        raise ValueError(
            f"[train_dqn] Dimension état incorrecte : "
            f"attendu {STATE_DIM}, reçu {transitions[0].state.shape[0]}"
        )

    policy_net = DQN(STATE_DIM, N_ACTIONS)
    target_net = DQN(STATE_DIM, N_ACTIONS)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer    = ReplayBuffer()

    for t in transitions:
        buffer.push(t)

    losses  = []
    epsilon = 1.0

    for episode in range(episodes):
        episode_loss = 0.0

        if len(buffer) < batch_size:
            print(
                f"  Episode {episode+1}/{episodes} | ⚠️  buffer trop petit "
                f"({len(buffer)} < {batch_size}), skip."
            )
            continue

        steps = min(100, len(buffer) // batch_size)
        for _ in range(steps):
            batch = buffer.sample(batch_size)

            states      = torch.FloatTensor([t.state       for t in batch])
            next_states = torch.FloatTensor([t.next_state  for t in batch])
            actions     = torch.LongTensor( [t.action      for t in batch])
            rewards     = torch.FloatTensor([t.reward      for t in batch])
            dones       = torch.FloatTensor([float(t.done) for t in batch])

            q_values      = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = target_net(next_states).max(1)[0].detach()
            expected_q    = rewards + gamma * next_q_values * (1.0 - dones)

            loss = nn.HuberLoss()(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
            optimizer.step()
            episode_loss += loss.item()

        # Soft update target network (τ = 0.005)
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.copy_(0.005 * pp.data + 0.995 * tp.data)

        epsilon = max(0.05, epsilon * 0.95)
        losses.append(episode_loss)
        print(
            f"  Episode {episode+1}/{episodes} | "
            f"loss={episode_loss:.4f} | ε={epsilon:.3f}"
        )

    # ── Évaluation finale ─────────────────────────────────────────────────
    policy_net.eval()
    all_states = torch.FloatTensor(np.stack([t.state for t in transitions]))
    with torch.no_grad():
        pred_actions = policy_net(all_states).argmax(dim=1).numpy()

    true_actions = np.array([t.action for t in transitions])
    accuracy     = float((pred_actions == true_actions).mean())

    for label, name in [(1, "DISPATCH"), (0, "SKIP")]:
        mask = true_actions == label
        if mask.sum() > 0:
            acc = float((pred_actions[mask] == true_actions[mask]).mean())
            print(f"  Accuracy {name}: {acc:.1%}  (n={mask.sum()})")

    return {
        "model":    policy_net,
        "accuracy": accuracy,
        "losses":   losses,
    }


# ─────────────────────────────────────────────
# MLflow training entry point
# ─────────────────────────────────────────────

def train_and_log(df: pd.DataFrame) -> str:
    print(f"[route_optimizer] Lignes reçues : {len(df)}")

    if len(df) == 0:
        print("[route_optimizer] ❌ DataFrame vide — abandon.")
        return ""

    transitions = build_transitions(df)
    n = len(transitions)
    print(f"[route_optimizer] {n} transitions construites")

    # ── Adapte batch_size et seuil minimum à la taille des données ────────
    MIN_TRANSITIONS = 8
    if n < MIN_TRANSITIONS:
        print(
            f"[route_optimizer] ❌ Pas assez de transitions "
            f"({n} < {MIN_TRANSITIONS}) — abandon."
        )
        return ""

    # batch_size = plus grande puissance de 2 ≤ n/2, min 4
    batch_size = max(4, 2 ** int(np.log2(n // 2)))
    print(
        f"[route_optimizer] batch_size adapté : {batch_size}  "
        f"(données : {n} transitions)"
    )
    if n < 64:
        print(
            f"[route_optimizer] ⚠️  Données limitées ({n} lignes) — "
            f"résultats indicatifs uniquement. Idéal : ≥ 500 lignes."
        )

    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="route_optimizer_v1") as run:
        result = train_dqn(transitions, batch_size=batch_size)

        mlflow.log_params({
            "model":         "DQN",
            "state_dim":     STATE_DIM,
            "n_actions":     N_ACTIONS,
            "gamma":         0.99,
            "lr":            1e-3,
            "episodes":      10,
            "batch_size":    batch_size,
            "n_transitions": n,
        })
        mlflow.log_metrics({
            "dispatch_accuracy": result["accuracy"],
            "final_loss":        result["losses"][-1] if result["losses"] else 0.0,
        })

        for i, loss_val in enumerate(result["losses"]):
            mlflow.log_metric("episode_loss", loss_val, step=i)

        if TORCH_AVAILABLE:
            tmp_path = os.path.join(os.environ.get("TEMP", "/tmp"), "route_dqn.pt")
            torch.save(result["model"].state_dict(), tmp_path)
            mlflow.log_artifact(tmp_path)

        print(f"[route_optimizer] ✅ Accuracy={result['accuracy']:.1%}")
        print(
            f"🏃 Run : {ML.mlflow_tracking_uri}/#/experiments/"
            f"{run.info.experiment_id}/runs/{run.info.run_id}"
        )

        return run.info.run_id


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    raw = load_raw(lookback_days=30)   # essaie 30j → fallback auto 365j si vide

    if len(raw) == 0:
        print("\n[route_optimizer] Aucune donnée chargée même sur 365j.")
        print("Vérifiez :")
        print("  1. Que dispatch_offers contient des lignes")
        print("  2. Que DB.url pointe vers la bonne base")
        print("  3. Que les JOINs (rides, drivers) matchent")
    else:
        train_and_log(raw)