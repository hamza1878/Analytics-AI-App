"""
scheduler/daily_runner.py
Moviroo — Scheduler automatique (exécution quotidienne à 02:00)
Lance tous les pipelines ML dans l'ordre correct.

Exécution manuelle  : python -m scheduler.daily_runner
Exécution scheduled : python -m scheduler.daily_runner --daemon

Corrections :
  - timeout=3600 sur chaque subprocess (évite hang infini)
  - SIGINT ignoré dans le runner principal → Ctrl+C ne tue pas les subprocesses
  - capture_output=True + logs écrits en temps réel via thread (non-bloquant)
  - step qui dépasse le timeout : marqué "timeout" et pipeline continue
  - FIX: signal.signal() ignoré si appelé depuis un thread secondaire
    (ex. BackgroundTask FastAPI) pour éviter ValueError
"""
import sys, time, logging, json, subprocess, os, signal, threading
from datetime import datetime

# ── UTF-8 logging handler ─────────────────────────────────────────────────────
# Force UTF-8 sur le handler logging du runner lui-même.
# StreamHandler hérite l'encodage du terminal (cp1252 sous PowerShell) ;
# on le remplace explicitement par un handler UTF-8 avec errors='replace'.
import io as _io

_utf8_handler = logging.StreamHandler(
    _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "buffer") else sys.stdout
)
_utf8_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(
    level=logging.INFO,
    handlers=[_utf8_handler],
)
log = logging.getLogger("scheduler.daily_runner")


# ── Ordre des pipelines à exécuter chaque jour ────────────────────────────────
PIPELINE_STEPS = [
    {
        "id":      "feature_engineering",
        "name":    "Feature Engineering Pipeline",
        "cmd":     [sys.executable, "-m", "feature_engineering.pipeline"],
        "timeout": 1800,   # 30 min max
    },
    {
        "id":      "surge_predictor",
        "name":    "Surge Predictor (US-3.3) — XGBoost + LightGBM",
        "cmd":     [sys.executable, "-m", "models.surge_predictor"],
        "timeout": 1800,
    },
    {
        "id":      "demand_forecast",
        "name":    "Demand Forecast — LSTM + Prophet",
        "cmd":     [sys.executable, "-m", "models.demand_forecast"],
        "timeout": 3600,   # 60 min max (LSTM peut être long)
    },
    {
        "id":      "churn_classifier",
        "name":    "Churn Classifier — XGBoost",
        "cmd":     [sys.executable, "-m", "models.churn_classifier"],
        "timeout": 1800,
    },
    {
        "id":      "eta_estimator",
        "name":    "ETA Estimator — Gradient Boosting",
        "cmd":     [sys.executable, "-m", "models.eta_estimator"],
        "timeout": 1800,
    },
    {
        "id":      "fraud_detector",
        "name":    "Fraud Detector — Isolation Forest",
        "cmd":     [sys.executable, "-m", "models.fraud_detector"],
        "timeout": 1800,
    },
]

STATUS_FILE = "data/pipeline_status.json"


# ── Sauvegarde statut (lu par le frontend) ────────────────────────────────────
def save_status(run_id: str, steps_results: list, running_step=None):
    os.makedirs("data", exist_ok=True)
    payload = {
        "run_id":       run_id,
        "started_at":   run_id,
        "updated_at":   datetime.utcnow().isoformat(),
        "running_step": running_step,
        "steps":        steps_results,
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


# ── Stream stdout/stderr d'un subprocess vers le logger en temps réel ─────────
def _stream_output(pipe, label: str):
    """Lit un pipe ligne par ligne et log chaque ligne. Tourne dans un thread."""
    try:
        for line in iter(pipe.readline, ""):
            line = line.rstrip("\n")
            if line:
                log.info("    [%s] %s", label, line)
    except Exception:
        pass
    finally:
        pipe.close()


# ── Exécution d'un step ───────────────────────────────────────────────────────
def run_step(step: dict, run_id: str, steps_results: list) -> dict:
    entry = {
        "id":          step["id"],
        "name":        step["name"],
        "status":      "running",
        "started_at":  datetime.utcnow().isoformat(),
        "finished_at": None,
        "duration_s":  None,
        "error":       None,
    }
    steps_results.append(entry)
    save_status(run_id, steps_results, running_step=step["id"])

    timeout = step.get("timeout", 3600)
    log.info("▶  %s (timeout=%ds)…", step["name"], timeout)
    t0 = time.time()

    try:
        # Force UTF-8 sur stdout/stderr de chaque subprocess enfant.
        # Sans ça, Python sur Windows hérite cp1252 du terminal PowerShell et
        # plante sur tout caractère Unicode dans print() ou MLflow.
        # PYTHONUTF8=1       -> Python ouvre stdout en UTF-8 (PEP 540)
        # PYTHONIOENCODING   -> fallback pour les libs qui lisent cette var
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            step["cmd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr -> stdout
            text=True,
            encoding="utf-8",           # le pipe lu par le runner est aussi UTF-8
            errors="replace",           # caractère illisible -> ? au lieu de crash
            env=child_env,
            cwd=os.getcwd(),
        )

        # Thread daemon : stream stdout sans bloquer le runner
        t = threading.Thread(
            target=_stream_output,
            args=(proc.stdout, step["id"]),
            daemon=True,
        )
        t.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = round(time.time() - t0, 1)
            entry.update({
                "status":      "timeout",
                "finished_at": datetime.utcnow().isoformat(),
                "duration_s":  elapsed,
                "error":       f"Dépassement du timeout ({timeout}s)",
            })
            log.error("  ⏱  %s TIMEOUT après %ss", step["name"], elapsed)
            save_status(run_id, steps_results)
            return entry

        elapsed = round(time.time() - t0, 1)
        t.join(timeout=5)   # attend max 5s que le thread finisse de vider le buffer

        if proc.returncode == 0:
            entry.update({
                "status":      "success",
                "finished_at": datetime.utcnow().isoformat(),
                "duration_s":  elapsed,
            })
            log.info("  ✓  %s terminé en %ss", step["name"], elapsed)
        else:
            entry.update({
                "status":      "failed",
                "finished_at": datetime.utcnow().isoformat(),
                "duration_s":  elapsed,
                "error":       f"returncode={proc.returncode}",
            })
            log.error("  ✗  %s ÉCHOUÉ (code %d) en %ss",
                      step["name"], proc.returncode, elapsed)

    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        entry.update({
            "status":      "failed",
            "finished_at": datetime.utcnow().isoformat(),
            "duration_s":  elapsed,
            "error":       str(exc),
        })
        log.error("  ✗  EXCEPTION dans %s : %s", step["name"], exc)

    save_status(run_id, steps_results, running_step=None)
    return entry


# ── Fonction principale : une exécution complète ─────────────────────────────
def run_all_pipelines() -> dict:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  MOVIROO ML — Exécution complète [%s]  ║", run_id)
    log.info("╚══════════════════════════════════════════════════╝")

    steps_results = []
    t_total = time.time()

    # ── FIX : signal.signal() lève ValueError si appelé depuis un thread
    # secondaire (ex. BackgroundTask de FastAPI via anyio/starlette).
    # On détecte si on est dans le thread principal et on n'installe le handler
    # SIGINT que dans ce cas ; sinon on continue sans protection Ctrl+C
    # (ce qui est correct : dans un thread worker, Ctrl+C n'a pas de sens).
    _is_main_thread = threading.current_thread() is threading.main_thread()

    if _is_main_thread:
        _original_sigint = signal.getsignal(signal.SIGINT)
        _ctrl_c_count = [0]

        def _sigint_handler(sig, frame):
            _ctrl_c_count[0] += 1
            if _ctrl_c_count[0] == 1:
                log.warning(
                    "Ctrl+C reçu — le step en cours se terminera proprement. "
                    "Appuyez à nouveau pour forcer l'arrêt."
                )
            else:
                log.warning("Forçage de l'arrêt.")
                signal.signal(signal.SIGINT, _original_sigint)
                raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _sigint_handler)
    else:
        log.debug(
            "run_all_pipelines() appelé depuis un thread secondaire — "
            "handler SIGINT non installé (normal en mode BackgroundTask)."
        )

    try:
        for step in PIPELINE_STEPS:
            run_step(step, run_id, steps_results)
    finally:
        # Restaure le handler original uniquement si on l'avait remplacé
        if _is_main_thread:
            signal.signal(signal.SIGINT, _original_sigint)

    total   = round(time.time() - t_total, 1)
    success = sum(1 for s in steps_results if s["status"] == "success")
    failed  = sum(1 for s in steps_results if s["status"] in ("failed", "timeout"))

    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  RÉSUMÉ : %d/%d réussis | durée totale : %ss",
             success, len(PIPELINE_STEPS), total)
    for s in steps_results:
        icon = ("✓" if s["status"] == "success"
                else ("⏱" if s["status"] == "timeout" else "✗"))
        log.info("║  %s  %-40s %ss", icon, s["name"][:40], s.get("duration_s", "?"))
    log.info("╚══════════════════════════════════════════════════╝")

    save_status(run_id, steps_results, running_step=None)
    return {
        "run_id":     run_id,
        "success":    success,
        "failed":     failed,
        "duration_s": total,
        "steps":      steps_results,
    }


# ── Mode daemon : exécution à 02:00 chaque jour ───────────────────────────────
def run_daemon():
    import schedule
    log.info("Daemon démarré — exécution planifiée tous les jours à 02:00")
    schedule.every().day.at("02:00").do(run_all_pipelines)
    log.info("Exécution initiale au démarrage…")
    run_all_pipelines()
    while True:
        schedule.run_pending()
        time.sleep(30)


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--daemon" in sys.argv:
        run_daemon()
    else:
        result = run_all_pipelines()
        sys.exit(0 if result["failed"] == 0 else 1)