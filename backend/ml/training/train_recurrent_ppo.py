"""Training script voor RecurrentPPO portfolio agent.

Curriculum learning in 2 fasen:
  Fase 1: 2 assets, 60-dagen episodes, 400K timesteps → basisbegrip
  Fase 2: Volledige universe, 252-dagen episodes, 1.6M timesteps → fine-tuning

Features:
  - Synthetische data generatie voor training
  - Callback met Sortino-tracking en early stopping
  - Model save met complete metadata
  - Evaluatie na elke fase
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Paden ───────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent.parent / "models"
_RPPO_MODEL_PATH = _MODEL_DIR / "recurrent_ppo"
_RPPO_METADATA_PATH = _MODEL_DIR / "recurrent_ppo_metadata.json"
_TRAINING_REPORT_PATH = _MODEL_DIR / "recurrent_ppo_training_report.json"


def generate_synthetic_data(
    n_timesteps: int = 2000,
    n_assets: int = 10,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Genereer synthetische marktdata voor training.

    Creëert realistische patronen met regimewisselingen,
    mean-reversion, en correlati-structuur.

    Returns:
        Dict met signal_data, price_data, regime_data, sentiment_data,
        vix_data, asset_types
    """
    rng = np.random.default_rng(seed)

    # ── Regimes (bull/bear/sideways/high_vol) ───────────────────────
    regime_data = np.zeros((n_timesteps, 4))
    regime_state = 1  # start sideways
    for t in range(n_timesteps):
        if rng.random() < 0.02:  # ~2% kans op regime-switch per dag
            regime_state = rng.integers(0, 4)
        regime_data[t, regime_state] = 1.0

    # ── Prijs data met regime-afhankelijke returns ──────────────────
    price_data = np.ones((n_timesteps, n_assets))
    regime_drift = {0: -0.001, 1: 0.0, 2: 0.001, 3: 0.0}  # bear/side/bull/highvol
    regime_vol = {0: 0.025, 1: 0.015, 2: 0.02, 3: 0.035}

    for i in range(n_assets):
        price = 100.0 + rng.normal(0, 20)  # random startprijs
        for t in range(1, n_timesteps):
            regime_idx = int(np.argmax(regime_data[t]))
            drift = regime_drift[regime_idx]
            vol = regime_vol[regime_idx]
            # Asset-specifieke volatiliteit
            asset_vol = vol * (0.8 + 0.4 * (i / n_assets))
            ret = drift + rng.normal(0, asset_vol)
            price *= (1 + ret)
            price_data[t, i] = price

    # ── Signal data (confidence, exp_return, risk) ──────────────────
    signal_data = np.zeros((n_timesteps, n_assets, 3))
    for t in range(n_timesteps):
        regime_idx = int(np.argmax(regime_data[t]))
        for i in range(n_assets):
            if t > 0:
                actual_return = (price_data[t, i] - price_data[t - 1, i]) / price_data[t - 1, i]
            else:
                actual_return = 0.0

            # Noisy signal dat gecorreleerd is met toekomstige return
            noise = rng.normal(0, 0.02)
            signal_data[t, i, 0] = np.clip(0.5 + abs(actual_return) * 10 + noise, 0.1, 0.99)  # confidence
            signal_data[t, i, 1] = actual_return + noise  # exp_return (noisy)
            signal_data[t, i, 2] = np.clip(0.3 + regime_vol[regime_idx] * 10 + rng.normal(0, 0.1), 0.1, 0.9)  # risk

    # ── Sentiment data ──────────────────────────────────────────────
    sentiment_data = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        regime_idx = int(np.argmax(regime_data[t]))
        base_sentiment = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.4}[regime_idx]
        sentiment_data[t] = np.clip(base_sentiment + rng.normal(0, 0.1), 0.0, 1.0)

    # ── VIX data ────────────────────────────────────────────────────
    vix_data = np.zeros(n_timesteps)
    vix = 20.0
    for t in range(n_timesteps):
        regime_idx = int(np.argmax(regime_data[t]))
        target_vix = {0: 28.0, 1: 18.0, 2: 15.0, 3: 35.0}[regime_idx]
        vix = 0.95 * vix + 0.05 * target_vix + rng.normal(0, 1.0)
        vix_data[t] = max(10.0, vix)

    # ── Asset types: helft crypto, helft stocks ─────────────────────
    asset_types = ["crypto" if i < n_assets // 2 else "stock" for i in range(n_assets)]

    return {
        "signal_data": signal_data.astype(np.float32),
        "price_data": price_data.astype(np.float32),
        "regime_data": regime_data.astype(np.float32),
        "sentiment_data": sentiment_data.astype(np.float32),
        "vix_data": vix_data.astype(np.float32),
        "asset_types": asset_types,
    }


def _make_eval_callback(
    eval_env: Any,
    best_model_path: str,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
) -> Any:
    """Maak een evaluatie-callback (indien sb3 beschikbaar)."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class SortinoEvalCallback(BaseCallback):
            """Custom callback die Sortino ratio bijhoudt."""

            def __init__(self, eval_env: Any, best_path: str, freq: int, n_eps: int):
                super().__init__(verbose=0)
                self.eval_env = eval_env
                self.best_path = best_path
                self.freq = freq
                self.n_eps = n_eps
                self.best_sortino: float = -np.inf
                self.eval_history: list = []

            def _on_step(self) -> bool:
                if self.n_calls % self.freq != 0:
                    return True

                # Evalueer
                returns = []
                for _ in range(self.n_eps):
                    obs, _ = self.eval_env.reset()
                    done = False
                    ep_values = [1.0]

                    # LSTM state handling
                    lstm_states = None
                    episode_starts = np.array([True])

                    while not done:
                        try:
                            action, lstm_states = self.model.predict(
                                obs.reshape(1, -1),
                                state=lstm_states,
                                episode_start=episode_starts,
                                deterministic=True,
                            )
                            episode_starts = np.array([False])
                        except TypeError:
                            action, _ = self.model.predict(
                                obs.reshape(1, -1),
                                deterministic=True,
                            )

                        obs, _, terminated, truncated, info = self.eval_env.step(
                            action.flatten(),
                        )
                        done = terminated or truncated
                        if "portfolio_value" in info:
                            ep_values.append(info["portfolio_value"])

                    ep_return = (ep_values[-1] / ep_values[0] - 1) * 100
                    returns.append(ep_return)

                mean_return = np.mean(returns)
                pct_returns = np.diff(returns) / (np.abs(np.array(returns[:-1])) + 1e-10) if len(returns) > 1 else [0]
                downside = [r for r in pct_returns if r < 0]
                ds_std = float(np.std(downside)) if len(downside) > 1 else 0.01
                sortino = float(np.mean(pct_returns)) / (ds_std + 1e-8) if len(pct_returns) > 0 else 0.0

                self.eval_history.append({
                    "timestep": self.n_calls,
                    "mean_return": round(float(mean_return), 2),
                    "sortino": round(sortino, 4),
                })

                if sortino > self.best_sortino:
                    self.best_sortino = sortino
                    self.model.save(self.best_path)
                    logger.info(
                        "Nieuw beste model bij stap %d — Sortino=%.4f, Return=%.2f%%",
                        self.n_calls, sortino, mean_return,
                    )

                return True

        return SortinoEvalCallback(eval_env, best_model_path, eval_freq, n_eval_episodes)

    except ImportError:
        return None


def train_recurrent_ppo(
    data: Optional[Dict[str, np.ndarray]] = None,
    total_timesteps: int = _RPPO_MODEL_PATH and 2_000_000,
    phase1_timesteps: int = 400_000,
    n_signals: int = 10,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train RecurrentPPO agent met curriculum learning.

    Fase 1: 2 assets, 60 dagen, 400K stappen
    Fase 2: Vol universe, 252 dagen, resterende stappen

    Args:
        data: voorbewerkte data dict (of None voor synthetics)
        total_timesteps: totaal training stappen
        phase1_timesteps: stappen voor fase 1
        n_signals: aantal assets in volledige modus
        learning_rate: learning rate
        seed: random seed

    Returns:
        Training report dict
    """
    from backend.ml.trading_env_v2 import TradingEnvV2

    logger.info("=" * 60)
    logger.info("RecurrentPPO Training Start")
    logger.info("=" * 60)
    start_time = time.time()

    # ── Data ────────────────────────────────────────────────────────
    if data is None:
        logger.info("Genereer synthetische training data...")
        data = generate_synthetic_data(n_timesteps=3000, n_assets=n_signals, seed=seed)

    signal_data = data["signal_data"]
    price_data = data["price_data"]
    regime_data = data.get("regime_data")
    sentiment_data = data.get("sentiment_data")
    vix_data = data.get("vix_data")
    asset_types = data.get("asset_types")

    report: Dict[str, Any] = {
        "model": "RecurrentPPO",
        "started_at": datetime.now().isoformat(),
        "n_assets": signal_data.shape[1],
        "n_timesteps_data": len(signal_data),
        "total_timesteps": total_timesteps,
        "phases": {},
    }

    # ── Detecteer beschikbare library ───────────────────────────────
    use_recurrent = False
    try:
        from sb3_contrib import RecurrentPPO
        use_recurrent = True
        logger.info("sb3-contrib gevonden — gebruik RecurrentPPO (LSTM)")
    except ImportError:
        logger.warning("sb3-contrib niet beschikbaar — gebruik standaard PPO")
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("Noch sb3-contrib noch stable-baselines3 beschikbaar!")
            report["error"] = "Geen RL library beschikbaar"
            return report

    # ══════════════════════════════════════════════════════════════════
    # FASE 1: Simpele omgeving, 2 assets, 60 dagen
    # ══════════════════════════════════════════════════════════════════
    logger.info("─" * 40)
    logger.info("FASE 1: Curriculum — 2 assets, 60 dagen")
    logger.info("─" * 40)

    env_p1 = TradingEnvV2(
        signal_data=signal_data,
        price_data=price_data,
        regime_data=regime_data,
        sentiment_data=sentiment_data,
        vix_data=vix_data,
        asset_types=asset_types,
        n_signals=n_signals,
        curriculum_phase=1,
    )

    # Model aanmaken
    common_kwargs = dict(
        env=env_p1,
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        seed=seed,
    )

    if use_recurrent:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            policy_kwargs={
                "lstm_hidden_size": 128,
                "n_lstm_layers": 1,
                "enable_critic_lstm": True,
            },
            **common_kwargs,
        )
    else:
        model = PPO(policy="MlpPolicy", **common_kwargs)

    # Evaluatie env (zelfde als training voor fase 1)
    eval_env_p1 = TradingEnvV2(
        signal_data=signal_data,
        price_data=price_data,
        regime_data=regime_data,
        sentiment_data=sentiment_data,
        vix_data=vix_data,
        asset_types=asset_types,
        n_signals=n_signals,
        curriculum_phase=1,
    )

    callback_p1 = _make_eval_callback(
        eval_env_p1,
        best_model_path=str(_RPPO_MODEL_PATH) + "_phase1_best",
        eval_freq=20_000,
    )

    logger.info("Start fase 1 training: %d timesteps...", phase1_timesteps)
    p1_start = time.time()
    model.learn(
        total_timesteps=phase1_timesteps,
        callback=callback_p1,
        progress_bar=False,
    )
    p1_time = time.time() - p1_start

    # Evalueer fase 1
    p1_eval = _evaluate_model(model, eval_env_p1, use_recurrent, n_episodes=5)
    report["phases"]["phase1"] = {
        "timesteps": phase1_timesteps,
        "duration_sec": round(p1_time, 1),
        "n_assets": env_p1.n_signals,
        "episode_length": env_p1.episode_length,
        "eval": p1_eval,
    }
    logger.info("Fase 1 klaar in %.1fs — Return: %.2f%%, Sortino: %.4f",
                p1_time, p1_eval["mean_return_pct"], p1_eval["mean_sortino"])

    # ══════════════════════════════════════════════════════════════════
    # FASE 2: Volledige omgeving, N assets, 252 dagen
    # ══════════════════════════════════════════════════════════════════
    logger.info("─" * 40)
    logger.info("FASE 2: Volledig — %d assets, 252 dagen", n_signals)
    logger.info("─" * 40)

    env_p2 = TradingEnvV2(
        signal_data=signal_data,
        price_data=price_data,
        regime_data=regime_data,
        sentiment_data=sentiment_data,
        vix_data=vix_data,
        asset_types=asset_types,
        n_signals=n_signals,
        curriculum_phase=2,
    )

    eval_env_p2 = TradingEnvV2(
        signal_data=signal_data,
        price_data=price_data,
        regime_data=regime_data,
        sentiment_data=sentiment_data,
        vix_data=vix_data,
        asset_types=asset_types,
        n_signals=n_signals,
        curriculum_phase=2,
    )

    # Zet model om naar volledige env (set_env)
    model.set_env(env_p2)

    callback_p2 = _make_eval_callback(
        eval_env_p2,
        best_model_path=str(_RPPO_MODEL_PATH) + "_best",
        eval_freq=50_000,
    )

    phase2_timesteps = total_timesteps - phase1_timesteps
    logger.info("Start fase 2 training: %d timesteps...", phase2_timesteps)
    p2_start = time.time()
    model.learn(
        total_timesteps=phase2_timesteps,
        callback=callback_p2,
        progress_bar=False,
        reset_num_timesteps=False,
    )
    p2_time = time.time() - p2_start

    # Evalueer fase 2
    p2_eval = _evaluate_model(model, eval_env_p2, use_recurrent, n_episodes=10)
    report["phases"]["phase2"] = {
        "timesteps": phase2_timesteps,
        "duration_sec": round(p2_time, 1),
        "n_assets": env_p2.n_signals,
        "episode_length": env_p2.episode_length,
        "eval": p2_eval,
    }
    logger.info("Fase 2 klaar in %.1fs — Return: %.2f%%, Sortino: %.4f",
                p2_time, p2_eval["mean_return_pct"], p2_eval["mean_sortino"])

    # ── Model opslaan ───────────────────────────────────────────────
    os.makedirs(_MODEL_DIR, exist_ok=True)
    model.save(str(_RPPO_MODEL_PATH))
    logger.info("Model opgeslagen: %s", _RPPO_MODEL_PATH)

    # Metadata
    total_time = time.time() - start_time
    metadata = {
        "model_type": "RecurrentPPO" if use_recurrent else "PPO",
        "policy": "MlpLstmPolicy" if use_recurrent else "MlpPolicy",
        "trained_at": datetime.now().isoformat(),
        "total_timesteps": total_timesteps,
        "total_training_time_sec": round(total_time, 1),
        "n_signals": n_signals,
        "obs_dim": env_p2._obs_dim,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "lstm_hidden_size": 128 if use_recurrent else None,
        },
        "final_eval": p2_eval,
        "seed": seed,
    }
    with open(_RPPO_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Report
    report["completed_at"] = datetime.now().isoformat()
    report["total_time_sec"] = round(total_time, 1)
    report["model_type"] = metadata["model_type"]
    report["final_eval"] = p2_eval

    with open(_TRAINING_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training voltooid in %.1f seconden", total_time)
    logger.info("Final Return: %.2f%%, Sortino: %.4f, MaxDD: %.2f%%",
                p2_eval["mean_return_pct"],
                p2_eval["mean_sortino"],
                p2_eval["mean_max_dd_pct"])
    logger.info("=" * 60)

    return report


def _evaluate_model(
    model: Any,
    env: Any,
    use_recurrent: bool,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evalueer model over meerdere episodes."""
    returns = []
    sortinos = []
    max_dds = []
    trade_counts = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_values = [1.0]
        lstm_states = None
        episode_starts = np.array([True])

        while not done:
            if use_recurrent:
                try:
                    action, lstm_states = model.predict(
                        obs.reshape(1, -1),
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    episode_starts = np.array([False])
                except TypeError:
                    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            else:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)

            obs, _, terminated, truncated, info = env.step(action.flatten())
            done = terminated or truncated
            if "portfolio_value" in info:
                ep_values.append(info["portfolio_value"])

        # Metrics
        total_return = (ep_values[-1] / ep_values[0] - 1.0) * 100
        returns.append(total_return)

        if len(ep_values) > 2:
            pv_returns = np.diff(ep_values) / (np.array(ep_values[:-1]) + 1e-10)
            downside = pv_returns[pv_returns < 0]
            ds_std = float(np.std(downside)) if len(downside) > 1 else 0.01
            sortino = float(np.mean(pv_returns)) / (ds_std + 1e-8)
        else:
            sortino = 0.0
        sortinos.append(sortino)

        peak = max(ep_values)
        max_dd = (peak - min(ep_values)) / peak * 100 if peak > 0 else 0.0
        max_dds.append(max_dd)
        trade_counts.append(info.get("total_trades", 0))

    return {
        "mean_return_pct": round(float(np.mean(returns)), 2),
        "std_return_pct": round(float(np.std(returns)), 2),
        "mean_sortino": round(float(np.mean(sortinos)), 4),
        "mean_max_dd_pct": round(float(np.mean(max_dds)), 2),
        "mean_trades": round(float(np.mean(trade_counts)), 1),
    }


# ── CLI entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Train RecurrentPPO agent")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Totaal training timesteps")
    parser.add_argument("--phase1", type=int, default=400_000, help="Fase 1 timesteps")
    parser.add_argument("--n-signals", type=int, default=10, help="Aantal assets")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    result = train_recurrent_ppo(
        total_timesteps=args.timesteps,
        phase1_timesteps=args.phase1,
        n_signals=args.n_signals,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print(f"\nTraining report: {json.dumps(result, indent=2)}")
