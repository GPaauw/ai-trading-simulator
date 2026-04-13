"""PPO RL Training Script — draait via GitHub Actions.

Traint een Stable-Baselines3 PPO agent op een custom TradingEnv.
De agent leert portfolio-allocatie beslissingen te nemen op basis van
LightGBM signalen.
"""

import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.config import (
    TOP_N_SIGNALS, RL_CONFIG, ML_MODELS_DIR, RL_MODEL_PATH,
    SIGNAL_MODEL_PATH, TRANSACTION_COST_PCT, MAX_POSITION_PCT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """Custom Gymnasium environment voor portfolio management training.

    Observatie: portfolio state + top-N signalen + risk metrics
    Acties: allocatie fractie per top-N signaal (continu, -1 tot 1)
    Reward: risk-adjusted return met penalties
    """

    metadata = {"render_modes": []}

    def __init__(self, signal_data: np.ndarray, price_data: np.ndarray, episode_length: int = 30):
        super().__init__()
        self.signal_data = signal_data  # shape: (timesteps, n_signals, 3) -> confidence, exp_ret, risk
        self.price_data = price_data    # shape: (timesteps, n_signals) -> prijzen
        self.episode_length = episode_length
        self.n_signals = min(TOP_N_SIGNALS, signal_data.shape[1])

        # Observatie: cash_ratio, invested_ratio, N*3 signal features, 4 risk metrics, 3 time features
        obs_dim = 2 + self.n_signals * 3 + 4 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Actie: allocatie per signaal (-1 = verkoop, +1 = koop)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_signals,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random startpunt
        max_start = len(self.signal_data) - self.episode_length - 1
        if max_start <= 0:
            self.start_idx = 0
        else:
            self.start_idx = self.np_random.integers(0, max_start)

        self.current_step = 0
        self.cash = 1.0  # genormaliseerd: 1.0 = 100% cash
        self.positions = np.zeros(self.n_signals)  # allocatie per signaal
        self.portfolio_values = [1.0]
        self.total_trades = 0

        return self._get_obs(), {}

    def step(self, action):
        # Huidige en volgende prijzen
        t = self.start_idx + self.current_step
        if t + 1 >= len(self.price_data):
            return self._get_obs(), 0.0, True, False, {}

        current_prices = self.price_data[t, :self.n_signals]
        next_prices = self.price_data[t + 1, :self.n_signals]

        # Bereken returns per positie
        price_returns = np.zeros(self.n_signals)
        for i in range(self.n_signals):
            if current_prices[i] > 0:
                price_returns[i] = (next_prices[i] - current_prices[i]) / current_prices[i]

        # Voer allocatie-veranderingen uit
        old_positions = self.positions.copy()
        for i in range(self.n_signals):
            target_alloc = np.clip(action[i] * MAX_POSITION_PCT, -MAX_POSITION_PCT, MAX_POSITION_PCT)
            trade_size = abs(target_alloc - old_positions[i])

            if trade_size > 0.01:  # minimum trade grootte
                cost = trade_size * TRANSACTION_COST_PCT
                self.cash -= cost
                self.total_trades += 1

            self.positions[i] = target_alloc

        # Update cash op basis van allocaties
        total_allocated = np.sum(np.abs(self.positions))
        self.cash = max(0, 1.0 - total_allocated)

        # Portfolio return
        position_return = np.sum(self.positions * price_returns)
        portfolio_value = self.portfolio_values[-1] * (1 + position_return)
        self.portfolio_values.append(portfolio_value)

        # Reward berekening
        reward = self._calculate_reward(position_return, portfolio_value)

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _calculate_reward(self, step_return: float, portfolio_value: float) -> float:
        """Risk-adjusted reward met penalties."""
        reward = step_return * 100  # Schaal returns

        # Drawdown penalty
        peak = max(self.portfolio_values)
        drawdown = (peak - portfolio_value) / peak
        if drawdown > 0.05:
            reward -= drawdown * 50

        # Overtrading penalty
        if self.total_trades > self.current_step * 3:
            reward -= 0.1

        # Consistentie bonus
        if len(self.portfolio_values) > 5:
            recent_returns = np.diff(self.portfolio_values[-5:]) / np.array(self.portfolio_values[-5:-1])
            if np.all(recent_returns > 0):
                reward += 0.5  # bonus voor consistente winst

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        t = self.start_idx + self.current_step
        t = min(t, len(self.signal_data) - 1)

        obs = []
        # Portfolio state
        obs.append(self.cash)
        obs.append(1.0 - self.cash)

        # Signal features
        for i in range(self.n_signals):
            if i < self.signal_data.shape[1]:
                obs.extend(self.signal_data[t, i, :].tolist())
            else:
                obs.extend([0.5, 0.0, 0.5])

        # Risk metrics
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            vol = float(np.std(returns)) if len(returns) > 1 else 0.0
            peak = max(self.portfolio_values)
            drawdown = (peak - self.portfolio_values[-1]) / peak
            sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8)) if len(returns) > 1 else 0.0
            win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.5
        else:
            vol, drawdown, sharpe, win_rate = 0.0, 0.0, 0.0, 0.5

        obs.extend([vol, drawdown, sharpe, win_rate])

        # Time features (genormaliseerd)
        obs.extend([
            (self.current_step % 5) / 4.0,  # dag van de week
            0.5,  # uur (niet relevant voor dagelijkse data)
            0.5,  # market phase
        ])

        return np.array(obs, dtype=np.float32)


def prepare_training_data(histories: dict[str, pd.DataFrame], n_signals: int = TOP_N_SIGNALS):
    """Bereid signaal- en prijsdata voor vanuit historische data."""
    from services.feature_engine import FeatureEngine
    from services.signal_predictor import SignalPredictor

    fe = FeatureEngine()
    predictor = SignalPredictor()

    # Gebruik de langste beschikbare historie als tijdas
    symbols = list(histories.keys())[:n_signals]
    min_len = min(len(histories[s]) for s in symbols)
    timesteps = min_len - 60  # eerste 60 voor feature berekening

    if timesteps < 30:
        raise ValueError("Te weinig data voor RL training")

    signal_data = np.zeros((timesteps, n_signals, 3))
    price_data = np.zeros((timesteps, n_signals))

    for t in range(timesteps):
        for i, sym in enumerate(symbols):
            hist = histories[sym].iloc[:60 + t + 1]
            features = fe.compute_features(hist)

            if features:
                # Simplified signaal features
                rsi = features.get("rsi_14", 50) / 100.0
                macd = np.clip(features.get("macd_hist", 0) * 10, -1, 1)
                vol = features.get("volatility_20d", 0.02) * 10

                signal_data[t, i] = [rsi, macd, vol]

            price_data[t, i] = float(histories[sym].iloc[60 + t]["close"])

    return signal_data, price_data


def main():
    os.makedirs(ML_MODELS_DIR, exist_ok=True)

    # Download data
    from ml.training.train_signals import download_data, get_training_symbols
    symbols = get_training_symbols()[:20]  # Gebruik subset voor RL (sneller)
    histories = download_data(symbols, period="2y")

    if len(histories) < 5:
        logger.error("Te weinig data. Training afgebroken.")
        sys.exit(1)

    logger.info("=== Preparing RL training data ===")
    signal_data, price_data = prepare_training_data(histories)
    logger.info("Signal data shape: %s, Price data shape: %s", signal_data.shape, price_data.shape)

    # Maak environment
    env = TradingEnv(signal_data, price_data, episode_length=RL_CONFIG["episode_length_days"])

    # Train PPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    logger.info("=== Training PPO agent ===")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=RL_CONFIG["learning_rate"],
        n_steps=RL_CONFIG["n_steps"],
        batch_size=RL_CONFIG["batch_size"],
        n_epochs=RL_CONFIG["n_epochs"],
        gamma=RL_CONFIG["gamma"],
        gae_lambda=RL_CONFIG["gae_lambda"],
        clip_range=RL_CONFIG["clip_range"],
        verbose=1,
        seed=42,
    )

    model.learn(total_timesteps=RL_CONFIG["total_timesteps"])

    # Opslaan
    model.save(RL_MODEL_PATH.replace(".zip", ""))
    logger.info("PPO model opgeslagen: %s", RL_MODEL_PATH)

    # Evaluatie
    logger.info("=== Evaluatie ===")
    eval_env = TradingEnv(signal_data, price_data, episode_length=RL_CONFIG["episode_length_days"])
    total_rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
        total_rewards.append(total_reward)
        final_value = eval_env.portfolio_values[-1]
        ret = (final_value - 1.0) * 100
        logger.info("  Episode %d: reward=%.2f, return=%.2f%%", ep + 1, total_reward, ret)

    avg_reward = np.mean(total_rewards)
    logger.info("Gemiddelde reward: %.2f", avg_reward)

    # Rapport opslaan
    report = {
        "trained_at": datetime.now().isoformat(),
        "total_timesteps": RL_CONFIG["total_timesteps"],
        "avg_eval_reward": round(float(avg_reward), 2),
        "eval_episodes": 10,
    }
    report_path = os.path.join(ML_MODELS_DIR, "rl_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=== RL Training voltooid ===")


if __name__ == "__main__":
    main()
