"""Trading Environment v2: Gymnasium env voor RecurrentPPO training.

Uitgebreide versie van de originele TradingEnv met:
  - 252 trading-dagen episodes (1 volledig jaar)
  - Uitgebreide state space: regime, sentiment, VIX, retraining-leeftijd
  - Sortino-gebaseerde reward met drawdown penalty en regime-switch bonus
  - Asset-type transactiekosten (stocks vs crypto)
  - Curriculum learning support (2-fase)

State space:
  [cash_ratio, invested_ratio]                    (2)
  + [confidence, exp_return, risk] × TOP_N        (N×3)
  + [vol, max_drawdown, sortino, win_rate]        (4)
  + [day_of_week, hour, market_phase]             (3)
  + [regime_bear, regime_sideways, regime_bull, regime_high_vol]  (4)
  + [sentiment_mean, vix_level]                   (2)
  + [days_since_retrain_normalized]               (1)
  = 2 + N×3 + 4 + 3 + 4 + 2 + 1 = 16 + N×3

Action space:
  Continuous allocation per top-N signal ∈ [-1, 1]
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ── Constanten ──────────────────────────────────────────────────────
_TRANSACTION_COST_STOCK: float = 0.001     # 0.1% per trade (stocks)
_TRANSACTION_COST_CRYPTO: float = 0.0004   # 0.04% per trade (crypto)
_SLIPPAGE_PCT: float = 0.0005             # 0.05% slippage
_MAX_POSITION_PCT: float = 0.15           # max 15% per positie
_MIN_TRADE_SIZE: float = 0.01             # minimum trade grootte
_DEFAULT_EPISODE_LENGTH: int = 252        # 1 jaar trading-dagen
_DOWNSIDE_TARGET: float = 0.0            # Sortino target return


class TradingEnvV2(gym.Env):
    """Gymnasium environment voor RecurrentPPO portfolio management.

    Ondersteunt curriculum learning via ``curriculum_phase``:
      Phase 1: 2 assets, korte episodes (60 dagen)
      Phase 2: volledige universe, 252-dagen episodes

    Args:
        signal_data: shape (timesteps, n_assets, 3) — confidence, exp_return, risk
        price_data: shape (timesteps, n_assets) — close prices
        regime_data: shape (timesteps, 4) — regime one-hot per timestep
        sentiment_data: shape (timesteps,) — gemiddeld sentiment per timestep
        vix_data: shape (timesteps,) — VIX niveau per timestep
        asset_types: list of "stock" or "crypto" per asset
        n_signals: max assets in state space
        episode_length: default episode lengte
        curriculum_phase: 1 = simpel, 2 = volledig
        days_since_retrain: initiële waarde voor retraining-leeftijd feature
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        signal_data: np.ndarray,
        price_data: np.ndarray,
        regime_data: Optional[np.ndarray] = None,
        sentiment_data: Optional[np.ndarray] = None,
        vix_data: Optional[np.ndarray] = None,
        asset_types: Optional[List[str]] = None,
        n_signals: int = 10,
        episode_length: int = _DEFAULT_EPISODE_LENGTH,
        curriculum_phase: int = 2,
        days_since_retrain: int = 0,
    ) -> None:
        super().__init__()

        self.signal_data = signal_data
        self.price_data = price_data
        self.n_timesteps = len(signal_data)
        self.n_assets_total = signal_data.shape[1]

        # Curriculum: fase 1 = 2 assets, korte episodes
        if curriculum_phase == 1:
            self.n_signals = min(2, self.n_assets_total)
            self.episode_length = min(60, episode_length)
        else:
            self.n_signals = min(n_signals, self.n_assets_total)
            self.episode_length = episode_length

        # Optionele extra data (met defaults)
        self.regime_data = regime_data if regime_data is not None else np.tile(
            [0.0, 1.0, 0.0, 0.0], (self.n_timesteps, 1),  # default: sideways
        )
        self.sentiment_data = sentiment_data if sentiment_data is not None else np.full(
            self.n_timesteps, 0.5,
        )
        self.vix_data = vix_data if vix_data is not None else np.full(
            self.n_timesteps, 20.0,
        )
        self.asset_types = asset_types or ["stock"] * self.n_assets_total
        self._days_since_retrain = days_since_retrain

        # Transaction costs per asset
        self._tx_costs = np.array([
            _TRANSACTION_COST_CRYPTO if t == "crypto" else _TRANSACTION_COST_STOCK
            for t in self.asset_types[:self.n_signals]
        ])

        # Observatie dimensie: 2 + N*3 + 4 + 3 + 4 + 2 + 1
        self._obs_dim = 2 + self.n_signals * 3 + 4 + 3 + 4 + 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Actie: allocatie per signaal
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_signals,),
            dtype=np.float32,
        )

        # Runtime state (wordt in reset() geïnitialiseerd)
        self.current_step: int = 0
        self.start_idx: int = 0
        self.cash: float = 1.0
        self.positions: np.ndarray = np.zeros(self.n_signals)
        self.portfolio_values: List[float] = [1.0]
        self.total_trades: int = 0
        self.total_costs: float = 0.0
        self._prev_regime_idx: int = 1  # sideways

    # ── Gymnasium interface ─────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset de environment voor een nieuwe episode."""
        super().reset(seed=seed)

        max_start = self.n_timesteps - self.episode_length - 1
        if max_start <= 0:
            self.start_idx = 0
        else:
            self.start_idx = int(self.np_random.integers(0, max_start))

        self.current_step = 0
        self.cash = 1.0
        self.positions = np.zeros(self.n_signals)
        self.portfolio_values = [1.0]
        self.total_trades = 0
        self.total_costs = 0.0
        self._prev_regime_idx = int(np.argmax(self.regime_data[self.start_idx]))

        info = {"start_idx": self.start_idx, "episode_length": self.episode_length}
        return self._get_obs(), info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Voer een trading-stap uit.

        Args:
            action: allocatie-vector ∈ [-1, 1] per asset.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        t = self.start_idx + self.current_step

        if t + 1 >= self.n_timesteps:
            return self._get_obs(), 0.0, True, False, self._get_info()

        # ── Prijsbewegingen ─────────────────────────────────────────
        current_prices = self.price_data[t, :self.n_signals]
        next_prices = self.price_data[t + 1, :self.n_signals]

        price_returns = np.zeros(self.n_signals)
        for i in range(self.n_signals):
            if current_prices[i] > 0:
                price_returns[i] = (next_prices[i] - current_prices[i]) / current_prices[i]

        # ── Allocatie uitvoeren met kosten ──────────────────────────
        old_positions = self.positions.copy()
        step_costs = 0.0

        for i in range(self.n_signals):
            target_alloc = np.clip(
                float(action[i]) * _MAX_POSITION_PCT,
                -_MAX_POSITION_PCT,
                _MAX_POSITION_PCT,
            )
            trade_size = abs(target_alloc - old_positions[i])

            if trade_size > _MIN_TRADE_SIZE:
                # Transactiekosten + slippage
                cost = trade_size * (self._tx_costs[i] + _SLIPPAGE_PCT)
                step_costs += cost
                self.total_trades += 1

            self.positions[i] = target_alloc

        self.cash -= step_costs
        self.total_costs += step_costs

        # ── Portfolio value update ──────────────────────────────────
        total_allocated = np.sum(np.abs(self.positions))
        self.cash = max(0.0, 1.0 - total_allocated)

        position_return = float(np.sum(self.positions * price_returns))
        portfolio_value = self.portfolio_values[-1] * (1.0 + position_return) - step_costs
        portfolio_value = max(portfolio_value, 0.0)
        self.portfolio_values.append(portfolio_value)

        # ── Reward ──────────────────────────────────────────────────
        reward = self._calculate_reward(position_return, portfolio_value, t)

        # ── Regime-switch bonus ─────────────────────────────────────
        current_regime_idx = int(np.argmax(self.regime_data[min(t + 1, self.n_timesteps - 1)]))
        if current_regime_idx != self._prev_regime_idx:
            # Regime is veranderd — beloon als agent zijn positie heeft aangepast
            position_change = float(np.sum(np.abs(self.positions - old_positions)))
            if position_change > 0.05:
                reward += 0.1  # bonus voor reageren op regime-switch
        self._prev_regime_idx = current_regime_idx

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        # Vroeg stoppen bij extreme drawdown
        peak = max(self.portfolio_values)
        drawdown = (peak - portfolio_value) / peak if peak > 0 else 0.0
        if drawdown > 0.30:
            terminated = True
            reward -= 5.0  # zware penalty voor extreme drawdown

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ── Reward functie ──────────────────────────────────────────────

    def _calculate_reward(
        self,
        step_return: float,
        portfolio_value: float,
        t: int,
    ) -> float:
        """Sortino-gebaseerde reward met drawdown-penalty.

        Reward = base_sortino + regime_bonus - drawdown_penalty - cost_penalty
        """
        # ── Base: Sortino ratio (incrementeel) ─────────────────────
        if len(self.portfolio_values) < 3:
            base = step_return * 100.0
        else:
            returns = np.diff(self.portfolio_values[-min(20, len(self.portfolio_values)):])
            prev_vals = np.array(self.portfolio_values[-min(20, len(self.portfolio_values)):-1])
            pct_returns = returns / (prev_vals + 1e-10)

            # Sortino: alleen downside deviation
            downside = pct_returns[pct_returns < _DOWNSIDE_TARGET]
            downside_std = float(np.std(downside)) if len(downside) > 1 else 0.01
            mean_return = float(np.mean(pct_returns))
            base = (mean_return / (downside_std + 1e-8)) * 10.0  # geschaald

        # ── Drawdown penalty: -0.5 × drawdown² ────────────────────
        peak = max(self.portfolio_values)
        drawdown = (peak - portfolio_value) / peak if peak > 0 else 0.0
        dd_penalty = -0.5 * (drawdown ** 2) * 100.0

        # ── Overtrading penalty ────────────────────────────────────
        cost_penalty = -self.total_costs * 50.0 if self.total_costs > 0.01 else 0.0

        reward = base + dd_penalty + cost_penalty
        return float(np.clip(reward, -10.0, 10.0))

    # ── Observatie ──────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Bouw de observatie-vector."""
        t = min(self.start_idx + self.current_step, self.n_timesteps - 1)
        obs: List[float] = []

        # Portfolio state (2)
        obs.append(self.cash)
        obs.append(1.0 - self.cash)

        # Signal features: confidence, expected_return, risk (N×3)
        for i in range(self.n_signals):
            if i < self.signal_data.shape[1]:
                obs.extend(self.signal_data[t, i, :].tolist())
            else:
                obs.extend([0.5, 0.0, 0.5])

        # Risk metrics (4)
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / (np.array(self.portfolio_values[:-1]) + 1e-10)
            vol = float(np.std(returns)) if len(returns) > 1 else 0.0
            peak = max(self.portfolio_values)
            max_dd = float((peak - min(self.portfolio_values[-20:])) / peak) if peak > 0 else 0.0
            sortino = self._calc_sortino(returns)
            win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.5
        else:
            vol, max_dd, sortino, win_rate = 0.0, 0.0, 0.0, 0.5
        obs.extend([vol, max_dd, np.clip(sortino, -5.0, 5.0), win_rate])

        # Time features (3)
        day_in_week = (t % 5) / 4.0
        hour_norm = 0.5  # dagelijkse data
        market_phase = (self.current_step / max(self.episode_length, 1))
        obs.extend([day_in_week, hour_norm, market_phase])

        # Regime one-hot (4)
        regime = self.regime_data[t]
        obs.extend(regime.tolist())

        # Sentiment + VIX (2)
        obs.append(float(self.sentiment_data[t]))
        obs.append(float(self.vix_data[t]) / 100.0)  # genormaliseerd

        # Days since retrain (1)
        retrain_norm = min(self._days_since_retrain + self.current_step, 365) / 365.0
        obs.append(retrain_norm)

        return np.array(obs, dtype=np.float32)

    # ── Info ────────────────────────────────────────────────────────

    def _get_info(self) -> Dict[str, Any]:
        """Retourneer episode-informatie voor logging."""
        pv = self.portfolio_values
        total_return = (pv[-1] / pv[0] - 1.0) * 100 if pv[0] > 0 else 0.0
        peak = max(pv)
        max_dd = (peak - min(pv)) / peak * 100 if peak > 0 else 0.0

        return {
            "portfolio_value": round(pv[-1], 6),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": self.total_trades,
            "total_costs": round(self.total_costs, 6),
            "steps": self.current_step,
        }

    # ── Sortino helper ──────────────────────────────────────────────

    @staticmethod
    def _calc_sortino(returns: np.ndarray) -> float:
        """Bereken Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < _DOWNSIDE_TARGET]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 0.01
        mean_return = float(np.mean(returns))
        return mean_return / (downside_std + 1e-8)
