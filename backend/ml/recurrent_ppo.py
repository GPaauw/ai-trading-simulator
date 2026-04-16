"""RecurrentPPO Agent: LSTM-gebaseerde PPO voor portfolio management.

Wrapper rond sb3-contrib RecurrentPPO met:
  - LSTM policy (MlpLstmPolicy)
  - Curriculum learning (fase 1 → fase 2)
  - Sortino-gebaseerde evaluatie
  - Model save/load met metadata
  - Graceful fallback naar regels als model niet beschikbaar is
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Model paden ─────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).parent / "models"
_RPPO_MODEL_PATH = _MODEL_DIR / "recurrent_ppo"
_RPPO_METADATA_PATH = _MODEL_DIR / "recurrent_ppo_metadata.json"

# ── Defaults ────────────────────────────────────────────────────────
_TOTAL_TIMESTEPS: int = 2_000_000
_PHASE1_TIMESTEPS: int = 400_000
_LEARNING_RATE: float = 3e-4
_N_STEPS: int = 256
_BATCH_SIZE: int = 64
_N_EPOCHS: int = 10
_GAMMA: float = 0.99
_GAE_LAMBDA: float = 0.95
_CLIP_RANGE: float = 0.2
_ENT_COEF: float = 0.01
_VF_COEF: float = 0.5
_LSTM_HIDDEN_SIZE: int = 128
_N_LSTM_LAYERS: int = 1


class RecurrentPPOAgent:
    """RecurrentPPO agent voor portfolio allocatie.

    Maakt gebruik van sb3-contrib RecurrentPPO met LSTM policy.
    Als sb3-contrib niet beschikbaar is, valt terug op standaard PPO
    of regelgebaseerde beslissingen.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        model_path: Optional[str] = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.model_path = model_path or str(_RPPO_MODEL_PATH)
        self.model: Any = None
        self.lstm_states: Any = None
        self.episode_start: bool = True
        self._predict_count: int = 0  # FIXED: issue-2 — track calls voor periodieke LSTM reset
        self._metadata: Dict[str, Any] = {}
        self._use_recurrent: bool = False

        self._load_model()

    # ── Model laden ─────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Laad het RecurrentPPO model. Fallback naar PPO, dan regels."""
        model_file = self.model_path + ".zip"

        if not os.path.exists(model_file):
            logger.warning(
                "RecurrentPPO model niet gevonden op %s — gebruik regels-fallback",
                model_file,
            )
            return

        # Probeer RecurrentPPO (sb3-contrib)
        try:
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(self.model_path)
            self._use_recurrent = True
            logger.info("RecurrentPPO model geladen (LSTM policy)")
        except ImportError:
            logger.warning("sb3-contrib niet geïnstalleerd — probeer standaard PPO")
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(self.model_path)
                self._use_recurrent = False
                logger.info("Fallback PPO model geladen (MlpPolicy)")
            except Exception as e:
                logger.warning("Kon model niet laden: %s — gebruik regels", e)
                return
        except Exception as e:
            logger.warning("Fout bij laden RecurrentPPO: %s — gebruik regels", e)
            return

        # Metadata laden
        meta_path = Path(self.model_path).parent / "recurrent_ppo_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    self._metadata = json.load(f)
                logger.info(
                    "Model metadata: trained=%s, timesteps=%s",
                    self._metadata.get("trained_at", "onbekend"),
                    self._metadata.get("total_timesteps", "onbekend"),
                )
            except Exception:
                pass

    # ── Predict ─────────────────────────────────────────────────────

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Voorspel allocatie-acties op basis van observatie.

        Args:
            observation: float32 array shape (obs_dim,)
            deterministic: of de policy deterministisch moet zijn

        Returns:
            (actions, info) — actions ∈ [-1, 1] shape (n_actions,),
                              info met confidence en methode
        """
        if self.model is None:
            return self._predict_rules(observation)

        try:
            obs = observation.reshape(1, -1).astype(np.float32)

            if self._use_recurrent:
                # FIXED: issue-2 — reset LSTM states elke 50 stappen om stale state te voorkomen
                if self._predict_count % 50 == 0 and self._predict_count > 0:
                    logger.debug("LSTM state reset na %d predict-calls.", self._predict_count)
                    self.lstm_states = None
                    self.episode_start = True

                actions, self.lstm_states = self.model.predict(
                    obs,
                    state=self.lstm_states,
                    episode_start=np.array([self.episode_start]),
                    deterministic=deterministic,
                )
                self.episode_start = False
                self._predict_count += 1  # FIXED: issue-2
            else:
                actions, _ = self.model.predict(obs, deterministic=deterministic)

            actions = np.clip(actions.flatten()[:self.n_actions], -1.0, 1.0)

            return actions, {
                "method": "recurrent_ppo" if self._use_recurrent else "ppo",
                "confidence": float(np.max(np.abs(actions))),
            }

        except Exception as e:
            logger.error("RecurrentPPO predict fout: %s — fallback naar regels", e)
            return self._predict_rules(observation)

    def reset_states(self) -> None:
        """Reset LSTM hidden states voor een nieuwe episode."""
        self.lstm_states = None
        self.episode_start = True
        self._predict_count = 0  # FIXED: issue-2 — reset counter bij episode boundary

    # ── Rules fallback ──────────────────────────────────────────────

    def _predict_rules(
        self,
        observation: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Regelgebaseerde fallback portfolio-allocatie.

        Strategie: alloceer proportioneel aan expected return × (1 - risk).
        """
        actions = np.zeros(self.n_actions, dtype=np.float32)

        # Parse signal features uit observatie (na cash + invested)
        sig_start = 2
        for i in range(self.n_actions):
            idx = sig_start + i * 3
            if idx + 2 < len(observation):
                confidence = observation[idx]
                exp_return = observation[idx + 1]
                risk = observation[idx + 2]
                # Score: verwacht rendement × (1 – risico)
                score = exp_return * (1.0 - risk) * confidence
                actions[i] = np.clip(score * 3.0, -1.0, 1.0)

        return actions, {"method": "rules", "confidence": 0.3}

    # ── Evaluatie ───────────────────────────────────────────────────

    def evaluate(
        self,
        env: Any,
        n_episodes: int = 10,
    ) -> Dict[str, float]:
        """Evalueer agent over meerdere episodes.

        Returns:
            Dict met mean_return, mean_sortino, mean_max_dd, mean_trades
        """
        returns: List[float] = []
        sortinos: List[float] = []
        max_dds: List[float] = []
        trade_counts: List[int] = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            self.reset_states()
            done = False
            episode_values = [1.0]

            while not done:
                actions, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                if "portfolio_value" in info:
                    episode_values.append(info["portfolio_value"])

            # Bereken metrics
            total_return = (episode_values[-1] / episode_values[0] - 1.0) * 100
            returns.append(total_return)

            pv_returns = np.diff(episode_values) / (np.array(episode_values[:-1]) + 1e-10)
            downside = pv_returns[pv_returns < 0]
            ds_std = float(np.std(downside)) if len(downside) > 1 else 0.01
            sortino = float(np.mean(pv_returns)) / (ds_std + 1e-8) if len(pv_returns) > 0 else 0.0
            sortinos.append(sortino)

            peak = max(episode_values)
            max_dd = (peak - min(episode_values)) / peak * 100 if peak > 0 else 0.0
            max_dds.append(max_dd)
            trade_counts.append(info.get("total_trades", 0))

        results = {
            "mean_return_pct": round(float(np.mean(returns)), 2),
            "std_return_pct": round(float(np.std(returns)), 2),
            "mean_sortino": round(float(np.mean(sortinos)), 4),
            "mean_max_dd_pct": round(float(np.mean(max_dds)), 2),
            "mean_trades": round(float(np.mean(trade_counts)), 1),
        }
        logger.info("Evaluatie: %s", results)
        return results

    # ── Metadata ────────────────────────────────────────────────────

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def policy_type(self) -> str:
        if self.model is None:
            return "rules"
        return "recurrent_ppo" if self._use_recurrent else "ppo"
