"""ML configuratie: feature definities, hyperparameters en constanten."""

from typing import Dict, List
import os

# ── Feature definities ──────────────────────────────────────────────
FEATURE_COLUMNS: List[str] = [
    # Prijs-returns
    "return_1d", "return_5d", "return_20d",
    "log_return_1d", "volatility_20d",
    # Technische indicatoren
    "rsi_14", "macd_hist", "macd_signal_diff",
    "sma_5", "sma_20", "sma_50",
    "ema_12", "ema_26", "ema_diff",
    "bb_upper_dist", "bb_lower_dist", "bb_width",
    "atr_14", "atr_pct",
    # Volume
    "volume_ratio", "volume_ma_ratio", "obv_slope",
    # Momentum
    "roc_10", "stoch_rsi", "williams_r",
    # Prijsrelaties
    "price_vs_sma20", "price_vs_sma50",
    "high_low_range",
]

NUM_FEATURES = len(FEATURE_COLUMNS)

# ── V2 feature definities (originele 28 + 22 nieuwe) ───────────────
FEATURE_COLUMNS_V2: List[str] = FEATURE_COLUMNS + [
    # Alternatieve data (Module 2: AlternativeDataCollector)
    "sentiment_score",                  # FinBERT ∈ [0, 1]
    "reddit_mention_velocity",          # 24h growth rate
    "insider_buy_sell_ratio",           # EDGAR Form 4 ∈ [0, 1]
    "institutional_holdings_delta",     # EDGAR 13F genormaliseerd
    # Macro (FRED)
    "vix_level",                        # huidig VIX niveau
    "vix_5d_change",                    # 5-dag VIX delta
    # Market regime (Module 1: RegimeClassifier)
    "regime_bear",                      # one-hot
    "regime_sideways",                  # one-hot
    "regime_bull",                      # one-hot
    "regime_high_vol",                  # one-hot
    # Order flow proxy
    "order_book_imbalance",             # (close-low)/(high-low) ∈ [0, 1]
    # Earnings
    "days_since_earnings",              # dagen sinds laatste earnings
    "earnings_surprise",                # beat/miss indicator
    # Extra technische indicatoren
    "keltner_position",                 # positie in Keltner Channel ∈ [0, 1]
    "vwap_deviation",                   # afwijking van VWAP (%)
    "heikin_ashi_streak",               # bullish ratio laatste 5 bars ∈ [0, 1]
    "adx_14",                           # Average Directional Index (trendsterkte)
    "cmf_20",                           # Chaikin Money Flow ∈ [-1, 1]
    "vol_ratio_5_20",                   # korte/lange volatiliteit ratio
    "momentum_divergence",              # prijs-RSI divergentie {-1, 0, 1}
]

NUM_FEATURES_V2 = len(FEATURE_COLUMNS_V2)

# ── Signal klassen ──────────────────────────────────────────────────
SIGNAL_CLASSES = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
NUM_CLASSES = len(SIGNAL_CLASSES)

# ── LightGBM hyperparameters ───────────────────────────────────────
LGBM_PARAMS: Dict = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "seed": 42,
}

# ── RL (PPO) hyperparameters ───────────────────────────────────────
RL_CONFIG: Dict = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "total_timesteps": 500_000,
    "episode_length_days": 30,
}

# ── Portfolio RL observatie-ruimte ──────────────────────────────────
TOP_N_SIGNALS = 10  # Hoeveel top-signalen het RL model ziet
RL_OBS_DIM = (
    2                  # cash_ratio, portfolio_value_ratio
    + TOP_N_SIGNALS * 3  # per signaal: confidence, expected_return, risk
    + 4                # risk metrics: volatility, max_drawdown, sharpe, win_rate
    + 3                # tijd: day_of_week (0-4), hour (0-23), market_phase (0-3)
)
RL_ACTION_DIM = TOP_N_SIGNALS + 1  # allocatie per signaal + hold/cash actie

# ── Risicobeheer ───────────────────────────────────────────────────
MAX_POSITION_PCT = 0.15        # Max 15% van portfolio per positie
MAX_DAILY_LOSS_PCT = 0.03      # Max 3% dagelijks verlies
MAX_DRAWDOWN_PCT = 0.10        # Max 10% drawdown → stop trading
TRANSACTION_COST_PCT = 0.001   # 0.1% transactiekosten
SLIPPAGE_PCT = 0.0005          # 0.05% slippage

# ── Training data ──────────────────────────────────────────────────
HISTORY_YEARS = 2
LABEL_HORIZON_SHORT = 1   # dagen vooruit voor daytrade label
LABEL_HORIZON_LONG = 5    # dagen vooruit voor swing label
CV_SPLITS = 5             # TimeSeriesSplit folds

# ── Thresholds voor label-discretisatie ────────────────────────────
# Prijsverandering → klasse mapping
LABEL_THRESHOLDS = {
    "strong_sell": -0.03,  # < -3%
    "sell": -0.01,         # -3% tot -1%
    "hold": 0.01,          # -1% tot +1%
    "buy": 0.03,           # +1% tot +3%
    "strong_buy": float("inf"),  # > +3%
}

# ── Paden ──────────────────────────────────────────────────────────
ML_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
SIGNAL_MODEL_PATH = os.path.join(ML_MODELS_DIR, "signal_model.joblib")
RETURN_MODEL_PATH = os.path.join(ML_MODELS_DIR, "return_model.joblib")
RL_MODEL_PATH = os.path.join(ML_MODELS_DIR, "portfolio_ppo.zip")

# ── Startkapitaal ──────────────────────────────────────────────────
START_BALANCE = float(os.getenv("START_BALANCE", "5000"))
