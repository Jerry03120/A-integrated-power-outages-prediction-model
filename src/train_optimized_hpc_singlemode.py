#!/usr/bin/env python3
"""
Improved Optimized Model Training Script - EVENT_ID Version
Single-cell version with reviewer-response upgrades

MAIN ADDITIONS
1. Feature-set ablation modes:
   - all
   - autoregressive_only
   - weather_only
   - no_outage_lag
2. Naive persistence baseline evaluation
3. Forward-in-time EVENT_ID CV instead of GroupKFold
4. Repeated forward CV summary for generalization stability
5. Safer XGBoost categorical handling for unseen EVENT_TYPE
6. Seed-ensemble variability naming cleanup
7. Faster row-wise summary statistics
8. Run config + experiment comparison outputs
9. CLI support for weather source mode
10. Grouped permutation importance outputs
11. Cross-run ablation summary table

IMPORTANT INTERPRETATION NOTE
- Seed ensemble below quantifies training stochasticity / optimization sensitivity.
- It does NOT represent full predictive uncertainty or formal generalization uncertainty.
- Generalization stability is summarized separately via repeated forward CV.
"""

import os
import sys
import json
import pickle
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import signal
import traceback

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from skopt import gp_minimize
from skopt.space import Integer, Real

warnings.filterwarnings("ignore")


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_to_baseline(metrics_df: pd.DataFrame, naive_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or naive_df.empty:
        return pd.DataFrame()

    ml_mean = metrics_df[["rmse", "mae", "r2"]].mean().to_dict()
    naive = naive_df.iloc[0].to_dict()

    return pd.DataFrame([{
        "rmse_ml_mean": safe_float(ml_mean["rmse"]),
        "rmse_naive": safe_float(naive["rmse"]),
        "rmse_improvement_vs_naive": safe_float(naive["rmse"]) - safe_float(ml_mean["rmse"]),
        "mae_ml_mean": safe_float(ml_mean["mae"]),
        "mae_naive": safe_float(naive["mae"]),
        "mae_improvement_vs_naive": safe_float(naive["mae"]) - safe_float(ml_mean["mae"]),
        "r2_ml_mean": safe_float(ml_mean["r2"]),
        "r2_naive": safe_float(naive["r2"]),
        "r2_improvement_vs_naive": safe_float(ml_mean["r2"]) - safe_float(naive["r2"]),
    }])


def time_block_cv_splits(
    df: pd.DataFrame,
    n_splits: int,
    repeat_idx: int = 0,
    jitter: int = 0
):
    """
    Forward-in-time CV by EVENT_ID blocks.
    Train on earlier EVENT_IDs, validate on later EVENT_IDs.

    repeat_idx/jitter allow slightly shifted split plans across repeats
    to measure stability across multiple forward-CV allocations.
    """
    unique_events = np.sort(df["EVENT_ID"].dropna().unique())
    if len(unique_events) < 3:
        raise ValueError("Not enough unique EVENT_ID values for time-block CV")

    n_splits = min(n_splits, max(2, len(unique_events) - 1))

    if jitter > 0 and repeat_idx > 0:
        approx_fold_size = max(1, len(unique_events) // n_splits)
        max_shift = max(0, approx_fold_size - 1)
        shift = min(repeat_idx * jitter, max_shift)
    else:
        shift = 0

    shifted_events = unique_events[shift:] if shift > 0 else unique_events
    if len(shifted_events) < 3:
        shifted_events = unique_events

    n_splits_eff = min(n_splits, max(2, len(shifted_events) - 1))
    folds = np.array_split(shifted_events, n_splits_eff)

    split_count = 0
    for i in range(1, len(folds)):
        val_events = folds[i]
        train_events = np.concatenate(folds[:i]) if i > 0 else np.array([], dtype=shifted_events.dtype)

        train_idx = df.index[df["EVENT_ID"].isin(train_events)].to_numpy()
        val_idx = df.index[df["EVENT_ID"].isin(val_events)].to_numpy()

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        split_count += 1
        yield train_idx, val_idx

    if split_count == 0:
        raise ValueError("No valid forward time CV splits could be created")


# ================================================================================
# LOGGING SETUP
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ================================================================================
# CONFIGURATION
# ================================================================================

@dataclass
class Config:
    TARGET: str = "recorded_outages"
    CATEGORICAL_FEATURE: str = "EVENT_TYPE"
    EVENT_ID_CUTOFF: int = 3028
    RANDOM_STATE: int = 42

    # CV / optimization
    N_SPLITS: int = 10
    N_CV_REPEATS: int = 10
    CV_BLOCK_JITTER: int = 1
    USE_REPEATED_TIME_CV: bool = True

    N_BAYESIAN_CALLS: int = 50
    N_RANDOM_STARTS: int = 10
    EARLY_STOPPING_ROUNDS: int = 30
    NUM_BOOST_ROUND: int = 500

    MODEL_TYPE: str = "Unified"          # Unified / Ensemble
    ALGORITHM: str = "LightGBM"          # LightGBM / XGBoost
    LAG: int = 1
    EVENT_TYPE_VALUE: Optional[str] = None

    # Seed-ensemble variability
    N_SEED_ENSEMBLE_MEMBERS: int = 10
    COMPUTE_PERMUTATION_IMPORTANCE: bool = True

    ENSEMBLE_RETRAIN_EACH_TEST: bool = True
    BOOTSTRAP_TRAIN: bool = False
    ENSEMBLE_FRACTION_FLOOR: float = 0.95
    XGB_FRACTION_FLOOR: float = 0.95

    # Feature sets
    FEATURE_SET_MODE: str = "all"        # all / autoregressive_only / weather_only / no_outage_lag
    WEATHER_SOURCE_MODE: str = "lagged_observed"  # lagged_observed / forecast

    def __post_init__(self):
        self.BASE_META_FEATURES = [
            "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG",
            "Centerline_Miles", "Lane_Miles", "Truck_DVMT",
            "Total DVMT", "Population", "LAI"
        ]

        weather_features = [
            "ALLSKY_KT", "ALLSKY_NKT", "ALLSKY_SFC_LW_DWN", "ALLSKY_SFC_PAR_TOT",
            "ALLSKY_SFC_SW_DIFF", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DWN",
            "ALLSKY_SFC_UVA", "ALLSKY_SFC_UVB", "ALLSKY_SFC_UV_INDEX",
            "ALLSKY_SRF_ALB", "AOD_55", "AOD_84", "CLRSKY_KT", "CLRSKY_NKT",
            "PRECSNOLAND", "PRECTOTCORR", "PS", "PW", "QV10M", "QV2M", "RH2M",
            "SNODP", "SZA", "T2MWET", "T2M", "TOA_SW_DNI", "TOA_SW_DWN",
            "V2M", "V50M", "WD10M", "WD2M", "WD50M", "WS10M", "WS2M", "WS50M"
        ]

        if self.WEATHER_SOURCE_MODE == "lagged_observed":
            self.WEATHER_FEATURES = [f"{feat}_lag{self.LAG}" for feat in weather_features]
        elif self.WEATHER_SOURCE_MODE == "forecast":
            self.WEATHER_FEATURES = [f"{feat}_forecast_tplus{self.LAG}" for feat in weather_features]
        else:
            raise ValueError(f"Unknown WEATHER_SOURCE_MODE: {self.WEATHER_SOURCE_MODE}")

        self.OUTAGE_LAG_FEATURE = f"{self.TARGET}_lag{self.LAG}"

        self.SELECTED_COLUMNS = (
            ["Date", "CNTY_NM", self.TARGET, "EVENT_TYPE", "EVENT_ID"]
            + self.BASE_META_FEATURES
            + self.WEATHER_FEATURES
            + [self.OUTAGE_LAG_FEATURE]
        )

        if self.MODEL_TYPE == "Ensemble":
            self.IGNORE_COLS = ["Date", "CNTY_NM", self.TARGET, "EVENT_ID", "EVENT_TYPE"]
            logger.info("Ensemble mode: EVENT_TYPE excluded from predictors")
        else:
            self.IGNORE_COLS = ["Date", "CNTY_NM", self.TARGET, "EVENT_ID"]
            logger.info("Unified mode: EVENT_TYPE included as categorical predictor")

        self.SEED_VARIABILITY_NOTE = (
            "Optimization/subsampling sensitivity only; not formal predictive uncertainty."
        )

        logger.info(
            f"Config: MODEL_TYPE={self.MODEL_TYPE}, ALGORITHM={self.ALGORITHM}, "
            f"LAG={self.LAG}, FEATURE_SET_MODE={self.FEATURE_SET_MODE}"
        )
        logger.info(f"Weather Source Mode: {self.WEATHER_SOURCE_MODE}")
        logger.info(
            f"Train/Test Split: EVENT_ID <= {self.EVENT_ID_CUTOFF} (train) vs >= {self.EVENT_ID_CUTOFF + 1} (test)"
        )

    def get_output_path(self, base_dir: Path) -> Path:
        feature_part = f"features_{self.FEATURE_SET_MODE}"
        weather_part = f"weather_{self.WEATHER_SOURCE_MODE}"
        if self.MODEL_TYPE == "Ensemble" and self.EVENT_TYPE_VALUE is not None:
            return (
                base_dir / self.MODEL_TYPE / self.ALGORITHM / f"lag{self.LAG}" /
                weather_part / feature_part / f"EVENT_TYPE_{self.EVENT_TYPE_VALUE}"
            )
        else:
            return base_dir / self.MODEL_TYPE / self.ALGORITHM / f"lag{self.LAG}" / weather_part / feature_part


# ================================================================================
# FEATURE GROUPING
# ================================================================================

def categorize_feature_group(feature: str, config: Config) -> str:
    if feature == config.CATEGORICAL_FEATURE:
        return "event_type"
    if feature == config.OUTAGE_LAG_FEATURE:
        return "autoregressive_outage"
    if feature in config.WEATHER_FEATURES:
        return "weather"
    if feature in {"Population", "Total DVMT", "Truck_DVMT"}:
        return "exposure_proxy"
    if feature in {
        "ALAND_SQMI", "AWATER_SQMI", "INTPTLAT", "INTPTLONG",
        "Centerline_Miles", "Lane_Miles", "LAI"
    }:
        return "geography_infrastructure"
    return "other"


# ================================================================================
# CHECKPOINT MANAGER
# ================================================================================

class CheckpointManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_bayesian_checkpoint(self, iteration: int, best_params: Dict, all_results: List):
        checkpoint = {
            "iteration": int(iteration),
            "best_params": best_params,
            "all_results": all_results,
            "timestamp": time.time()
        }
        path = self.checkpoint_dir / f"bayesian_iter{iteration:03d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info(f"✓ Saved Bayesian checkpoint: iteration {iteration}")

    def load_bayesian_checkpoint(self) -> Tuple[int, Optional[Dict], List]:
        files = sorted(self.checkpoint_dir.glob("bayesian_iter*.pkl"))
        if not files:
            return 0, None, []

        latest = files[-1]
        with open(latest, "rb") as f:
            checkpoint = pickle.load(f)

        logger.info(f"✓ Loaded Bayesian checkpoint: iteration {checkpoint['iteration']}")
        return checkpoint["iteration"], checkpoint["best_params"], checkpoint["all_results"]

    def save_model_checkpoint(self, model, algorithm: str):
        path = self.checkpoint_dir / "model_checkpoint.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("✓ Saved model checkpoint")

    def load_model_checkpoint(self):
        path = self.checkpoint_dir / "model_checkpoint.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("✓ Loaded model checkpoint")
        return model

    def save_test_predictions_checkpoint(
        self,
        test_num: int,
        predictions: pd.DataFrame,
        importance: Optional[pd.DataFrame] = None
    ):
        checkpoint = {
            "test_num": test_num,
            "predictions": predictions,
            "importance": importance,
            "timestamp": time.time()
        }
        path = self.checkpoint_dir / f"test_predictions_{test_num:02d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info(f"✓ Saved test predictions checkpoint: test {test_num}")

    def load_test_predictions_checkpoints(self) -> List[int]:
        files = list(self.checkpoint_dir.glob("test_predictions_*.pkl"))
        completed = []
        for f in files:
            try:
                completed.append(int(f.stem.split("_")[-1]))
            except Exception:
                pass
        return sorted(completed)

    def load_test_checkpoint(self, test_num: int):
        path = self.checkpoint_dir / f"test_predictions_{test_num:02d}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)


# ================================================================================
# DATA LOADER
# ================================================================================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = None

    def _build_features(self, df: pd.DataFrame) -> List[str]:
        mode = self.config.FEATURE_SET_MODE

        if mode == "all":
            features = (
                self.config.BASE_META_FEATURES
                + self.config.WEATHER_FEATURES
                + [self.config.OUTAGE_LAG_FEATURE]
            )
        elif mode == "autoregressive_only":
            features = [self.config.OUTAGE_LAG_FEATURE]
        elif mode == "weather_only":
            features = self.config.WEATHER_FEATURES
        elif mode == "no_outage_lag":
            features = self.config.BASE_META_FEATURES + self.config.WEATHER_FEATURES
        else:
            raise ValueError(f"Unknown FEATURE_SET_MODE: {mode}")

        if self.config.MODEL_TYPE == "Unified":
            features = features + [self.config.CATEGORICAL_FEATURE]

        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing selected features for mode={mode}: {missing}")

        return features

    def load_and_prepare(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        logger.info(f"Loading data with shape: {df.shape}")

        missing = set(self.config.SELECTED_COLUMNS) - set(df.columns)
        if missing:
            logger.error(f"Missing columns for LAG={self.config.LAG}: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[self.config.SELECTED_COLUMNS].copy()

        if "EVENT_ID" not in df.columns:
            raise ValueError("EVENT_ID column not found in dataframe")

        df["EVENT_ID"] = pd.to_numeric(df["EVENT_ID"], errors="coerce")
        if df["EVENT_ID"].isna().any():
            n_bad = int(df["EVENT_ID"].isna().sum())
            logger.warning(f"Found {n_bad} rows with invalid EVENT_ID, dropping them")
            df = df.dropna(subset=["EVENT_ID"])

        logger.info(f"EVENT_ID range: {df['EVENT_ID'].min():.0f} to {df['EVENT_ID'].max():.0f}")

        if self.config.MODEL_TYPE == "Ensemble" and self.config.EVENT_TYPE_VALUE is not None:
            initial_size = len(df)
            df = df[df["EVENT_TYPE"] == self.config.EVENT_TYPE_VALUE].copy()
            logger.info(f"Filtered to EVENT_TYPE={self.config.EVENT_TYPE_VALUE}: {len(df)} rows (from {initial_size})")
            if len(df) < 100:
                logger.warning(f"Very small dataset after filtering: {len(df)} rows")

        initial = len(df)
        df = df.dropna()
        dropped = initial - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing values ({100*dropped/max(initial,1):.2f}%)")
        logger.info(f"Final data shape: {df.shape}")

        train_df = df[df["EVENT_ID"] <= self.config.EVENT_ID_CUTOFF].copy()
        test_df = df[df["EVENT_ID"] >= self.config.EVENT_ID_CUTOFF + 1].copy()

        logger.info(f"Train: {train_df.shape[0]} rows")
        logger.info(f"Test: {test_df.shape[0]} rows")

        if len(train_df) == 0:
            raise ValueError("Training set is empty after split")

        features = self._build_features(train_df)
        logger.info(f"Using {len(features)} features (mode={self.config.FEATURE_SET_MODE})")
        logger.info(f"Final features: {features}")

        if self.config.MODEL_TYPE == "Unified":
            if self.config.CATEGORICAL_FEATURE not in features:
                raise ValueError(f"{self.config.CATEGORICAL_FEATURE} must be in features for Unified mode")

            logger.info(
                f"Train EVENT_TYPE unique: {sorted(train_df[self.config.CATEGORICAL_FEATURE].astype(str).unique())}"
            )
            if len(test_df) > 0:
                logger.info(
                    f"Test EVENT_TYPE unique: {sorted(test_df[self.config.CATEGORICAL_FEATURE].astype(str).unique())}"
                )

            if self.config.ALGORITHM == "LightGBM":
                train_cats = train_df[self.config.CATEGORICAL_FEATURE].astype(str)
                cat_dtype = pd.CategoricalDtype(categories=sorted(train_cats.unique()))
                train_df[self.config.CATEGORICAL_FEATURE] = (
                    train_df[self.config.CATEGORICAL_FEATURE].astype(str).astype(cat_dtype)
                )

                if len(test_df) > 0:
                    test_as_str = test_df[self.config.CATEGORICAL_FEATURE].astype(str)
                    unseen = sorted(set(test_as_str.unique()) - set(cat_dtype.categories))
                    if unseen:
                        logger.warning(f"Unseen EVENT_TYPE values in test for LightGBM: {unseen}")
                        test_df = test_df[test_as_str.isin(cat_dtype.categories)].copy()
                        test_as_str = test_df[self.config.CATEGORICAL_FEATURE].astype(str)
                    test_df[self.config.CATEGORICAL_FEATURE] = test_as_str.astype(cat_dtype)

                logger.info(f"Set {self.config.CATEGORICAL_FEATURE} as categorical feature (LightGBM)")

            else:
                train_cats = train_df[self.config.CATEGORICAL_FEATURE].astype(str)
                test_cats = (
                    test_df[self.config.CATEGORICAL_FEATURE].astype(str)
                    if len(test_df) > 0 else pd.Series(dtype=str)
                )

                seen = set(train_cats.unique())
                unseen = sorted(set(test_cats.unique()) - seen)
                if unseen:
                    logger.warning(f"Unseen EVENT_TYPE values in test for XGBoost: {unseen}")
                    test_df = test_df[test_df[self.config.CATEGORICAL_FEATURE].astype(str).isin(seen)].copy()

                self.label_encoder = LabelEncoder()
                train_df[self.config.CATEGORICAL_FEATURE] = self.label_encoder.fit_transform(
                    train_df[self.config.CATEGORICAL_FEATURE].astype(str)
                )
                if len(test_df) > 0:
                    test_df[self.config.CATEGORICAL_FEATURE] = self.label_encoder.transform(
                        test_df[self.config.CATEGORICAL_FEATURE].astype(str)
                    )

                logger.info(f"Label encoded {self.config.CATEGORICAL_FEATURE} for XGBoost")
                logger.info(f"Classes: {list(self.label_encoder.classes_)}")

        return train_df, test_df, features


# ================================================================================
# BAYESIAN OPTIMIZER
# ================================================================================

class BayesianOptimizerWithCheckpoint:
    def __init__(
        self,
        config: Config,
        train_df: pd.DataFrame,
        features: List[str],
        checkpoint_manager: CheckpointManager
    ):
        self.config = config
        self.train_df = train_df
        self.features = features
        self.checkpoint_manager = checkpoint_manager
        self._interrupted = False
        self._iteration_count = 0

        def _handle_signal(signum, frame):
            logger.info(f"Received signal {signum}; will stop after current evaluation.")
            self._interrupted = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass

        if config.ALGORITHM == "LightGBM":
            self.space = [
                Integer(32, 96, name="num_leaves"),
                Integer(5, 12, name="max_depth"),
                Real(0.05, 0.2, name="learning_rate", prior="log-uniform"),
                Real(0.6, 1.0, name="feature_fraction"),
                Real(0.6, 1.0, name="bagging_fraction"),
                Real(0.0, 3.0, name="lambda_l1"),
                Real(0.0, 3.0, name="lambda_l2"),
                Integer(20, 60, name="min_data_in_leaf")
            ]
        else:
            self.space = [
                Integer(5, 12, name="max_depth"),
                Real(0.05, 0.2, name="learning_rate", prior="log-uniform"),
                Real(0.6, 1.0, name="subsample"),
                Real(0.6, 1.0, name="colsample_bytree"),
                Real(0.0, 3.0, name="reg_alpha"),
                Real(0.0, 3.0, name="reg_lambda"),
                Integer(20, 60, name="min_child_weight")
            ]

    def _objective(self, params_list):
        if self._interrupted:
            logger.info("Interrupted flag set, returning large value to stop")
            return 1e10

        self._iteration_count += 1
        logger.info(f"Bayesian iteration {self._iteration_count}/{self.config.N_BAYESIAN_CALLS}")

        try:
            if self.config.ALGORITHM == "LightGBM":
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "num_leaves": int(params_list[0]),
                    "max_depth": int(params_list[1]),
                    "learning_rate": float(params_list[2]),
                    "feature_fraction": float(params_list[3]),
                    "bagging_fraction": float(params_list[4]),
                    "lambda_l1": float(params_list[5]),
                    "lambda_l2": float(params_list[6]),
                    "min_data_in_leaf": int(params_list[7]),
                    "bagging_freq": 1,
                    "random_state": self.config.RANDOM_STATE
                }
            else:
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "max_depth": int(params_list[0]),
                    "learning_rate": float(params_list[1]),
                    "subsample": float(params_list[2]),
                    "colsample_bytree": float(params_list[3]),
                    "reg_alpha": float(params_list[4]),
                    "reg_lambda": float(params_list[5]),
                    "min_child_weight": int(params_list[6]),
                    "seed": self.config.RANDOM_STATE,
                    "verbosity": 0,
                    "tree_method": "hist"
                }

            rmse_scores = []
            n_repeats = self.config.N_CV_REPEATS if self.config.USE_REPEATED_TIME_CV else 1

            for repeat_idx in range(n_repeats):
                cv_splits = list(
                    time_block_cv_splits(
                        self.train_df,
                        self.config.N_SPLITS,
                        repeat_idx=repeat_idx,
                        jitter=self.config.CV_BLOCK_JITTER
                    )
                )

                for fold_num, (train_idx, val_idx) in enumerate(cv_splits, 1):
                    try:
                        train_fold = self.train_df.loc[train_idx]
                        val_fold = self.train_df.loc[val_idx]

                        X_train = train_fold[self.features]
                        y_train = train_fold[self.config.TARGET]
                        X_val = val_fold[self.features]
                        y_val = val_fold[self.config.TARGET]

                        if self.config.ALGORITHM == "LightGBM":
                            cat_feat = [self.config.CATEGORICAL_FEATURE] if (self.config.MODEL_TYPE == "Unified") else "auto"
                            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feat)
                            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_feat)

                            model = lgb.train(
                                params,
                                train_data,
                                num_boost_round=self.config.NUM_BOOST_ROUND,
                                valid_sets=[val_data],
                                callbacks=[
                                    lgb.early_stopping(stopping_rounds=self.config.EARLY_STOPPING_ROUNDS),
                                    lgb.log_evaluation(period=0)
                                ]
                            )
                            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
                        else:
                            train_data = xgb.DMatrix(X_train, label=y_train)
                            val_data = xgb.DMatrix(X_val, label=y_val)

                            model = xgb.train(
                                params,
                                train_data,
                                num_boost_round=self.config.NUM_BOOST_ROUND,
                                evals=[(val_data, "valid")],
                                early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                                verbose_eval=False
                            )
                            best_iter = model.best_iteration + 1 if model.best_iteration is not None else self.config.NUM_BOOST_ROUND
                            y_pred = model.predict(val_data, iteration_range=(0, best_iter))

                        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

                    except Exception as e:
                        logger.warning(f"Repeat {repeat_idx} fold {fold_num} failed: {e}")
                        continue

            if not rmse_scores:
                logger.error("All folds failed! Returning large penalty.")
                return 1e10

            avg_rmse = float(np.mean(rmse_scores))
            std_rmse = float(np.std(rmse_scores))
            logger.info(f"Repeated forward CV RMSE: {avg_rmse:.4f} ± {std_rmse:.4f} (from {len(rmse_scores)} folds)")
            return avg_rmse

        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            logger.error(traceback.format_exc())
            return 1e10

    def optimize(self, resume: bool = False) -> Dict:
        logger.info("=" * 80)
        logger.info(f"STEP 1: BAYESIAN OPTIMIZATION ({self.config.N_BAYESIAN_CALLS} calls)")
        logger.info("=" * 80)

        x0 = None
        y0 = None
        n_initial_points = self.config.N_RANDOM_STARTS

        if resume:
            iteration, best_params, all_results = self.checkpoint_manager.load_bayesian_checkpoint()
            if iteration > 0:
                logger.info(f"Resuming from iteration {iteration}")
                x0 = [r["x"] for r in all_results]
                y0 = [r["y"] for r in all_results]
                n_initial_points = max(0, self.config.N_RANDOM_STARTS - len(x0))
                self._iteration_count = iteration
            else:
                iteration = 0
        else:
            iteration = 0

        def _callback(res):
            nonlocal iteration
            iteration += 1
            all_results = [{"x": list(x), "y": float(y)} for x, y in zip(res.x_iters, res.func_vals)]
            best_params = {"x": list(res.x), "y": float(res.fun)}
            self.checkpoint_manager.save_bayesian_checkpoint(iteration, best_params, all_results)

            if self._interrupted:
                logger.info("Stopping optimization due to interrupt")
                return True
            return False

        try:
            result = gp_minimize(
                self._objective,
                self.space,
                n_calls=self.config.N_BAYESIAN_CALLS,
                n_initial_points=n_initial_points,
                x0=x0,
                y0=y0,
                random_state=self.config.RANDOM_STATE,
                callback=_callback,
                verbose=True
            )

            logger.info(f"Best CV RMSE: {result.fun:.4f}")
            logger.info("✓ Bayesian Optimization Complete!")

            if self.config.ALGORITHM == "LightGBM":
                best_params = {
                    "num_leaves": int(result.x[0]),
                    "max_depth": int(result.x[1]),
                    "learning_rate": float(result.x[2]),
                    "feature_fraction": float(result.x[3]),
                    "bagging_fraction": float(result.x[4]),
                    "lambda_l1": float(result.x[5]),
                    "lambda_l2": float(result.x[6]),
                    "min_data_in_leaf": int(result.x[7])
                }
            else:
                best_params = {
                    "max_depth": int(result.x[0]),
                    "learning_rate": float(result.x[1]),
                    "subsample": float(result.x[2]),
                    "colsample_bytree": float(result.x[3]),
                    "reg_alpha": float(result.x[4]),
                    "reg_lambda": float(result.x[5]),
                    "min_child_weight": int(result.x[6])
                }

            all_results = [{"x": list(x), "y": float(y)} for x, y in zip(result.x_iters, result.func_vals)]
            self.checkpoint_manager.save_bayesian_checkpoint(
                len(result.x_iters),
                {"x": list(result.x), "y": float(result.fun)},
                all_results
            )

            return best_params

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            logger.error(traceback.format_exc())
            raise


# ================================================================================
# SINGLE MODEL TRAINER
# ================================================================================

class SingleModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self._warned_fraction = False

    def _jitter_fraction(self, value: float, floor: float, seed: int, width: float = 0.02) -> float:
        rng = np.random.default_rng(seed)
        scale = 1.0 + rng.uniform(-width, width)
        return float(min(1.0, max(floor, value * scale)))

    def _maybe_make_stochastic_params_lgb(self, params: Dict, seed: int) -> Dict:
        p = params.copy()
        p["bagging_freq"] = 1
        ff = float(p.get("feature_fraction", 1.0))
        bf = float(p.get("bagging_fraction", 1.0))
        if ff >= 0.999:
            ff = self.config.ENSEMBLE_FRACTION_FLOOR
        if bf >= 0.999:
            bf = self.config.ENSEMBLE_FRACTION_FLOOR
        p["feature_fraction"] = self._jitter_fraction(ff, self.config.ENSEMBLE_FRACTION_FLOOR, seed)
        p["bagging_fraction"] = self._jitter_fraction(bf, self.config.ENSEMBLE_FRACTION_FLOOR, seed + 1)
        if not self._warned_fraction and (p["feature_fraction"] < 0.999 or p["bagging_fraction"] < 0.999):
            logger.info(
                f"Applied LGBM fraction floor/jitter for seed-variability runs "
                f"(feature={p['feature_fraction']:.3f}, bagging={p['bagging_fraction']:.3f})"
            )
            self._warned_fraction = True
        return p

    def _maybe_make_stochastic_params_xgb(self, params: Dict, seed: int) -> Dict:
        p = params.copy()
        ss = float(p.get("subsample", 1.0))
        cs = float(p.get("colsample_bytree", 1.0))
        if ss >= 0.999:
            ss = self.config.XGB_FRACTION_FLOOR
        if cs >= 0.999:
            cs = self.config.XGB_FRACTION_FLOOR
        p["subsample"] = self._jitter_fraction(ss, self.config.XGB_FRACTION_FLOOR, seed)
        p["colsample_bytree"] = self._jitter_fraction(cs, self.config.XGB_FRACTION_FLOOR, seed + 1)
        if not self._warned_fraction and (p["subsample"] < 0.999 or p["colsample_bytree"] < 0.999):
            logger.info(
                f"Applied XGB fraction floor/jitter for seed-variability runs "
                f"(subsample={p['subsample']:.3f}, colsample={p['colsample_bytree']:.3f})"
            )
            self._warned_fraction = True
        return p

    def _train_one_model(self, train_df: pd.DataFrame, features: List[str], best_params: Dict, seed: int):
        try:
            X_train = train_df[features]
            y_train = train_df[self.config.TARGET]

            if self.config.ALGORITHM == "LightGBM":
                params = {
                    "objective": "regression",
                    "metric": "rmse",
                    "verbosity": -1,
                    "boosting_type": "gbdt",
                    "bagging_freq": 1,
                    "random_state": seed,
                    **best_params
                }
                params = self._maybe_make_stochastic_params_lgb(params, seed)

                if self.config.BOOTSTRAP_TRAIN:
                    n = len(X_train)
                    rng = np.random.default_rng(seed)
                    idx = rng.integers(0, n, size=n)
                    Xb = X_train.iloc[idx]
                    yb = y_train.iloc[idx]
                else:
                    Xb, yb = X_train, y_train

                train_data = lgb.Dataset(
                    Xb,
                    label=yb,
                    categorical_feature=[self.config.CATEGORICAL_FEATURE] if (self.config.MODEL_TYPE == "Unified") else "auto"
                )
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.config.NUM_BOOST_ROUND,
                    callbacks=[lgb.log_evaluation(period=50)]
                )
            else:
                params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "seed": seed,
                    "verbosity": 0,
                    "tree_method": "hist",
                    **best_params
                }
                params = self._maybe_make_stochastic_params_xgb(params, seed)

                if self.config.BOOTSTRAP_TRAIN:
                    n = len(X_train)
                    rng = np.random.default_rng(seed)
                    idx = rng.integers(0, n, size=n)
                    Xb = X_train.iloc[idx]
                    yb = y_train.iloc[idx]
                else:
                    Xb, yb = X_train, y_train

                train_data = xgb.DMatrix(Xb, label=yb)
                model = xgb.train(
                    params,
                    train_data,
                    num_boost_round=self.config.NUM_BOOST_ROUND,
                    verbose_eval=50
                )

            return model

        except Exception as e:
            logger.error(f"Error training model with seed {seed}: {e}")
            logger.error(traceback.format_exc())
            raise

    def train_model(self, train_df: pd.DataFrame, features: List[str], best_params: Dict):
        logger.info("=" * 80)
        logger.info("STEP 2: TRAIN SINGLE MODEL (for export)")
        logger.info("=" * 80)

        model = self._train_one_model(train_df, features, best_params, seed=self.config.RANDOM_STATE)
        logger.info("✓ Model Training Complete!")
        return model

    def seed_ensemble_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: List[str],
        best_params: Dict,
        checkpoint_manager: CheckpointManager,
        resume: bool = False
    ):
        logger.info("=" * 80)
        logger.info(
            f"STEP 3: SEED-VARIABILITY RUNS — "
            f"{self.config.N_SEED_ENSEMBLE_MEMBERS} MODELS WITH PERMUTATION IMPORTANCE"
        )
        logger.info("=" * 80)
        logger.info("Seed runs capture training stochasticity, not full predictive uncertainty.")

        completed_tests = []
        if resume:
            completed_tests = checkpoint_manager.load_test_predictions_checkpoints()
            if completed_tests:
                logger.info(f"Resuming: {len(completed_tests)} tests already completed -> {completed_tests}")

        all_predictions: List[pd.DataFrame] = []
        all_metrics: List[Dict[str, float]] = []
        all_importance: List[pd.DataFrame] = []

        X_test = test_df[features]
        y_test = test_df[self.config.TARGET]

        for test_num in range(1, self.config.N_SEED_ENSEMBLE_MEMBERS + 1):
            if test_num in completed_tests:
                checkpoint = checkpoint_manager.load_test_checkpoint(test_num)
                if checkpoint:
                    all_predictions.append(checkpoint["predictions"])
                    if checkpoint["importance"] is not None:
                        all_importance.append(checkpoint["importance"])
                    logger.info(f"✓ Loaded seed-variability run {test_num} from checkpoint")
                    continue

            logger.info(f"Training seed-variability member {test_num}/{self.config.N_SEED_ENSEMBLE_MEMBERS}")
            seed = self.config.RANDOM_STATE + test_num

            try:
                model_i = self._train_one_model(train_df, features, best_params, seed=seed)

                if self.config.ALGORITHM == "LightGBM":
                    y_pred = model_i.predict(X_test)
                else:
                    test_data = xgb.DMatrix(X_test)
                    y_pred = model_i.predict(test_data)

                scores = evaluate_predictions(y_test, y_pred)
                logger.info(
                    f"Seed {seed} → RMSE={scores['rmse']:.4f}, "
                    f"MAE={scores['mae']:.4f}, R²={scores['r2']:.4f}"
                )

                pred_df = test_df[["Date", "CNTY_NM", "EVENT_TYPE", "EVENT_ID", self.config.TARGET]].copy()
                pred_df[f"pred_{test_num}"] = y_pred
                all_predictions.append(pred_df)

                metrics_dict = {"seed_run": test_num, "seed": seed, **scores}
                all_metrics.append(metrics_dict)

                importance_df = None
                if self.config.COMPUTE_PERMUTATION_IMPORTANCE:
                    logger.info(f"Computing permutation importance (seed-variability member {test_num})...")
                    importance_df = self._compute_permutation_importance(
                        model_i, X_test, y_test, features, test_num, scores["rmse"]
                    )
                    all_importance.append(importance_df)

                checkpoint_manager.save_test_predictions_checkpoint(test_num, pred_df, importance_df)

            except Exception as e:
                logger.error(f"Failed to train seed-variability member {test_num}: {e}")
                logger.error(traceback.format_exc())
                continue

        if not all_predictions:
            raise RuntimeError("All seed-variability runs failed")

        meta_cols = ["Date", "CNTY_NM", "EVENT_TYPE", "EVENT_ID", self.config.TARGET]
        combined_predictions = all_predictions[0][meta_cols].copy()
        for pred_df in all_predictions:
            pred_cols = [c for c in pred_df.columns if c.startswith("pred_")]
            for c in pred_cols:
                combined_predictions[c] = pred_df[c].values

        metrics_df = pd.DataFrame(all_metrics)
        logger.info(
            f"✓ Completed {len(all_predictions)}/{self.config.N_SEED_ENSEMBLE_MEMBERS} seed-variability runs"
        )
        return combined_predictions, metrics_df, all_importance

    def _compute_permutation_importance(self, model, X_test, y_test, features, test_num, baseline_rmse):
        logger.info(f"Computing permutation importance for seed-run {test_num} ({len(features)} features)")
        importance_list = []

        for idx, feature in enumerate(features, 1):
            try:
                is_event_type = (feature == self.config.CATEGORICAL_FEATURE and self.config.MODEL_TYPE == "Unified")
                X_test_permuted = X_test.copy()

                rng = np.random.default_rng(self.config.RANDOM_STATE + test_num + idx)
                shuffled_values = rng.permutation(X_test[feature].values)
                X_test_permuted.loc[:, feature] = shuffled_values

                if self.config.ALGORITHM == "LightGBM":
                    y_pred_permuted = model.predict(X_test_permuted)
                else:
                    test_data_permuted = xgb.DMatrix(X_test_permuted)
                    y_pred_permuted = model.predict(test_data_permuted)

                permuted_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_permuted)))
                importance = permuted_rmse - baseline_rmse

                importance_list.append({
                    "feature": feature,
                    "feature_type": "categorical" if is_event_type else "numerical",
                    "group": categorize_feature_group(feature, self.config),
                    "seed_run": test_num,
                    "importance": importance,
                    "rmse_baseline": baseline_rmse,
                    "rmse_shuffled": permuted_rmse
                })

            except Exception as e:
                if feature == self.config.CATEGORICAL_FEATURE:
                    logger.error(f"CRITICAL ERROR computing EVENT_TYPE importance: {e}")
                    logger.error(traceback.format_exc())
                    raise
                else:
                    logger.warning(f"Failed to compute importance for {feature}: {e}")
                    continue

        importance_df = pd.DataFrame(importance_list)
        if importance_df.empty:
            raise ValueError("Importance DataFrame cannot be empty")

        if self.config.MODEL_TYPE == "Unified":
            if self.config.CATEGORICAL_FEATURE not in importance_df["feature"].values:
                raise ValueError("EVENT_TYPE must be in importance results for Unified mode")

        return importance_df


# ================================================================================
# REPEATED FORWARD CV EVALUATION
# ================================================================================

def run_repeated_forward_cv(
    config: Config,
    train_df: pd.DataFrame,
    features: List[str],
    best_params: Dict
) -> pd.DataFrame:
    """
    Separate from seed-variability test runs.
    This estimates stability across repeated forward-CV allocations.
    """
    rows = []
    trainer = SingleModelTrainer(config)

    n_repeats = config.N_CV_REPEATS if config.USE_REPEATED_TIME_CV else 1

    logger.info("=" * 80)
    logger.info(f"STEP 2A: REPEATED FORWARD CV EVALUATION ({n_repeats} repeats)")
    logger.info("=" * 80)

    for repeat_idx in range(n_repeats):
        splits = list(
            time_block_cv_splits(
                train_df,
                config.N_SPLITS,
                repeat_idx=repeat_idx,
                jitter=config.CV_BLOCK_JITTER
            )
        )

        logger.info(f"Repeated forward CV repeat {repeat_idx + 1}/{n_repeats}: {len(splits)} folds")

        for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
            try:
                tr = train_df.loc[train_idx]
                va = train_df.loc[val_idx]

                model = trainer._train_one_model(
                    tr, features, best_params,
                    seed=config.RANDOM_STATE + repeat_idx
                )

                X_val = va[features]
                y_val = va[config.TARGET]

                if config.ALGORITHM == "LightGBM":
                    y_pred = model.predict(X_val)
                else:
                    y_pred = model.predict(xgb.DMatrix(X_val))

                scores = evaluate_predictions(y_val, y_pred)
                rows.append({
                    "repeat_idx": repeat_idx,
                    "fold_num": fold_num,
                    **scores
                })

            except Exception as e:
                logger.warning(f"Repeated CV failed at repeat {repeat_idx}, fold {fold_num}: {e}")
                continue

    return pd.DataFrame(rows)


# ================================================================================
# RESULTS SAVER
# ================================================================================

class ResultsSaver:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.summary_dir = output_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = output_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

    def save_run_config(self, config: Config):
        path = self.summary_dir / "run_config.json"
        with open(path, "w") as f:
            json.dump(convert_to_json_serializable(config.__dict__), f, indent=2)
        logger.info("✓ Saved: run_config.json")

    def save_best_params(self, best_params: Dict[str, Any]):
        path = self.summary_dir / "best_params.json"
        with open(path, "w") as f:
            json.dump(convert_to_json_serializable(best_params), f, indent=2)
        logger.info("✓ Saved: best_params.json")

    def save_model(self, model, algorithm: str):
        try:
            if algorithm == "LightGBM":
                path = self.models_dir / "lightgbm_model.txt"
                model.save_model(str(path))
            else:
                path = self.models_dir / "xgboost_model.json"
                model.save_model(str(path))
            logger.info(f"✓ Saved model: {path.name}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def save_predictions(self, predictions: pd.DataFrame):
        path = self.summary_dir / "predictions.csv"
        predictions.to_csv(path, index=False)
        logger.info(f"✓ Saved: predictions.csv ({len(predictions)} rows)")

    def save_seed_variability_metrics(self, metrics: pd.DataFrame):
        path = self.summary_dir / "metrics_seed_variability.csv"
        metrics.to_csv(path, index=False)
        logger.info(f"✓ Saved: metrics_seed_variability.csv ({len(metrics)} rows)")

    def save_repeated_cv_metrics(self, metrics: pd.DataFrame):
        if metrics.empty:
            logger.warning("No repeated forward CV metrics to save")
            return
        path = self.summary_dir / "metrics_repeated_forward_cv.csv"
        metrics.to_csv(path, index=False)
        logger.info(f"✓ Saved: metrics_repeated_forward_cv.csv ({len(metrics)} rows)")

        summary = pd.DataFrame([{
            "rmse_mean": float(metrics["rmse"].mean()),
            "rmse_std": float(metrics["rmse"].std(ddof=1)) if len(metrics) > 1 else 0.0,
            "mae_mean": float(metrics["mae"].mean()),
            "mae_std": float(metrics["mae"].std(ddof=1)) if len(metrics) > 1 else 0.0,
            "r2_mean": float(metrics["r2"].mean()),
            "r2_std": float(metrics["r2"].std(ddof=1)) if len(metrics) > 1 else 0.0,
            "n_rows": int(len(metrics))
        }])
        summary_path = self.summary_dir / "metrics_repeated_forward_cv_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("✓ Saved: metrics_repeated_forward_cv_summary.csv")

    def save_naive_metrics(self, metrics: pd.DataFrame):
        path = self.summary_dir / "naive_baseline_metrics.csv"
        metrics.to_csv(path, index=False)
        logger.info("✓ Saved: naive_baseline_metrics.csv")

    def save_naive_predictions(self, naive_predictions: pd.DataFrame):
        if naive_predictions.empty:
            logger.warning("No naive predictions to save")
            return
        path = self.summary_dir / "naive_predictions.csv"
        naive_predictions.to_csv(path, index=False)
        logger.info(f"✓ Saved: naive_predictions.csv ({len(naive_predictions)} rows)")

    def save_naive_summary_statistics(self, naive_predictions: pd.DataFrame, target_col: str):
        if naive_predictions.empty:
            logger.warning("No naive predictions available for summary statistics")
            return
        summary_cols = ["Date", "CNTY_NM", "EVENT_TYPE", "EVENT_ID", target_col, "naive_prediction"]
        existing_cols = [c for c in summary_cols if c in naive_predictions.columns]
        summary = naive_predictions[existing_cols].copy()
        if target_col in summary.columns and "naive_prediction" in summary.columns:
            summary["naive_residual"] = summary[target_col] - summary["naive_prediction"]
            summary["naive_absolute_error"] = (summary[target_col] - summary["naive_prediction"]).abs()
        path = self.summary_dir / "naive_predictions_summary.csv"
        summary.to_csv(path, index=False)
        logger.info(f"✓ Saved: naive_predictions_summary.csv ({len(summary)} rows)")

    def save_baseline_comparison(self, comparison_df: pd.DataFrame):
        if comparison_df.empty:
            return
        path = self.summary_dir / "baseline_comparison.csv"
        comparison_df.to_csv(path, index=False)
        logger.info("✓ Saved: baseline_comparison.csv")

    def save_summary_statistics(self, predictions: pd.DataFrame, target_col: str):
        try:
            pred_cols = [c for c in predictions.columns if c.startswith("pred_")]
            if not pred_cols:
                logger.warning("No prediction columns found for summary statistics")
                return

            summary = predictions[["Date", "CNTY_NM", "EVENT_TYPE", "EVENT_ID", target_col]].copy()
            pred_values = predictions[pred_cols].to_numpy(dtype=np.float32)

            summary["Pred_mean"] = pred_values.mean(axis=1)
            summary["seed_variability_std"] = pred_values.std(axis=1)
            summary["seed_variability_min"] = pred_values.min(axis=1)
            summary["seed_variability_max"] = pred_values.max(axis=1)
            summary["N_seed_models"] = len(pred_cols)

            path = self.summary_dir / "predictions_summary.csv"
            summary.to_csv(path, index=False)
            logger.info(f"✓ Saved: predictions_summary.csv ({len(summary)} rows)")

        except Exception as e:
            logger.error(f"Error in save_summary_statistics: {e}")
            logger.error(traceback.format_exc())
            raise

    def save_permutation_importance(self, imp_df: pd.DataFrame, test_num: int):
        path = self.summary_dir / f"permutation_importance_seedrun{test_num:02d}.csv"
        imp_df.sort_values("importance", ascending=False).to_csv(path, index=False)
        logger.info(f"✓ Saved: {path.name}")

    def save_aggregated_permutation_importance(self, all_importance: List[pd.DataFrame]):
        if not all_importance:
            logger.warning("No importance data to aggregate")
            return

        combined = pd.concat(all_importance, ignore_index=True)
        agg = (
            combined
            .groupby(["feature", "feature_type"], as_index=False)
            .agg(
                mean_importance=("importance", "mean"),
                std_importance=("importance", "std"),
                n_seed_runs=("seed_run", "nunique"),
                mean_rmse_base=("rmse_baseline", "mean")
            )
            .sort_values("mean_importance", ascending=False)
        )

        path = self.summary_dir / "permutation_importance_aggregated.csv"
        agg.to_csv(path, index=False)
        logger.info("✓ Saved: permutation_importance_aggregated.csv")

    def save_grouped_permutation_importance(self, all_importance: List[pd.DataFrame]):
        if not all_importance:
            return

        combined = pd.concat(all_importance, ignore_index=True)
        if "group" not in combined.columns:
            logger.warning("No feature group column found; skipping grouped importance save")
            return

        grouped = (
            combined
            .groupby("group", as_index=False)
            .agg(
                mean_importance=("importance", "mean"),
                std_importance=("importance", "std"),
                n_seed_runs=("seed_run", "nunique")
            )
            .sort_values("mean_importance", ascending=False)
        )

        path = self.summary_dir / "permutation_importance_grouped.csv"
        grouped.to_csv(path, index=False)
        logger.info("✓ Saved: permutation_importance_grouped.csv")

    def save_experiment_comparison(self, row: Dict[str, Any]):
        path = self.summary_dir / "experiment_comparison.csv"
        row_df = pd.DataFrame([row])
        if path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old, row_df], ignore_index=True)
            new.to_csv(path, index=False)
        else:
            row_df.to_csv(path, index=False)
        logger.info("✓ Updated: experiment_comparison.csv")

    def save_model_summary_row(
        self,
        config: Config,
        seed_variability_metrics: pd.DataFrame,
        repeated_cv_metrics: pd.DataFrame,
        naive_metrics: pd.DataFrame,
    ):
        row = {
            "model_type": config.MODEL_TYPE,
            "algorithm": config.ALGORITHM,
            "lag": config.LAG,
            "feature_set_mode": config.FEATURE_SET_MODE,
            "weather_source_mode": config.WEATHER_SOURCE_MODE,
            "event_type_value": config.EVENT_TYPE_VALUE,
            "seed_var_rmse_mean": float(seed_variability_metrics["rmse"].mean()) if not seed_variability_metrics.empty else np.nan,
            "seed_var_mae_mean": float(seed_variability_metrics["mae"].mean()) if not seed_variability_metrics.empty else np.nan,
            "seed_var_r2_mean": float(seed_variability_metrics["r2"].mean()) if not seed_variability_metrics.empty else np.nan,
            "repeated_cv_rmse_mean": float(repeated_cv_metrics["rmse"].mean()) if not repeated_cv_metrics.empty else np.nan,
            "repeated_cv_rmse_std": float(repeated_cv_metrics["rmse"].std(ddof=1)) if len(repeated_cv_metrics) > 1 else np.nan,
            "repeated_cv_mae_mean": float(repeated_cv_metrics["mae"].mean()) if not repeated_cv_metrics.empty else np.nan,
            "repeated_cv_mae_std": float(repeated_cv_metrics["mae"].std(ddof=1)) if len(repeated_cv_metrics) > 1 else np.nan,
            "repeated_cv_r2_mean": float(repeated_cv_metrics["r2"].mean()) if not repeated_cv_metrics.empty else np.nan,
            "repeated_cv_r2_std": float(repeated_cv_metrics["r2"].std(ddof=1)) if len(repeated_cv_metrics) > 1 else np.nan,
            "naive_rmse": float(naive_metrics["rmse"].iloc[0]) if not naive_metrics.empty else np.nan,
            "naive_mae": float(naive_metrics["mae"].iloc[0]) if not naive_metrics.empty else np.nan,
            "naive_r2": float(naive_metrics["r2"].iloc[0]) if not naive_metrics.empty else np.nan,
        }

        path = self.output_dir.parent / "ablation_summary_all_runs.csv"
        row_df = pd.DataFrame([row])

        if path.exists():
            old = pd.read_csv(path)
            new = pd.concat([old, row_df], ignore_index=True)
            new = new.drop_duplicates(
                subset=["model_type", "algorithm", "lag", "feature_set_mode", "weather_source_mode", "event_type_value"],
                keep="last"
            )
            new.to_csv(path, index=False)
        else:
            row_df.to_csv(path, index=False)

        logger.info(f"✓ Updated: {path.name}")


# ================================================================================
# BASELINE
# ================================================================================

def build_naive_predictions(test_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    lag_col = config.OUTAGE_LAG_FEATURE
    if lag_col not in test_df.columns:
        raise ValueError(f"Missing lag column for naive baseline: {lag_col}")

    base_cols = ["Date", "CNTY_NM", "EVENT_TYPE", "EVENT_ID", config.TARGET]
    existing_cols = [c for c in base_cols if c in test_df.columns]
    naive_df = test_df[existing_cols].copy()
    naive_df["naive_prediction"] = test_df[lag_col].values
    if config.TARGET in naive_df.columns:
        naive_df["naive_residual"] = naive_df[config.TARGET] - naive_df["naive_prediction"]
        naive_df["naive_absolute_error"] = (naive_df[config.TARGET] - naive_df["naive_prediction"]).abs()
    naive_df["naive_source_column"] = lag_col
    naive_df["lag"] = config.LAG
    naive_df["model"] = "naive_persistence"
    return naive_df


def evaluate_naive_baseline(test_df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    naive_predictions = build_naive_predictions(test_df, config)
    y_true = naive_predictions[config.TARGET].values
    y_pred = naive_predictions["naive_prediction"].values
    scores = evaluate_predictions(y_true, y_pred)

    metrics = pd.DataFrame([{
        "model": "naive_persistence",
        "lag": config.LAG,
        **scores
    }])
    return metrics, naive_predictions


# ================================================================================
# MAIN EXECUTION FUNCTIONS
# ================================================================================

def run_single_model(df: pd.DataFrame, config: Config, base_output_dir: Path, resume: bool = False):
    output_dir = config.get_output_path(base_output_dir)
    logger.info(f"Output directory: {output_dir}")

    checkpoint_manager = CheckpointManager(output_dir)
    data_loader = DataLoader(config)

    train_df, test_df, features = data_loader.load_and_prepare(df)

    no_test_rows = test_df.shape[0] == 0
    if no_test_rows:
        logger.warning("No test rows found for this configuration. Will skip testing phase.")

    optimizer = BayesianOptimizerWithCheckpoint(config, train_df, features, checkpoint_manager)
    best_params = optimizer.optimize(resume=resume)

    repeated_cv_metrics = run_repeated_forward_cv(config, train_df, features, best_params)

    trainer = SingleModelTrainer(config)
    model = trainer.train_model(train_df, features, best_params)
    checkpoint_manager.save_model_checkpoint(model, config.ALGORITHM)

    if no_test_rows:
        predictions = pd.DataFrame()
        seed_variability_metrics = pd.DataFrame()
        all_importance = []
        naive_metrics = pd.DataFrame()
        naive_predictions = pd.DataFrame()
        baseline_comparison = pd.DataFrame()
    else:
        predictions, seed_variability_metrics, all_importance = trainer.seed_ensemble_predict(
            train_df, test_df, features, best_params,
            checkpoint_manager=checkpoint_manager, resume=resume
        )
        baseline_comparison = pd.DataFrame()
        if config.OUTAGE_LAG_FEATURE in test_df.columns:
            naive_metrics, naive_predictions = evaluate_naive_baseline(test_df, config)
            baseline_comparison = compare_to_baseline(seed_variability_metrics[["rmse", "mae", "r2"]], naive_metrics)
        else:
            naive_metrics = pd.DataFrame()
            naive_predictions = pd.DataFrame()
            logger.warning(
                f"Skipping naive baseline because {config.OUTAGE_LAG_FEATURE} is not available "
                f"in the selected test features/data."
            )

    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    saver = ResultsSaver(output_dir)
    saver.save_run_config(config)
    saver.save_best_params(best_params)
    saver.save_model(model, config.ALGORITHM)
    saver.save_repeated_cv_metrics(repeated_cv_metrics)

    if not predictions.empty:
        logger.warning(
            "Seed-variability metrics reflect optimization/subsampling sensitivity only; "
            "they should not be interpreted as formal predictive uncertainty intervals."
        )

        saver.save_seed_variability_metrics(seed_variability_metrics)
        if not naive_metrics.empty:
            saver.save_naive_metrics(naive_metrics)
        if not naive_predictions.empty:
            saver.save_naive_predictions(naive_predictions)
            saver.save_naive_summary_statistics(naive_predictions, config.TARGET)
        saver.save_baseline_comparison(baseline_comparison)

        if config.COMPUTE_PERMUTATION_IMPORTANCE and all_importance:
            try:
                for imp_df in all_importance:
                    test_num = int(imp_df["seed_run"].iloc[0])
                    saver.save_permutation_importance(imp_df, test_num)
                saver.save_aggregated_permutation_importance(all_importance)
                saver.save_grouped_permutation_importance(all_importance)
            except Exception as e:
                logger.error(f"Failed to save permutation importance: {e}")
                logger.error(traceback.format_exc())

        try:
            saver.save_predictions(predictions)
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            logger.error(traceback.format_exc())

        try:
            saver.save_summary_statistics(predictions, config.TARGET)
        except MemoryError as e:
            logger.error(f"Failed to save summary statistics due to memory error: {e}")
            logger.warning("Summary statistics NOT saved (can be regenerated from predictions.csv)")
        except Exception as e:
            logger.error(f"Unexpected error saving summary statistics: {e}")
            logger.error(traceback.format_exc())

        experiment_row = {
            "model_type": config.MODEL_TYPE,
            "algorithm": config.ALGORITHM,
            "lag": config.LAG,
            "feature_set_mode": config.FEATURE_SET_MODE,
            "weather_source_mode": config.WEATHER_SOURCE_MODE,
            "event_type_value": config.EVENT_TYPE_VALUE,

            "seed_var_rmse_mean": float(seed_variability_metrics["rmse"].mean()) if not seed_variability_metrics.empty else np.nan,
            "seed_var_mae_mean": float(seed_variability_metrics["mae"].mean()) if not seed_variability_metrics.empty else np.nan,
            "seed_var_r2_mean": float(seed_variability_metrics["r2"].mean()) if not seed_variability_metrics.empty else np.nan,

            "repeated_cv_rmse_mean": float(repeated_cv_metrics["rmse"].mean()) if not repeated_cv_metrics.empty else np.nan,
            "repeated_cv_rmse_std": float(repeated_cv_metrics["rmse"].std(ddof=1)) if len(repeated_cv_metrics) > 1 else np.nan,
            "repeated_cv_mae_mean": float(repeated_cv_metrics["mae"].mean()) if not repeated_cv_metrics.empty else np.nan,
            "repeated_cv_mae_std": float(repeated_cv_metrics["mae"].std(ddof=1)) if len(repeated_cv_metrics) > 1 else np.nan,
            "repeated_cv_r2_mean": float(repeated_cv_metrics["r2"].mean()) if not repeated_cv_metrics.empty else np.nan,
            "repeated_cv_r2_std": float(repeated_cv_metrics["r2"].std(ddof=1)) if len(repeated_cv_metrics) > 1 else np.nan,

            "naive_rmse": float(naive_metrics["rmse"].iloc[0]) if not naive_metrics.empty else np.nan,
            "naive_mae": float(naive_metrics["mae"].iloc[0]) if not naive_metrics.empty else np.nan,
            "naive_r2": float(naive_metrics["r2"].iloc[0]) if not naive_metrics.empty else np.nan,
        }
        saver.save_experiment_comparison(experiment_row)

        saver.save_model_summary_row(
            config=config,
            seed_variability_metrics=seed_variability_metrics,
            repeated_cv_metrics=repeated_cv_metrics,
            naive_metrics=naive_metrics,
        )

    else:
        logger.info("No predictions/metrics to save because test was skipped.")

    logger.info("=" * 80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("=" * 80)


def run_ensemble_models(df: pd.DataFrame, config: Config, base_output_dir: Path, resume: bool = False):
    logger.info("=" * 80)
    logger.info("ENSEMBLE MODE: Training separate models for each EVENT_TYPE")
    logger.info("=" * 80)

    event_types = sorted(df["EVENT_TYPE"].dropna().astype(str).unique())
    logger.info(f"Found {len(event_types)} unique EVENT_TYPEs: {event_types}")

    for event_type in event_types:
        logger.info("=" * 80)
        logger.info(f"TRAINING MODEL FOR EVENT_TYPE: {event_type}")
        logger.info("=" * 80)

        try:
            event_config = Config()
            event_config.MODEL_TYPE = config.MODEL_TYPE
            event_config.ALGORITHM = config.ALGORITHM
            event_config.LAG = config.LAG
            event_config.EVENT_TYPE_VALUE = event_type
            event_config.COMPUTE_PERMUTATION_IMPORTANCE = config.COMPUTE_PERMUTATION_IMPORTANCE
            event_config.ENSEMBLE_RETRAIN_EACH_TEST = config.ENSEMBLE_RETRAIN_EACH_TEST
            event_config.BOOTSTRAP_TRAIN = config.BOOTSTRAP_TRAIN
            event_config.ENSEMBLE_FRACTION_FLOOR = config.ENSEMBLE_FRACTION_FLOOR
            event_config.XGB_FRACTION_FLOOR = config.XGB_FRACTION_FLOOR
            event_config.N_SEED_ENSEMBLE_MEMBERS = config.N_SEED_ENSEMBLE_MEMBERS
            event_config.N_CV_REPEATS = config.N_CV_REPEATS
            event_config.CV_BLOCK_JITTER = config.CV_BLOCK_JITTER
            event_config.USE_REPEATED_TIME_CV = config.USE_REPEATED_TIME_CV
            event_config.FEATURE_SET_MODE = config.FEATURE_SET_MODE
            event_config.WEATHER_SOURCE_MODE = config.WEATHER_SOURCE_MODE
            event_config.__post_init__()

            run_single_model(df, event_config, base_output_dir, resume)
            logger.info(f"✓ Completed EVENT_TYPE: {event_type}")

        except Exception as e:
            logger.error(f"Failed to train model for EVENT_TYPE {event_type}: {e}")
            logger.error(traceback.format_exc())
            continue


def main(
    parquet_file: str,
    model_type: str = "Unified",
    algorithm: str = "LightGBM",
    lag: int = 1,
    base_output_dir: Optional[str] = None,
    resume: bool = False,
    compute_importance: bool = True,
    feature_set_mode: str = "all",
    weather_source_mode: str = "lagged_observed",
):
    try:
        config = Config()
        config.MODEL_TYPE = model_type
        config.ALGORITHM = algorithm
        config.LAG = lag
        config.COMPUTE_PERMUTATION_IMPORTANCE = compute_importance
        config.FEATURE_SET_MODE = feature_set_mode
        config.WEATHER_SOURCE_MODE = weather_source_mode
        config.__post_init__()

        if base_output_dir is None:
            base_output_dir = Path.cwd() / "outputs"
        else:
            base_output_dir = Path(base_output_dir)

        output_dir = config.get_output_path(base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / "python_training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        logger.info("=" * 80)
        logger.info("IMPROVED OPTIMIZED MODEL TRAINING - EVENT_ID VERSION")
        logger.info("=" * 80)
        logger.info(f"Loading data from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        logger.info(f"Data loaded: {df.shape}")

        logger.info(f"Model Type: {config.MODEL_TYPE}")
        logger.info(f"Algorithm: {config.ALGORITHM}")
        logger.info(f"Lag: {config.LAG}")
        logger.info(f"Feature Set Mode: {config.FEATURE_SET_MODE}")
        logger.info(f"Weather Source Mode: {config.WEATHER_SOURCE_MODE}")
        logger.info(f"EVENT_ID Cutoff: {config.EVENT_ID_CUTOFF}")
        logger.info(f"Resume: {resume}")
        logger.info(f"Compute Permutation Importance: {config.COMPUTE_PERMUTATION_IMPORTANCE}")
        logger.info(f"Output Directory: {output_dir.absolute()}")
        logger.info(f"Seed-Variability Members: {config.N_SEED_ENSEMBLE_MEMBERS}")
        logger.info(f"Repeated Forward CV Repeats: {config.N_CV_REPEATS}")

        if config.MODEL_TYPE == "Ensemble":
            run_ensemble_models(df, config, base_output_dir, resume)
        else:
            run_single_model(df, config, base_output_dir, resume)

        logger.info("=" * 80)
        logger.info("ALL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED WITH ERROR:")
        logger.error("=" * 80)
        logger.error(str(e))
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Improved optimized model training script (single feature-set mode per process)."
    )
    parser.add_argument("parquet_file", help="Input parquet file")
    parser.add_argument("model_type", choices=["Unified", "Ensemble"], help="Model type")
    parser.add_argument("algorithm", choices=["LightGBM", "XGBoost"], help="Algorithm")
    parser.add_argument("lag", type=int, choices=[1, 12, 24], help="Forecast lag")
    parser.add_argument("output_dir", nargs="?", default=None, help="Base output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from prior outputs when possible")
    parser.add_argument("--no-importance", action="store_true", help="Disable permutation importance")
    parser.add_argument(
        "--feature-set",
        dest="feature_set_mode",
        required=True,
        choices=["all", "autoregressive_only", "weather_only", "no_outage_lag"],
        help="Exactly one feature-set mode to run in this process",
    )
    parser.add_argument(
        "--weather-source",
        dest="weather_source_mode",
        default="lagged_observed",
        choices=["lagged_observed", "forecast"],
        help="Weather feature source mode",
    )

    args = parser.parse_args()

    if "," in args.feature_set_mode:
        raise SystemExit("Only one --feature-set value may be passed per job.")

    main(
        parquet_file=args.parquet_file,
        model_type=args.model_type,
        algorithm=args.algorithm,
        lag=args.lag,
        base_output_dir=args.output_dir,
        resume=args.resume,
        compute_importance=not args.no_importance,
        feature_set_mode=args.feature_set_mode,
        weather_source_mode=args.weather_source_mode,
    )
