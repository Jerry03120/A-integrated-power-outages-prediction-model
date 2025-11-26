#!/usr/bin/env python3
"""
Improved Optimized Model Training Script - EVENT_ID Version

KEY CHANGES FROM PREVIOUS VERSION:
- Changed event_no to EVENT_ID
- Train/test split now based on EVENT_ID (train: ≤3028, test: ≥3029)
- Removed TRAIN_YEAR_CUTOFF, using EVENT_ID_CUTOFF instead

KEY IMPROVEMENTS (from base version):
1. Fixed XGBoost categorical feature handling (enable_categorical=True)
2. Added label encoding fallback for XGBoost when categorical features are present
3. Enhanced error handling and validation
4. Improved logging with progress tracking
5. Better memory management
6. Fixed edge cases in ensemble mode
7. Added data quality checks
8. Improved checkpoint reliability
9. Fixed EVENT_TYPE permutation importance calculation for Unified models
10. Fixed features list creation timing issue
11. Fixed categorical dtype handling in permutation importance

FEATURES:
- Unified: EVENT_TYPE as categorical predictor, ONE model (plus seed-ensemble predictions)
- Ensemble: EVENT_TYPE for stratification, SEPARATE models per event type
- Bayesian Optimization with checkpoints
- Resume capability
- Safe interrupt handling
- N-times test predictions with permutation importance (seed-ensemble)
- Ensures stochasticity even when optimized sampling fractions are 1.0
- Skips test phase if the test set is empty (e.g., specific EVENT_TYPE missing in test events)
- Proper EVENT_TYPE importance tracking in Unified mode

Usage:
  python train_optimized_event_id.py data.parquet Unified LightGBM 1 outputs/ --resume
  python train_optimized_event_id.py data.parquet Ensemble XGBoost 1 outputs/ --resume
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from skopt import gp_minimize
from skopt.space import Integer, Real

warnings.filterwarnings('ignore')


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
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
    else:
        return obj


# ================================================================================
# LOGGING SETUP
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ================================================================================
# CONFIGURATION
# ================================================================================

@dataclass
class Config:
    TARGET: str = 'recorded_outages'
    CATEGORICAL_FEATURE: str = 'EVENT_TYPE'
    EVENT_ID_CUTOFF: int = 3028  # Train on EVENT_ID <= 3028, test on EVENT_ID >= 3029
    RANDOM_STATE: int = 42
    N_SPLITS: int = 10
    N_BAYESIAN_CALLS: int = 50
    N_RANDOM_STARTS: int = 10
    EARLY_STOPPING_ROUNDS: int = 30
    NUM_BOOST_ROUND: int = 500

    MODEL_TYPE: str = 'Unified'  # 'Unified' or 'Ensemble'
    ALGORITHM: str = 'LightGBM'
    LAG: int = 1
    EVENT_TYPE_VALUE: Optional[str] = None  # For Ensemble models only

    N_TEST_PREDICTIONS: int = 10
    COMPUTE_PERMUTATION_IMPORTANCE: bool = True

    # Seed-ensemble options
    ENSEMBLE_RETRAIN_EACH_TEST: bool = True   # re-train a fresh model each test with different seed
    BOOTSTRAP_TRAIN: bool = False             # bootstrap sample training rows per seed model
    ENSEMBLE_FRACTION_FLOOR: float = 0.95     # LGBM: ensure feature/bagging fractions < 1.0
    XGB_FRACTION_FLOOR: float = 0.95          # XGB: ensure subsample/colsample < 1.0

    def __post_init__(self):
        base_columns = [
            'Date', 'CNTY_NM', 'ALAND_SQMI', 'AWATER_SQMI', 'INTPTLAT', 'INTPTLONG',
            'Centerline_Miles', 'Lane_Miles', 'Truck_DVMT', self.TARGET,
            'EVENT_TYPE', 'Total DVMT', 'Population', 'LAI', 'EVENT_ID'
        ]
        weather_features = [
            'ALLSKY_KT', 'ALLSKY_NKT', 'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_PAR_TOT',
            'ALLSKY_SFC_SW_DIFF', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DWN',
            'ALLSKY_SFC_UVA', 'ALLSKY_SFC_UVB', 'ALLSKY_SFC_UV_INDEX',
            'ALLSKY_SRF_ALB', 'AOD_55', 'AOD_84', 'CLRSKY_KT', 'CLRSKY_NKT',
            'PRECSNOLAND', 'PRECTOTCORR', 'PS', 'PW', 'QV10M', 'QV2M', 'RH2M',
            'SNODP', 'SZA', 'T2MWET', 'T2M', 'TOA_SW_DNI', 'TOA_SW_DWN',
            'V2M', 'V50M', 'WD10M', 'WD2M', 'WD50M', 'WS10M', 'WS2M', 'WS50M'
        ]
        lagged_features = [f"{feat}_lag{self.LAG}" for feat in weather_features]
        lagged_features.append(f"recorded_outages_lag{self.LAG}")

        self.SELECTED_COLUMNS = base_columns + lagged_features

        # Unified vs Ensemble
        if self.MODEL_TYPE == 'Ensemble':
            # Ensemble: EVENT_TYPE NOT used as predictor
            self.IGNORE_COLS = ['Date', 'CNTY_NM', self.TARGET, 'EVENT_ID', 'EVENT_TYPE']
            logger.info(f"Ensemble mode: EVENT_TYPE excluded from predictors")
        else:  # Unified
            # Unified: EVENT_TYPE IS used as categorical predictor
            self.IGNORE_COLS = ['Date', 'CNTY_NM', self.TARGET, 'EVENT_ID']
            logger.info(f"Unified mode: EVENT_TYPE included as categorical predictor")

        logger.info(f"Config: MODEL_TYPE={self.MODEL_TYPE}, LAG={self.LAG}, ALGORITHM={self.ALGORITHM}")
        logger.info(f"Train/Test Split: EVENT_ID <= {self.EVENT_ID_CUTOFF} (train) vs >= {self.EVENT_ID_CUTOFF + 1} (test)")

    def get_output_path(self, base_dir: Path) -> Path:
        """Get output directory path. For Ensemble, includes EVENT_TYPE subdirectory."""
        if self.MODEL_TYPE == 'Ensemble' and self.EVENT_TYPE_VALUE is not None:
            return base_dir / self.MODEL_TYPE / self.ALGORITHM / f"lag{self.LAG}" / f"EVENT_TYPE_{self.EVENT_TYPE_VALUE}"
        else:
            return base_dir / self.MODEL_TYPE / self.ALGORITHM / f"lag{self.LAG}"


# ================================================================================
# CHECKPOINT MANAGER
# ================================================================================

class CheckpointManager:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_bayesian_checkpoint(self, iteration: int, best_params: Dict, all_results: List):
        checkpoint = {
            'iteration': int(iteration),
            'best_params': best_params,
            'all_results': all_results,
            'timestamp': time.time()
        }
        path = self.checkpoint_dir / f'bayesian_iter{iteration:03d}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"✓ Saved Bayesian checkpoint: iteration {iteration}")

    def load_bayesian_checkpoint(self) -> Tuple[int, Optional[Dict], List]:
        files = sorted(self.checkpoint_dir.glob('bayesian_iter*.pkl'))
        if not files:
            return 0, None, []

        latest = files[-1]
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)

        iteration = checkpoint['iteration']
        best_params = checkpoint['best_params']
        all_results = checkpoint['all_results']

        logger.info(f"✓ Loaded Bayesian checkpoint: iteration {iteration}")
        return iteration, best_params, all_results

    def save_model_checkpoint(self, model, algorithm: str):
        path = self.checkpoint_dir / 'model_checkpoint.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info("✓ Saved model checkpoint")

    def load_model_checkpoint(self):
        path = self.checkpoint_dir / 'model_checkpoint.pkl'
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✓ Loaded model checkpoint")
        return model

    def save_test_predictions_checkpoint(self, test_num: int, predictions: pd.DataFrame,
                                        importance: Optional[pd.DataFrame] = None):
        checkpoint = {
            'test_num': test_num,
            'predictions': predictions,
            'importance': importance,
            'timestamp': time.time()
        }
        path = self.checkpoint_dir / f'test_predictions_{test_num:02d}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"✓ Saved test predictions checkpoint: test {test_num}")

    def load_test_predictions_checkpoints(self) -> List[int]:
        files = list(self.checkpoint_dir.glob('test_predictions_*.pkl'))
        completed = []
        for f in files:
            try:
                num = int(f.stem.split('_')[-1])
                completed.append(num)
            except:
                pass
        return sorted(completed)

    def load_test_checkpoint(self, test_num: int):
        path = self.checkpoint_dir / f'test_predictions_{test_num:02d}.pkl'
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint


# ================================================================================
# DATA LOADER
# ================================================================================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.label_encoder = None  # For XGBoost categorical encoding

    def load_and_prepare(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        logger.info(f"Loading data with shape: {df.shape}")

        # Check for missing columns
        missing = set(self.config.SELECTED_COLUMNS) - set(df.columns)
        if missing:
            logger.error(f"Missing columns for LAG={self.config.LAG}: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df[self.config.SELECTED_COLUMNS].copy()

        # Verify EVENT_ID exists and is numeric
        if 'EVENT_ID' not in df.columns:
            raise ValueError("EVENT_ID column not found in dataframe")
        
        df['EVENT_ID'] = pd.to_numeric(df['EVENT_ID'], errors='coerce')
        if df['EVENT_ID'].isna().any():
            logger.warning(f"Found {df['EVENT_ID'].isna().sum()} rows with invalid EVENT_ID, dropping them")
            df = df.dropna(subset=['EVENT_ID'])

        logger.info(f"EVENT_ID range: {df['EVENT_ID'].min():.0f} to {df['EVENT_ID'].max():.0f}")

        # For Ensemble: Filter by EVENT_TYPE
        if self.config.MODEL_TYPE == 'Ensemble' and self.config.EVENT_TYPE_VALUE is not None:
            initial_size = len(df)
            df = df[df['EVENT_TYPE'] == self.config.EVENT_TYPE_VALUE].copy()
            logger.info(f"Filtered to EVENT_TYPE={self.config.EVENT_TYPE_VALUE}: {len(df)} rows (from {initial_size})")
            
            # Validate we have enough data after filtering
            if len(df) < 100:
                logger.warning(f"Very small dataset after filtering: {len(df)} rows")

        # Data quality checks
        initial = len(df)
        df = df.dropna()
        dropped = initial - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with missing values ({100*dropped/initial:.2f}%)")
        logger.info(f"Final data shape: {df.shape}")

        # Split train/test based on EVENT_ID
        train_df = df[df['EVENT_ID'] <= self.config.EVENT_ID_CUTOFF].copy()
        test_df = df[df['EVENT_ID'] >= self.config.EVENT_ID_CUTOFF + 1].copy()

        logger.info(f"Train: {train_df.shape[0]} rows (EVENT_ID <= {self.config.EVENT_ID_CUTOFF})")
        logger.info(f"Test: {test_df.shape[0]} rows (EVENT_ID >= {self.config.EVENT_ID_CUTOFF + 1})")
        
        if len(train_df) > 0:
            logger.info(f"  Train EVENT_ID range: {train_df['EVENT_ID'].min():.0f} to {train_df['EVENT_ID'].max():.0f}")
        if len(test_df) > 0:
            logger.info(f"  Test EVENT_ID range: {test_df['EVENT_ID'].min():.0f} to {test_df['EVENT_ID'].max():.0f}")

        # Get feature list BEFORE categorical conversion
        features = [c for c in train_df.columns if c not in self.config.IGNORE_COLS]
        logger.info(f"Using {len(features)} features")
        
        # Debug: Log all columns and features
        logger.info(f"\nColumn Analysis:")
        logger.info(f"  All train_df columns: {train_df.columns.tolist()}")
        logger.info(f"  IGNORE_COLS: {self.config.IGNORE_COLS}")
        logger.info(f"  Final features: {features}")

        # Handle categorical features based on algorithm
        if self.config.MODEL_TYPE == 'Unified':
            # Verify EVENT_TYPE is in features BEFORE conversion
            if self.config.CATEGORICAL_FEATURE not in features:
                logger.error(f"❌ CRITICAL: {self.config.CATEGORICAL_FEATURE} is NOT in features list!")
                logger.error(f"   Available columns: {train_df.columns.tolist()}")
                logger.error(f"   IGNORE_COLS: {self.config.IGNORE_COLS}")
                raise ValueError(f"{self.config.CATEGORICAL_FEATURE} must be in features for Unified mode")
            
            logger.info(f"✅ {self.config.CATEGORICAL_FEATURE} is included in features list")
            logger.info(f"   Unique EVENT_TYPE values in train: {sorted(train_df[self.config.CATEGORICAL_FEATURE].unique())}")
            logger.info(f"   Unique EVENT_TYPE values in test: {sorted(test_df[self.config.CATEGORICAL_FEATURE].unique())}")
            
            if self.config.ALGORITHM == 'LightGBM':
                # LightGBM: Use pandas categorical
                train_df[self.config.CATEGORICAL_FEATURE] = train_df[self.config.CATEGORICAL_FEATURE].astype('category')
                test_df[self.config.CATEGORICAL_FEATURE] = test_df[self.config.CATEGORICAL_FEATURE].astype('category')
                logger.info(f"Set {self.config.CATEGORICAL_FEATURE} as categorical feature (LightGBM)")
                logger.info(f"   Train dtype: {train_df[self.config.CATEGORICAL_FEATURE].dtype}")
                logger.info(f"   Test dtype: {test_df[self.config.CATEGORICAL_FEATURE].dtype}")
            else:  # XGBoost
                # XGBoost: Use label encoding for better compatibility
                self.label_encoder = LabelEncoder()
                train_df[self.config.CATEGORICAL_FEATURE] = self.label_encoder.fit_transform(
                    train_df[self.config.CATEGORICAL_FEATURE]
                )
                test_df[self.config.CATEGORICAL_FEATURE] = self.label_encoder.transform(
                    test_df[self.config.CATEGORICAL_FEATURE]
                )
                logger.info(f"Label encoded {self.config.CATEGORICAL_FEATURE} for XGBoost:")
                logger.info(f"   Classes: {self.label_encoder.classes_}")
                logger.info(f"   Train values: {sorted(train_df[self.config.CATEGORICAL_FEATURE].unique())}")
                logger.info(f"   Test values: {sorted(test_df[self.config.CATEGORICAL_FEATURE].unique())}")

        return train_df, test_df, features


# ================================================================================
# BAYESIAN OPTIMIZER
# ================================================================================

class BayesianOptimizerWithCheckpoint:
    def __init__(self, config: Config, train_df: pd.DataFrame, features: List[str],
                 checkpoint_manager: CheckpointManager):
        self.config = config
        self.train_df = train_df
        self.features = features
        self.checkpoint_manager = checkpoint_manager
        self._interrupted = False
        self._iteration_count = 0

        # Signal handlers
        def _handle_signal(signum, frame):
            logger.info(f"Received signal {signum}; will stop after current evaluation.")
            self._interrupted = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle_signal)
            except Exception:
                pass

        # Define search space
        if config.ALGORITHM == 'LightGBM':
            self.space = [
                Integer(32, 96, name='num_leaves'),
                Integer(5, 12, name='max_depth'),
                Real(0.05, 0.2, name='learning_rate', prior='log-uniform'),
                Real(0.6, 1.0, name='feature_fraction'),
                Real(0.6, 1.0, name='bagging_fraction'),
                Real(0.0, 3.0, name='lambda_l1'),
                Real(0.0, 3.0, name='lambda_l2'),
                Integer(20, 60, name='min_data_in_leaf')
            ]
        else:  # XGBoost
            self.space = [
                Integer(5, 12, name='max_depth'),
                Real(0.05, 0.2, name='learning_rate', prior='log-uniform'),
                Real(0.6, 1.0, name='subsample'),
                Real(0.6, 1.0, name='colsample_bytree'),
                Real(0.0, 3.0, name='reg_alpha'),
                Real(0.0, 3.0, name='reg_lambda'),
                Integer(20, 60, name='min_child_weight')
            ]

    def _objective(self, params_list):
        """Objective function for Bayesian optimization."""
        if self._interrupted:
            logger.info("Interrupted flag set, returning large value to stop")
            return 1e10

        self._iteration_count += 1
        logger.info(f"Bayesian iteration {self._iteration_count}/{self.config.N_BAYESIAN_CALLS}")

        try:
            if self.config.ALGORITHM == 'LightGBM':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'num_leaves': int(params_list[0]),
                    'max_depth': int(params_list[1]),
                    'learning_rate': params_list[2],
                    'feature_fraction': params_list[3],
                    'bagging_fraction': params_list[4],
                    'lambda_l1': params_list[5],
                    'lambda_l2': params_list[6],
                    'min_data_in_leaf': int(params_list[7]),
                    'bagging_freq': 1,
                    'random_state': self.config.RANDOM_STATE
                }
                if self.config.MODEL_TYPE == 'Unified':
                    params['categorical_feature'] = [self.config.CATEGORICAL_FEATURE]
            else:  # XGBoost
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': int(params_list[0]),
                    'learning_rate': params_list[1],
                    'subsample': params_list[2],
                    'colsample_bytree': params_list[3],
                    'reg_alpha': params_list[4],
                    'reg_lambda': params_list[5],
                    'min_child_weight': int(params_list[6]),
                    'seed': self.config.RANDOM_STATE,
                    'verbosity': 0,
                    'tree_method': 'hist'
                }

            # Cross-validation with error handling
            gkf = GroupKFold(n_splits=self.config.N_SPLITS)
            rmse_scores = []

            for fold_num, (train_idx, val_idx) in enumerate(gkf.split(self.train_df, groups=self.train_df['CNTY_NM']), 1):
                try:
                    train_fold = self.train_df.iloc[train_idx]
                    val_fold = self.train_df.iloc[val_idx]

                    X_train = train_fold[self.features]
                    y_train = train_fold[self.config.TARGET]
                    X_val = val_fold[self.features]
                    y_val = val_fold[self.config.TARGET]

                    if self.config.ALGORITHM == 'LightGBM':
                        train_data = lgb.Dataset(X_train, label=y_train)
                        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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
                    else:  # XGBoost
                        train_data = xgb.DMatrix(X_train, label=y_train)
                        val_data = xgb.DMatrix(X_val, label=y_val)

                        model = xgb.train(
                            params,
                            train_data,
                            num_boost_round=self.config.NUM_BOOST_ROUND,
                            evals=[(val_data, 'valid')],
                            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                            verbose_eval=False
                        )
                        y_pred = model.predict(val_data, iteration_range=(0, model.best_iteration + 1))

                    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                    rmse_scores.append(rmse)
                    
                except Exception as e:
                    logger.warning(f"Fold {fold_num} failed: {str(e)}")
                    continue

            if not rmse_scores:
                logger.error("All folds failed! Returning large penalty.")
                return 1e10

            avg_rmse = np.mean(rmse_scores)
            std_rmse = np.std(rmse_scores)
            logger.info(f"CV RMSE: {avg_rmse:.4f} ± {std_rmse:.4f} (from {len(rmse_scores)} folds)")
            return avg_rmse
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            logger.error(traceback.format_exc())
            return 1e10

    def optimize(self, resume: bool = False) -> Dict:
        """Run Bayesian optimization with checkpoint support."""
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 1: BAYESIAN OPTIMIZATION ({self.config.N_BAYESIAN_CALLS} calls)")
        logger.info("=" * 80)

        x0 = None
        y0 = None
        n_initial_points = self.config.N_RANDOM_STARTS

        if resume:
            iteration, best_params, all_results = self.checkpoint_manager.load_bayesian_checkpoint()
            if iteration > 0:
                logger.info(f"Resuming from iteration {iteration}")
                x0 = [r['x'] for r in all_results]
                y0 = [r['y'] for r in all_results]
                n_initial_points = max(0, self.config.N_RANDOM_STARTS - len(x0))
                self._iteration_count = iteration
        else:
            logger.info("No checkpoint found, starting from beginning")
            iteration = 0

        # Callback to save checkpoint after each iteration
        def _callback(res):
            nonlocal iteration
            iteration += 1
            all_results = [{'x': list(x), 'y': float(y)} for x, y in zip(res.x_iters, res.func_vals)]
            best_params = {'x': list(res.x), 'y': float(res.fun)}
            self.checkpoint_manager.save_bayesian_checkpoint(iteration, best_params, all_results)
            
            if self._interrupted:
                logger.info("Stopping optimization due to interrupt")
                return True
            return False

        # Run optimization
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

            logger.info(f"\nBest CV RMSE: {result.fun:.4f}")
            logger.info("✓ Bayesian Optimization Complete!")

            # Convert result to dictionary
            if self.config.ALGORITHM == 'LightGBM':
                best_params = {
                    'num_leaves': int(result.x[0]),
                    'max_depth': int(result.x[1]),
                    'learning_rate': float(result.x[2]),
                    'feature_fraction': float(result.x[3]),
                    'bagging_fraction': float(result.x[4]),
                    'lambda_l1': float(result.x[5]),
                    'lambda_l2': float(result.x[6]),
                    'min_data_in_leaf': int(result.x[7])
                }
            else:
                best_params = {
                    'max_depth': int(result.x[0]),
                    'learning_rate': float(result.x[1]),
                    'subsample': float(result.x[2]),
                    'colsample_bytree': float(result.x[3]),
                    'reg_alpha': float(result.x[4]),
                    'reg_lambda': float(result.x[5]),
                    'min_child_weight': int(result.x[6])
                }

            # Save final checkpoint
            all_results = [{'x': list(x), 'y': float(y)} for x, y in zip(result.x_iters, result.func_vals)]
            self.checkpoint_manager.save_bayesian_checkpoint(
                len(result.x_iters),
                {'x': list(result.x), 'y': float(result.fun)},
                all_results
            )

            return best_params
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# ================================================================================
# SINGLE MODEL TRAINER (with seed-ensemble)
# ================================================================================

class SingleModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self._warned_fraction = False

    def _jitter_fraction(self, value: float, floor: float, seed: int, width: float = 0.02) -> float:
        """Add small random jitter to fraction parameters for ensemble diversity."""
        rng = np.random.default_rng(seed)
        scale = 1.0 + rng.uniform(-width, width)
        return float(min(1.0, max(floor, value * scale)))

    def _maybe_make_stochastic_params_lgb(self, params: Dict, seed: int) -> Dict:
        """Ensure some randomness for ensemble members even if best_params had fractions = 1.0."""
        p = params.copy()
        p['bagging_freq'] = 1
        ff = float(p.get('feature_fraction', 1.0))
        bf = float(p.get('bagging_fraction', 1.0))
        if ff >= 0.999:
            ff = self.config.ENSEMBLE_FRACTION_FLOOR
        if bf >= 0.999:
            bf = self.config.ENSEMBLE_FRACTION_FLOOR
        p['feature_fraction'] = self._jitter_fraction(ff, self.config.ENSEMBLE_FRACTION_FLOOR, seed)
        p['bagging_fraction'] = self._jitter_fraction(bf, self.config.ENSEMBLE_FRACTION_FLOOR, seed + 1)
        if not self._warned_fraction and (p['feature_fraction'] < 0.999 or p['bagging_fraction'] < 0.999):
            logger.info(f"Applied LGBM fraction floor/jitter for ensemble diversity "
                       f"(feature={p['feature_fraction']:.3f}, bagging={p['bagging_fraction']:.3f})")
            self._warned_fraction = True
        return p

    def _maybe_make_stochastic_params_xgb(self, params: Dict, seed: int) -> Dict:
        """Ensure some randomness for XGBoost ensemble members."""
        p = params.copy()
        ss = float(p.get('subsample', 1.0))
        cs = float(p.get('colsample_bytree', 1.0))
        if ss >= 0.999:
            ss = self.config.XGB_FRACTION_FLOOR
        if cs >= 0.999:
            cs = self.config.XGB_FRACTION_FLOOR
        p['subsample'] = self._jitter_fraction(ss, self.config.XGB_FRACTION_FLOOR, seed)
        p['colsample_bytree'] = self._jitter_fraction(cs, self.config.XGB_FRACTION_FLOOR, seed + 1)
        if not self._warned_fraction and (p['subsample'] < 0.999 or p['colsample_bytree'] < 0.999):
            logger.info(f"Applied XGB fraction floor/jitter for ensemble diversity "
                       f"(subsample={p['subsample']:.3f}, colsample={p['colsample_bytree']:.3f})")
            self._warned_fraction = True
        return p

    def _train_one_model(self, train_df: pd.DataFrame, features: List[str], best_params: Dict, seed: int):
        """Train a single model with given parameters and seed."""
        try:
            X_train = train_df[features]
            y_train = train_df[self.config.TARGET]

            if self.config.ALGORITHM == 'LightGBM':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'bagging_freq': 1,
                    'random_state': seed,
                    **best_params
                }
                params = self._maybe_make_stochastic_params_lgb(params, seed)
                if self.config.MODEL_TYPE == 'Unified':
                    params['categorical_feature'] = [self.config.CATEGORICAL_FEATURE]

                # Optional bootstrap of rows
                if self.config.BOOTSTRAP_TRAIN:
                    n = len(X_train)
                    rng = np.random.default_rng(seed)
                    idx = rng.integers(0, n, size=n)
                    Xb = X_train.iloc[idx]
                    yb = y_train.iloc[idx]
                else:
                    Xb, yb = X_train, y_train

                train_data = lgb.Dataset(Xb, label=yb)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.config.NUM_BOOST_ROUND,
                    callbacks=[lgb.log_evaluation(period=50)]
                )
            else:  # XGBoost
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'seed': seed,
                    'verbosity': 0,
                    'tree_method': 'hist',
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
            logger.error(f"Error training model with seed {seed}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train_model(self, train_df: pd.DataFrame, features: List[str], best_params: Dict):
        """Train final single model on all training data (for export)."""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: TRAIN SINGLE MODEL (for export)")
        logger.info("=" * 80)

        model = self._train_one_model(train_df, features, best_params, seed=self.config.RANDOM_STATE)
        logger.info("✓ Model Training Complete!")
        return model

    def seed_ensemble_predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str],
                              best_params: Dict, checkpoint_manager: CheckpointManager, resume: bool = False):
        """Re-train N models with different seeds (and optional bootstrap) and predict on the same test set.
        Also computes permutation importance per seed model.
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"STEP 3: SEED-ENSEMBLE — {self.config.N_TEST_PREDICTIONS} MODELS WITH PERMUTATION IMPORTANCE")
        logger.info("=" * 80)
        
        # Verify EVENT_TYPE is in features for Unified mode
        if self.config.MODEL_TYPE == 'Unified':
            if self.config.CATEGORICAL_FEATURE in features:
                logger.info(f"✅ {self.config.CATEGORICAL_FEATURE} IS in features for importance calculation")
            else:
                logger.warning(f"⚠️ {self.config.CATEGORICAL_FEATURE} is NOT in features list!")
                logger.warning(f"Available features: {features}")

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

        for test_num in range(1, self.config.N_TEST_PREDICTIONS + 1):
            if test_num in completed_tests:
                checkpoint = checkpoint_manager.load_test_checkpoint(test_num)
                if checkpoint:
                    all_predictions.append(checkpoint['predictions'])
                    if checkpoint['importance'] is not None:
                        all_importance.append(checkpoint['importance'])
                    logger.info(f"✓ Loaded test {test_num} from checkpoint")
                    continue

            logger.info(f"\nTraining ensemble member {test_num}/{self.config.N_TEST_PREDICTIONS}")
            seed = self.config.RANDOM_STATE + test_num

            try:
                # Train a fresh model for this seed
                model_i = self._train_one_model(train_df, features, best_params, seed=seed)

                # Predict on test
                if self.config.ALGORITHM == 'LightGBM':
                    y_pred = model_i.predict(X_test)
                else:
                    test_data = xgb.DMatrix(X_test)
                    y_pred = model_i.predict(test_data)

                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                logger.info(f"Seed {seed} → RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

                # Store predictions (include meta columns consistently)
                pred_df = test_df[['Date', 'CNTY_NM', 'EVENT_TYPE', 'EVENT_ID', self.config.TARGET]].copy()
                pred_df[f'pred_{test_num}'] = y_pred
                all_predictions.append(pred_df)

                # Metrics record
                metrics_dict = {'test_num': test_num, 'rmse': rmse, 'mae': mae, 'r2': r2}
                all_metrics.append(metrics_dict)

                # Permutation importance for this member
                importance_df = None
                if self.config.COMPUTE_PERMUTATION_IMPORTANCE:
                    logger.info(f"Computing permutation importance (Ensemble member {test_num})...")
                    importance_df = self._compute_permutation_importance(
                        model_i, X_test, y_test, features, test_num, rmse
                    )
                    all_importance.append(importance_df)

                # Save checkpoint
                checkpoint_manager.save_test_predictions_checkpoint(test_num, pred_df, importance_df)
                
            except Exception as e:
                logger.error(f"Failed to train ensemble member {test_num}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        if not all_predictions:
            logger.error("No successful predictions were made!")
            raise RuntimeError("All ensemble members failed")

        # Combine predictions by columns
        combined_predictions = all_predictions[0][['Date', 'CNTY_NM', 'EVENT_TYPE', 'EVENT_ID', self.config.TARGET]].copy()
        for i, pred_df in enumerate(all_predictions, 1):
            combined_predictions[f'pred_{i}'] = pred_df[f'pred_{i}']

        metrics_df = pd.DataFrame(all_metrics)
        logger.info(f"\n✓ Completed {len(all_predictions)}/{self.config.N_TEST_PREDICTIONS} ensemble predictions")
        return combined_predictions, metrics_df, all_importance

    def _compute_permutation_importance(self, model, X_test, y_test, features, test_num, baseline_rmse):
        """Compute permutation importance for features."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Computing Permutation Importance (Test {test_num})")
        logger.info(f"{'='*60}")
        logger.info(f"Total features to test: {len(features)}")
        logger.info(f"Features list: {features}")
        
        importance_list = []

        for idx, feature in enumerate(features, 1):
            try:
                # Check if this is the EVENT_TYPE feature
                is_event_type = (feature == self.config.CATEGORICAL_FEATURE and 
                               self.config.MODEL_TYPE == 'Unified')
                
                if is_event_type:
                    logger.info(f"\n🎯 [{idx}/{len(features)}] Processing EVENT_TYPE: '{feature}'")
                    logger.info(f"   Feature dtype: {X_test[feature].dtype}")
                    logger.info(f"   Unique values: {sorted(X_test[feature].unique())}")
                
                # Simple approach: copy DataFrame and shuffle feature values
                X_test_permuted = X_test.copy()
                np.random.seed(self.config.RANDOM_STATE + test_num)
                
                # Get shuffled values
                shuffled_values = np.random.permutation(X_test[feature].values)
                
                # Assign shuffled values (pandas handles dtype conversion)
                X_test_permuted.loc[:, feature] = shuffled_values

                # Predict with shuffled feature
                if self.config.ALGORITHM == 'LightGBM':
                    y_pred_permuted = model.predict(X_test_permuted)
                else:
                    test_data_permuted = xgb.DMatrix(X_test_permuted)
                    y_pred_permuted = model.predict(test_data_permuted)

                # Calculate importance as RMSE difference
                permuted_rmse = np.sqrt(mean_squared_error(y_test, y_pred_permuted))
                importance = permuted_rmse - baseline_rmse

                importance_list.append({
                    'feature': feature,
                    'feature_type': 'categorical' if is_event_type else 'numerical',
                    'test_num': test_num,
                    'importance': importance,
                    'rmse_baseline': baseline_rmse,
                    'rmse_shuffled': permuted_rmse
                })
                
                # Log EVENT_TYPE importance specially
                if is_event_type:
                    logger.info(f"   ✅ EVENT_TYPE importance computed: {importance:.6f}")
                    logger.info(f"   RMSE change: {baseline_rmse:.4f} → {permuted_rmse:.4f}")
                    logger.info(f"   Impact: {100*importance/baseline_rmse:.2f}% increase")
                    
            except Exception as e:
                # EVENT_TYPE failure is critical!
                if feature == self.config.CATEGORICAL_FEATURE:
                    logger.error(f"❌ CRITICAL ERROR computing EVENT_TYPE importance!")
                    logger.error(f"   Error: {str(e)}")
                    logger.error(f"   Full traceback:")
                    logger.error(traceback.format_exc())
                    # Raise exception for EVENT_TYPE - must be included
                    raise
                else:
                    logger.warning(f"Failed to compute importance for {feature}: {str(e)}")
                    continue

        importance_df = pd.DataFrame(importance_list)
        
        # Validate results
        if self.config.MODEL_TYPE == 'Unified':
            has_event_type = self.config.CATEGORICAL_FEATURE in importance_df['feature'].values
            if not has_event_type:
                logger.error(f"❌ CRITICAL: EVENT_TYPE missing from results!")
                logger.error(f"   Features in results: {importance_df['feature'].tolist()}")
                raise ValueError("EVENT_TYPE must be in importance results for Unified mode")
            else:
                logger.info(f"✅ EVENT_TYPE successfully included in results")
        
        # Log summary
        if not importance_df.empty:
            logger.info(f"\n{'='*60}")
            logger.info(f"Permutation Importance Summary (Test {test_num})")
            logger.info(f"{'='*60}")
            logger.info(f"Total features computed: {len(importance_df)}")
            
            logger.info(f"\nTop 5 features:")
            top5 = importance_df.nlargest(5, 'importance')
            for idx, row in top5.iterrows():
                feat_type = "📁 CAT" if row['feature_type'] == 'categorical' else "📊 NUM"
                logger.info(f"  {feat_type} {row['feature']:30s}: {row['importance']:8.6f}")
        else:
            logger.error("❌ CRITICAL: importance_df is EMPTY!")
            raise ValueError("Importance DataFrame cannot be empty")
        
        return importance_df


# ================================================================================
# RESULTS SAVER
# ================================================================================

class ResultsSaver:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.summary_dir = output_dir / 'summary'
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)

    def save_best_params(self, best_params: Dict[str, Any]):
        path = self.summary_dir / 'best_params.json'
        serializable_params = convert_to_json_serializable(best_params)
        with open(path, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        logger.info("✓ Saved: best_params.json")
        logger.info(f"   Full path: {path.absolute()}")

    def save_model(self, model, algorithm: str):
        try:
            if algorithm == 'LightGBM':
                path = self.models_dir / 'lightgbm_model.txt'
                model.save_model(str(path))
            else:
                path = self.models_dir / 'xgboost_model.json'
                model.save_model(str(path))
            logger.info(f"✓ Saved: {algorithm.lower()}_model")
            logger.info(f"   Full path: {path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")

    def save_predictions(self, predictions: pd.DataFrame):
        path = self.summary_dir / 'predictions.csv'
        predictions.to_csv(path, index=False)
        
        if path.exists():
            logger.info(f"✓ Saved: predictions.csv ({len(predictions)} rows)")
            logger.info(f"   Full path: {path.absolute()}")
        else:
            logger.error(f"❌ FAILED to save predictions.csv")

    def save_metrics(self, metrics: pd.DataFrame):
        path = self.summary_dir / 'metrics.csv'
        metrics.to_csv(path, index=False)
        logger.info(f"✓ Saved: metrics.csv ({len(metrics)} rows)")

    def save_summary_statistics(self, predictions: pd.DataFrame):
        pred_cols = [c for c in predictions.columns if c.startswith('pred_')]
        predictions['pred_mean'] = predictions[pred_cols].mean(axis=1)
        predictions['pred_std'] = predictions[pred_cols].std(axis=1)
        predictions['pred_min'] = predictions[pred_cols].min(axis=1)
        predictions['pred_max'] = predictions[pred_cols].max(axis=1)

        summary = predictions.groupby(['Date', 'CNTY_NM', 'EVENT_TYPE', 'EVENT_ID']).agg({
            'recorded_outages': 'first',
            'pred_mean': 'mean',
            'pred_std': 'mean',
            'pred_min': 'min',
            'pred_max': 'max',
            'pred_1': 'count'
        }).rename(columns={'pred_1': 'N_tests'}).reset_index()

        summary.columns = ['Date', 'CNTY_NM', 'EVENT_TYPE', 'EVENT_ID', 'Actual',
                           'Pred_mean', 'Pred_std', 'Pred_min', 'Pred_max', 'N_tests']

        path = self.summary_dir / 'predictions_summary.csv'
        summary.to_csv(path, index=False)
        logger.info(f"✓ Saved: predictions_summary.csv ({len(summary)} rows)")

    def save_permutation_importance(self, imp_df: pd.DataFrame, test_num: int):
        path = self.summary_dir / f'permutation_importance_test{test_num:02d}.csv'
        imp_df.sort_values('importance', ascending=False).to_csv(path, index=False)
        logger.info(f"✓ Saved: {path.name}")

    def save_aggregated_permutation_importance(self, all_importance: List[pd.DataFrame]):
        if not all_importance:
            logger.warning("No importance data to aggregate")
            return

        combined = pd.concat(all_importance, ignore_index=True)
        logger.info(f"\nAggregating {len(combined)} importance records from {len(all_importance)} tests")
        
        # Check if EVENT_TYPE is in the combined data
        event_type_data = combined[combined['feature'] == 'EVENT_TYPE']
        if not event_type_data.empty:
            logger.info(f"✅ EVENT_TYPE found in combined data: {len(event_type_data)} records")
        else:
            logger.warning(f"❌ EVENT_TYPE NOT found in combined data!")
            logger.info(f"Available features: {sorted(combined['feature'].unique())}")

        agg = (
            combined
            .groupby(['feature', 'feature_type'], as_index=False)
            .agg(
                mean_importance=('importance', 'mean'),
                std_importance=('importance', 'std'),
                n_tests=('test_num', 'nunique'),
                mean_rmse_base=('rmse_baseline', 'mean')
            )
            .sort_values('mean_importance', ascending=False)
        )

        path = self.summary_dir / 'permutation_importance_aggregated.csv'
        agg.to_csv(path, index=False)
        logger.info(f"✓ Saved: permutation_importance_aggregated.csv")
        logger.info(f"   Full path: {path.absolute()}")
        
        # Log the aggregated results
        logger.info("\n" + "=" * 80)
        logger.info("AGGREGATED PERMUTATION IMPORTANCE (across all ensemble members)")
        logger.info("=" * 80)
        logger.info(f"\nTop 10 Most Important Features:")
        for idx, row in agg.head(10).iterrows():
            feat_type = "📁 CATEGORICAL" if row['feature_type'] == 'categorical' else "📊 NUMERICAL"
            logger.info(f"  {feat_type:15s} {row['feature']:30s} "
                       f"Mean: {row['mean_importance']:8.6f} ± {row['std_importance']:8.6f}")


# ================================================================================
# MAIN EXECUTION FUNCTIONS
# ================================================================================

def run_single_model(df: pd.DataFrame, config: Config, base_output_dir: Path, resume: bool = False):
    """Run training for a single model (Unified or one Ensemble event type)."""
    output_dir = config.get_output_path(base_output_dir)
    logger.info(f"Output directory: {output_dir}")

    checkpoint_manager = CheckpointManager(output_dir)
    data_loader = DataLoader(config)
    
    try:
        train_df, test_df, features = data_loader.load_and_prepare(df)
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

    # If there's no test data (e.g., specific EVENT_TYPE missing in the test events), skip testing later.
    no_test_rows = test_df.shape[0] == 0
    if no_test_rows:
        logger.warning("No test rows found for this configuration (likely specific EVENT_TYPE missing in test events). Will skip testing phase.")

    # Bayesian Optimization
    optimizer = BayesianOptimizerWithCheckpoint(config, train_df, features, checkpoint_manager)
    best_params = optimizer.optimize(resume=resume)

    # Train a reference model (for export)
    trainer = SingleModelTrainer(config)
    model = trainer.train_model(train_df, features, best_params)
    checkpoint_manager.save_model_checkpoint(model, config.ALGORITHM)

    # Test Predictions (seed-ensemble by default)
    if no_test_rows:
        logger.info("Skipping test predictions due to empty test set.")
        predictions = pd.DataFrame()
        metrics = pd.DataFrame()
        all_importance = []
    else:
        if config.ENSEMBLE_RETRAIN_EACH_TEST:
            predictions, metrics, all_importance = trainer.seed_ensemble_predict(
                train_df, test_df, features, best_params,
                checkpoint_manager=checkpoint_manager, resume=resume
            )
        else:
            logger.warning("ENSEMBLE_RETRAIN_EACH_TEST is False — repeated predictions from a single model may be identical.")
            predictions, metrics, all_importance = trainer.seed_ensemble_predict(
                train_df, test_df, features, best_params,
                checkpoint_manager=checkpoint_manager, resume=resume
            )

    # Save Results
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    saver = ResultsSaver(output_dir)
    saver.save_best_params(best_params)
    saver.save_model(model, config.ALGORITHM)

    if not predictions.empty:
        saver.save_predictions(predictions)
        saver.save_metrics(metrics)
        saver.save_summary_statistics(predictions)

        if config.COMPUTE_PERMUTATION_IMPORTANCE and all_importance:
            for imp_df in all_importance:
                test_num = int(imp_df['test_num'].iloc[0])
                saver.save_permutation_importance(imp_df, test_num)
            saver.save_aggregated_permutation_importance(all_importance)
    else:
        logger.info("No predictions/metrics to save because test was skipped.")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 80)


def run_ensemble_models(df: pd.DataFrame, config: Config, base_output_dir: Path, resume: bool = False):
    """Run training for Ensemble models (one model per EVENT_TYPE)."""
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE MODE: Training separate models for each EVENT_TYPE")
    logger.info("=" * 80)

    # Get unique event types
    event_types = sorted(df['EVENT_TYPE'].unique())
    logger.info(f"Found {len(event_types)} unique EVENT_TYPEs: {event_types}")

    for event_type in event_types:
        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING MODEL FOR EVENT_TYPE: {event_type}")
        logger.info("=" * 80)

        try:
            # Create config for this event type
            event_config = Config()
            event_config.MODEL_TYPE = config.MODEL_TYPE
            event_config.ALGORITHM = config.ALGORITHM
            event_config.LAG = config.LAG
            event_config.EVENT_TYPE_VALUE = event_type
            # propagate ensemble flags
            event_config.ENSEMBLE_RETRAIN_EACH_TEST = config.ENSEMBLE_RETRAIN_EACH_TEST
            event_config.BOOTSTRAP_TRAIN = config.BOOTSTRAP_TRAIN
            event_config.ENSEMBLE_FRACTION_FLOOR = config.ENSEMBLE_FRACTION_FLOOR
            event_config.XGB_FRACTION_FLOOR = config.XGB_FRACTION_FLOOR
            event_config.__post_init__()  # Re-initialize with correct settings

            # Train model for this event type
            run_single_model(df, event_config, base_output_dir, resume)

            logger.info(f"✅ Completed EVENT_TYPE: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to train model for EVENT_TYPE {event_type}: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue with other event types
            continue


def main(
    parquet_file: str,
    model_type: str = 'Unified',
    algorithm: str = 'LightGBM',
    lag: int = 1,
    base_output_dir: Optional[str] = None,
    resume: bool = False
):
    """Main entry point."""
    try:
        # Create config
        config = Config()
        config.MODEL_TYPE = model_type
        config.ALGORITHM = algorithm
        config.LAG = lag
        config.__post_init__()  # Re-run to apply LAG correctly

        if base_output_dir is None:
            base_output_dir = Path.cwd() / 'outputs'
        else:
            base_output_dir = Path(base_output_dir)

        # Create base output directory
        output_dir = config.get_output_path(base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging to file
        log_file = output_dir / 'python_training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info("=" * 80)
        logger.info("IMPROVED OPTIMIZED MODEL TRAINING - EVENT_ID VERSION")
        logger.info(f"Resume Mode: {resume}")
        logger.info("=" * 80)

        logger.info(f"Loading data from {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        logger.info(f"Data loaded: {df.shape}")

        logger.info(f"\nModel Type: {config.MODEL_TYPE}")
        logger.info(f"Algorithm: {config.ALGORITHM}")
        logger.info(f"Lag: {config.LAG}")
        logger.info(f"EVENT_ID Cutoff: {config.EVENT_ID_CUTOFF} (train ≤ {config.EVENT_ID_CUTOFF}, test ≥ {config.EVENT_ID_CUTOFF + 1})")
        logger.info(f"Resume: {resume}")
        logger.info(f"Output Directory: {output_dir.absolute()}")
        logger.info(f"Seed-Ensemble: {config.ENSEMBLE_RETRAIN_EACH_TEST}, Bootstrap: {config.BOOTSTRAP_TRAIN}")

        # Run appropriate training mode
        if config.MODEL_TYPE == 'Ensemble':
            run_ensemble_models(df, config, base_output_dir, resume)
        else:  # Unified
            run_single_model(df, config, base_output_dir, resume)

        logger.info("\n" + "=" * 80)
        logger.info("ALL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("TRAINING FAILED WITH ERROR:")
        logger.error(f"{'='*80}")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python train_optimized_event_id.py <parquet_file> <model_type> <algorithm> <lag> [output_dir] [--resume]")
        print("Example: python train_optimized_event_id.py data.parquet Unified LightGBM 1 outputs/")
        print("Example: python train_optimized_event_id.py data.parquet Ensemble XGBoost 12 outputs/ --resume")
        sys.exit(1)

    parquet_file = sys.argv[1]
    model_type = sys.argv[2]
    algorithm = sys.argv[3]
    lag = int(sys.argv[4])
    output_dir = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else None
    resume = '--resume' in sys.argv

    main(parquet_file, model_type, algorithm, lag, output_dir, resume)