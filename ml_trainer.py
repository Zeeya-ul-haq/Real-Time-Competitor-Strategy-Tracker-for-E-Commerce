"""
Professional Machine Learning Price Prediction System
Configuration-driven ML pipeline with multiple algorithms including KNN
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import learning_curve

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config_manager import BaseComponent, PathConfig

@dataclass
class ModelMetrics:
    """Data structure for model performance metrics"""
    model_name: str
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_mape: float
    test_mape: float
    cv_mean_r2: float
    cv_std_r2: float
    training_time: float

@dataclass
class ModelResult:
    """Data structure for complete model results"""
    model: Any
    metrics: ModelMetrics
    predictions_train: np.ndarray
    predictions_test: np.ndarray
    feature_importance: Optional[np.ndarray] = None

class MLPricePredictor(BaseComponent):
    """
    Professional ML price prediction system with configurable algorithms
    """

    def __init__(self):
        super().__init__("ml_trainer")
        self.ml_config = self.get_component_config('machine_learning')
        self.training_config = self.ml_config.get('training', {})
        self.feature_config = self.ml_config.get('feature_engineering', {})
        self.algorithm_configs = self.ml_config.get('algorithms', {})

        # Initialize components
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, ModelResult] = {}
        self.preprocessing_pipeline: Optional[Pipeline] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []

        # Data containers
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def _get_scaler(self, method: str) -> Any:
        """Get scaler based on configuration"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(method, StandardScaler())

    def _get_text_vectorizer(self, config: Dict[str, Any]) -> Any:
        """Get text vectorizer based on configuration"""
        method = config.get('method', 'tfidf')

        common_params = {
            'max_features': config.get('max_features', 100),
            'min_df': config.get('min_df', 2),
            'max_df': config.get('max_df', 0.95),
            'ngram_range': tuple(config.get('ngram_range', [1, 2])),
            'stop_words': 'english'
        }

        if method == 'tfidf':
            return TfidfVectorizer(**common_params)
        elif method == 'count':
            return CountVectorizer(**common_params)
        else:
            return TfidfVectorizer(**common_params)

    def load_and_validate_data(self) -> bool:
        """Load and validate training data"""
        try:
            # Load processed mobile data
            mobile_file = self.paths.processed_mobile_file

            if not mobile_file.exists():
                self.logger.error(f"Processed mobile data not found: {mobile_file}")
                return False

            self.df = pd.read_csv(mobile_file)
            self.logger.info(f"Loaded {len(self.df)} mobile records")

            # Validate required columns
            required_cols = self.feature_config.get('categorical_features', []) + \
                          self.feature_config.get('numerical_features', []) + \
                          self.feature_config.get('text_features', []) + \
                          [self.feature_config.get('target_feature')]

            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False

            # Basic data validation
            if self.df.empty:
                self.logger.error("Dataset is empty")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False

    def preprocess_data(self) -> bool:
        """Comprehensive data preprocessing pipeline"""
        try:
            df = self.df.copy()
            target_col = self.feature_config['target_feature']

            self.logger.info("Starting data preprocessing...")

            # Clean target variable
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df = df.dropna(subset=[target_col])

            # Feature engineering
            categorical_features = self.feature_config.get('categorical_features', [])
            numerical_features = self.feature_config.get('numerical_features', [])
            text_features = self.feature_config.get('text_features', [])

            # Extract brand if not present
            if 'brand' in categorical_features and 'brand' not in df.columns:
                text_col = text_features[0] if text_features else 'mobilename'
                if text_col in df.columns:
                    df['brand'] = df[text_col].str.extract(r'^(\w+)')[0].fillna('Unknown')

            # Clean numerical features
            for col in numerical_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Handle categorical features
            feature_matrix_parts = []

            # Encode categorical features
            for col in categorical_features:
                if col in df.columns:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    feature_matrix_parts.append(f'{col}_encoded')

            # Add numerical features
            available_numerical = [col for col in numerical_features if col in df.columns]
            feature_matrix_parts.extend(available_numerical)

            # Process text features
            text_vectorizer_config = self.feature_config.get('text_vectorization', {})

            if text_features and text_features[0] in df.columns:
                self.text_vectorizer = self._get_text_vectorizer(text_vectorizer_config)
                text_data = df[text_features[0]].fillna('').astype(str)

                # Fit and transform text data
                text_features_matrix = self.text_vectorizer.fit_transform(text_data)
                text_feature_names = [f'text_{i}' for i in range(text_features_matrix.shape[1])]

                # Convert to DataFrame
                text_df = pd.DataFrame(
                    text_features_matrix.toarray(),
                    columns=text_feature_names,
                    index=df.index
                )

                # Combine with other features
                basic_features_df = df[feature_matrix_parts].reset_index(drop=True)
                text_df_reset = text_df.reset_index(drop=True)
                X = pd.concat([basic_features_df, text_df_reset], axis=1)

                self.feature_names = feature_matrix_parts + text_feature_names
            else:
                X = df[feature_matrix_parts]
                self.feature_names = feature_matrix_parts

            y = df[target_col].values

            # Remove outliers if configured
            outlier_config = self.ml_config.get('data_processing', {}).get('cleaning', {}).get('outlier_removal', {})
            if outlier_config.get('enabled', True):
                method = outlier_config.get('method', 'iqr')
                threshold = outlier_config.get('threshold', 1.5)

                if method == 'iqr':
                    Q1 = np.percentile(y, 25)
                    Q3 = np.percentile(y, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    mask = (y >= lower_bound) & (y <= upper_bound)
                    X = X[mask]
                    y = y[mask]

                    self.logger.info(f"Removed {(~mask).sum()} outliers using IQR method")

            # Feature scaling
            scaling_config = self.feature_config.get('scaling', {})
            if scaling_config.get('method'):
                numerical_indices = [i for i, col in enumerate(X.columns) 
                                   if col in available_numerical]

                if numerical_indices:
                    scaler = self._get_scaler(scaling_config['method'])
                    X_scaled = X.copy()
                    X_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X.iloc[:, numerical_indices])
                    X = X_scaled
                    self.scaler = scaler

            # Split data
            test_size = self.training_config.get('test_size', 0.2)
            random_state = self.training_config.get('random_state', 42)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            self.logger.info(f"Data preprocessing completed:")
            self.logger.info(f"  Features: {X.shape[1]}")
            self.logger.info(f"  Train samples: {self.X_train.shape[0]}")
            self.logger.info(f"  Test samples: {self.X_test.shape[0]}")

            return True

        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {e}")
            return False

    def _get_model_instance(self, algorithm_name: str, hyperparameters: Dict[str, Any]) -> Any:
        """Get model instance based on algorithm name and hyperparameters"""
        models_map = {
            'linear_regression': LinearRegression,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'knn': KNeighborsRegressor,
            'xgboost': XGBRegressor if XGBOOST_AVAILABLE else None
        }

        model_class = models_map.get(algorithm_name)

        if model_class is None:
            raise ValueError(f"Unknown or unavailable algorithm: {algorithm_name}")

        return model_class(**hyperparameters)

    def train_models(self) -> bool:
        """Train all configured ML models"""
        try:
            cv_folds = self.training_config.get('cross_validation_folds', 5)

            for algorithm_name, config in self.algorithm_configs.items():
                if not config.get('enabled', False):
                    continue

                display_name = algorithm_name.replace('_', ' ').title()
                self.logger.info(f"Training {display_name}...")

                start_time = datetime.now()

                try:
                    # Get model instance
                    hyperparameters = config.get('hyperparameters', {})
                    model = self._get_model_instance(algorithm_name, hyperparameters)

                    # Train model
                    model.fit(self.X_train, self.y_train)

                    # Make predictions
                    y_pred_train = model.predict(self.X_train)
                    y_pred_test = model.predict(self.X_test)

                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, self.X_train, self.y_train,
                        cv=cv_folds, scoring='r2'
                    )

                    training_time = (datetime.now() - start_time).total_seconds()

                    # Calculate metrics
                    metrics = ModelMetrics(
                        model_name=display_name,
                        train_r2=r2_score(self.y_train, y_pred_train),
                        test_r2=r2_score(self.y_test, y_pred_test),
                        train_rmse=np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                        test_rmse=np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                        train_mae=mean_absolute_error(self.y_train, y_pred_train),
                        test_mae=mean_absolute_error(self.y_test, y_pred_test),
                        train_mape=mean_absolute_percentage_error(self.y_train, y_pred_train),
                        test_mape=mean_absolute_percentage_error(self.y_test, y_pred_test),
                        cv_mean_r2=cv_scores.mean(),
                        cv_std_r2=cv_scores.std(),
                        training_time=training_time
                    )

                    # Feature importance
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        feature_importance = np.abs(model.coef_)

                    # Store results
                    result = ModelResult(
                        model=model,
                        metrics=metrics,
                        predictions_train=y_pred_train,
                        predictions_test=y_pred_test,
                        feature_importance=feature_importance
                    )

                    self.models[algorithm_name] = model
                    self.results[display_name] = result

                    # Save individual model
                    model_file = self.paths.models_dir / f"{algorithm_name}_model.pkl"
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)

                    self.logger.info(f"✅ {display_name} completed:")
                    self.logger.info(f"   R² Score: {metrics.test_r2:.4f}")
                    self.logger.info(f"   RMSE: ₹{metrics.test_rmse:,.0f}")
                    self.logger.info(f"   Training Time: {training_time:.2f}s")

                except Exception as e:
                    self.logger.error(f"Failed to train {display_name}: {e}")
                    continue

            return len(self.results) > 0

        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            return False

    def create_visualizations(self) -> None:
        """Create comprehensive visualizations"""
        try:
            if not self.results:
                self.logger.warning("No results available for visualization")
                return

            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

            # 1. Model comparison
            self._create_model_comparison_plot()

            # 2. Actual vs Predicted plots
            self._create_prediction_plots()

            # 3. Feature importance plots
            self._create_feature_importance_plots()

            # 4. Learning curves (for select models)
            self._create_learning_curves()

            self.logger.info(f"✅ Visualizations saved to {self.paths.plots_dir}")

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")

    def _create_model_comparison_plot(self) -> None:
        """Create model comparison visualization"""
        model_names = list(self.results.keys())
        metrics_data = {
            'R² Score': [self.results[name].metrics.test_r2 for name in model_names],
            'RMSE': [self.results[name].metrics.test_rmse for name in model_names],
            'MAE': [self.results[name].metrics.test_mae for name in model_names]
        }

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

        for idx, (metric, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            bars = ax.bar(model_names, values, color=colors)
            ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}' if metric == 'R² Score' else f'{value:.0f}',
                       ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.paths.plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_prediction_plots(self) -> None:
        """Create actual vs predicted plots"""
        n_models = len(self.results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()

        for idx, (model_name, result) in enumerate(self.results.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Scatter plot
            ax.scatter(self.y_test, result.predictions_test, alpha=0.6, s=20)

            # Perfect prediction line
            min_val = min(self.y_test.min(), result.predictions_test.min())
            max_val = max(self.y_test.max(), result.predictions_test.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('Actual Price (₹)')
            ax.set_ylabel('Predicted Price (₹)')
            ax.set_title(f'{model_name}\nR² = {result.metrics.test_r2:.3f}')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.paths.plots_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_feature_importance_plots(self) -> None:
        """Create feature importance plots for applicable models"""
        for model_name, result in self.results.items():
            if result.feature_importance is not None:
                # Get top 20 features
                indices = np.argsort(result.feature_importance)[::-1][:20]

                plt.figure(figsize=(12, 8))
                plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
                plt.barh(range(len(indices)), result.feature_importance[indices])
                plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
                plt.xlabel('Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()

                filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
                plt.savefig(self.paths.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()

    def _create_learning_curves(self) -> None:
        """Create learning curves for select models"""
        models_to_plot = ['Random Forest', 'Gradient Boosting']

        for model_name in models_to_plot:
            if model_name in self.results:
                model = self.results[model_name].model

                train_sizes, train_scores, val_scores = learning_curve(
                    model, self.X_train, self.y_train,
                    cv=3, train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='r2', n_jobs=-1
                )

                plt.figure(figsize=(10, 6))
                plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
                plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
                plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                               np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
                plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                               np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)

                plt.xlabel('Training Set Size')
                plt.ylabel('R² Score')
                plt.title(f'Learning Curve - {model_name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                filename = f'learning_curve_{model_name.lower().replace(" ", "_")}.png'
                plt.savefig(self.paths.plots_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()

    def save_results(self) -> None:
        """Save training results and preprocessing objects"""
        try:
            # Save model results
            results_data = []
            for model_name, result in self.results.items():
                metrics_dict = asdict(result.metrics)
                metrics_dict['timestamp'] = datetime.now().isoformat()
                results_data.append(metrics_dict)

            results_df = pd.DataFrame(results_data)
            results_df.to_csv(self.paths.model_results_file, index=False)

            # Save preprocessing objects
            preprocessing_objects = {
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'scaler': getattr(self, 'scaler', None),
                'text_vectorizer': getattr(self, 'text_vectorizer', None)
            }

            with open(self.paths.preprocessing_objects_file, 'wb') as f:
                pickle.dump(preprocessing_objects, f)

            self.logger.info(f"✅ Results saved to {self.paths.model_results_file}")
            self.logger.info(f"✅ Preprocessing objects saved to {self.paths.preprocessing_objects_file}")

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def run(self) -> bool:
        """Main training pipeline execution"""
        try:
            # Load and validate data
            if not self.load_and_validate_data():
                return False

            # Preprocess data
            if not self.preprocess_data():
                return False

            # Train models
            if not self.train_models():
                return False

            # Create visualizations
            self.create_visualizations()

            # Save results
            self.save_results()

            # Print summary
            self._print_training_summary()

            return True

        except Exception as e:
            self.logger.error(f"ML training pipeline failed: {e}")
            return False

    def _print_training_summary(self) -> None:
        """Print training summary"""
        print("\n" + "="*80)
        print("🎯 ML TRAINING SUMMARY")
        print("="*80)

        if not self.results:
            print("No models were trained successfully.")
            return

        # Sort by test R² score
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: x[1].metrics.test_r2, reverse=True)

        print(f"{'Model':<20} | {'R² Score':<10} | {'RMSE':<12} | {'MAE':<12} | {'Time (s)':<10}")
        print("-" * 80)

        for model_name, result in sorted_results:
            metrics = result.metrics
            print(f"{model_name:<20} | {metrics.test_r2:<10.3f} | "
                  f"₹{metrics.test_rmse:<11,.0f} | ₹{metrics.test_mae:<11,.0f} | "
                  f"{metrics.training_time:<10.2f}")

        # Best model
        best_model = sorted_results[0]
        print("="*80)
        print(f"🏆 Best Model: {best_model[0]} (R² = {best_model[1].metrics.test_r2:.3f})")
        print("="*80)

def run_training(ml_config: Dict[str, Any], paths: PathConfig) -> bool:
    """Entry point for running ML training"""
    try:
        trainer = MLPricePredictor()
        return trainer.run()
    except Exception as e:
        print(f"ML training failed: {e}")
        return False

if __name__ == "__main__":
    # For testing
    from config_manager import get_config_manager
    config_manager = get_config_manager()
    ml_config = config_manager.get('machine_learning')
    paths = config_manager.paths
    run_training(ml_config, paths)
