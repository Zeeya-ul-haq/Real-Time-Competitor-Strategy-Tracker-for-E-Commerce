"""
Professional Configuration Management System
Centralized configuration handling with path management and validation
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class PathConfig:
    """Data class for path configuration"""
    base_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    output_dir: Path
    logs_dir: Path
    models_dir: Path
    plots_dir: Path
    temp_dir: Path

    # File paths
    raw_mobile_file: Path
    raw_reviews_file: Path
    processed_mobile_file: Path
    processed_reviews_file: Path
    model_results_file: Path
    preprocessing_objects_file: Path
    main_log_file: Path
    scraper_log_file: Path
    ml_log_file: Path

class ConfigManager:
    """
    Professional configuration manager with validation and path resolution
    """

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._paths: Optional[PathConfig] = None
        self._logger = self._setup_logger()

        # Load and validate configuration
        self.load_config()
        self.setup_paths()

    def _setup_logger(self) -> logging.Logger:
        """Setup basic logger for configuration management"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def load_config(self) -> None:
        """Load and validate configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            self._validate_config()
            self._logger.info(f"Configuration loaded successfully from {self.config_path}")

        except yaml.YAMLError as e:
            self._logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Configuration loading failed: {e}")
            raise

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields"""
        required_sections = [
            'project', 'paths', 'logging', 'flows', 'scraping', 
            'data_processing', 'machine_learning', 'dashboard'
        ]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate paths section
        paths_config = self._config.get('paths', {})
        required_paths = [
            'base_dir', 'raw_data_dir', 'processed_data_dir', 
            'raw_mobile_file', 'raw_reviews_file'
        ]

        for path_key in required_paths:
            if path_key not in paths_config:
                raise ValueError(f"Missing required path configuration: {path_key}")

    def setup_paths(self) -> None:
        """Setup all path configurations with proper resolution"""
        paths_config = self._config['paths']
        base_dir = Path(paths_config['base_dir']).resolve()

        # Directory paths
        raw_data_dir = base_dir / paths_config['raw_data_dir']
        processed_data_dir = base_dir / paths_config['processed_data_dir']
        output_dir = base_dir / paths_config['output_dir']
        logs_dir = base_dir / paths_config['logs_dir']
        models_dir = base_dir / paths_config['models_dir']
        plots_dir = base_dir / paths_config['plots_dir']
        temp_dir = base_dir / paths_config['temp_dir']

        # File paths
        raw_mobile_file = raw_data_dir / paths_config['raw_mobile_file']
        raw_reviews_file = raw_data_dir / paths_config['raw_reviews_file']
        processed_mobile_file = processed_data_dir / paths_config['processed_mobile_file']
        processed_reviews_file = processed_data_dir / paths_config['processed_reviews_file']
        model_results_file = output_dir / paths_config['model_results_file']
        preprocessing_objects_file = models_dir / paths_config['preprocessing_objects_file']
        main_log_file = logs_dir / paths_config['main_log_file']
        scraper_log_file = logs_dir / paths_config['scraper_log_file']
        ml_log_file = logs_dir / paths_config['ml_log_file']

        # Create PathConfig instance
        self._paths = PathConfig(
            base_dir=base_dir,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            output_dir=output_dir,
            logs_dir=logs_dir,
            models_dir=models_dir,
            plots_dir=plots_dir,
            temp_dir=temp_dir,
            raw_mobile_file=raw_mobile_file,
            raw_reviews_file=raw_reviews_file,
            processed_mobile_file=processed_mobile_file,
            processed_reviews_file=processed_reviews_file,
            model_results_file=model_results_file,
            preprocessing_objects_file=preprocessing_objects_file,
            main_log_file=main_log_file,
            scraper_log_file=scraper_log_file,
            ml_log_file=ml_log_file
        )

        # Auto-create directories if configured
        if self._config.get('auto_create_dirs', True):
            self.create_directories()

    def create_directories(self) -> None:
        """Create all required directories"""
        directories = [
            self._paths.raw_data_dir,
            self._paths.processed_data_dir,
            self._paths.output_dir,
            self._paths.logs_dir,
            self._paths.models_dir,
            self._paths.plots_dir,
            self._paths.temp_dir
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self._logger.debug(f"Directory created/verified: {directory}")
            except Exception as e:
                self._logger.error(f"Failed to create directory {directory}: {e}")
                raise

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        if self._config is None:
            raise RuntimeError("Configuration not loaded")
        return self._config

    @property
    def paths(self) -> PathConfig:
        """Get the path configuration"""
        if self._paths is None:
            raise RuntimeError("Paths not configured")
        return self._paths

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'scraping.amazon.max_listing_pages')"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_flow_config(self, flow_name: str) -> Dict[str, Any]:
        """Get configuration for a specific flow"""
        flows = self._config.get('flows', {})
        if flow_name not in flows:
            raise ValueError(f"Flow configuration not found: {flow_name}")
        return flows[flow_name]

    def is_flow_enabled(self, flow_name: str) -> bool:
        """Check if a specific flow is enabled"""
        flow_config = self.get_flow_config(flow_name)
        return flow_config.get('enabled', False)

    def get_scraper_config(self, platform: str) -> Dict[str, Any]:
        """Get scraper configuration for a specific platform"""
        scraping_config = self._config.get('scraping', {})
        if platform not in scraping_config:
            raise ValueError(f"Scraper configuration not found: {platform}")
        return scraping_config[platform]

    def get_ml_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """Get ML algorithm configuration"""
        ml_config = self._config.get('machine_learning', {})
        algorithms = ml_config.get('algorithms', {})
        if algorithm not in algorithms:
            raise ValueError(f"ML algorithm configuration not found: {algorithm}")
        return algorithms[algorithm]

    def reload_config(self) -> None:
        """Reload configuration from file"""
        self.load_config()
        self.setup_paths()
        self._logger.info("Configuration reloaded successfully")

# Singleton instance for global access
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Union[str, Path] = "config.yaml") -> ConfigManager:
    """Get or create singleton configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager

def get_config() -> Dict[str, Any]:
    """Get configuration dictionary"""
    return get_config_manager().config

def get_paths() -> PathConfig:
    """Get path configuration"""
    return get_config_manager().paths

class BaseComponent(ABC):
    """
    Abstract base class for all system components
    Provides standardized configuration and logging
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.config_manager = get_config_manager()
        self.config = self.config_manager.config
        self.paths = self.config_manager.paths
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup component-specific logger"""
        logger = logging.getLogger(f"{__name__}.{self.component_name}")

        # Configure based on config
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        logger.setLevel(level)

        # Clear existing handlers
        logger.handlers.clear()

        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            formatter = logging.Formatter(log_config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_config.get('file_output', True):
            log_file = self.paths.logs_dir / f"{self.component_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            formatter = logging.Formatter(log_config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @abstractmethod
    def run(self) -> bool:
        """Abstract method that each component must implement"""
        pass

    def get_component_config(self, config_key: str) -> Dict[str, Any]:
        """Get configuration specific to this component"""
        return self.config.get(config_key, {})
