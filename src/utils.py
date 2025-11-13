"""
Utility Functions
Helper functions used across the project
"""

import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
import hashlib


def setup_logging(name, level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path):
    """Save configuration to YAML file"""
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_json(json_path):
    """Load JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_path):
    """Save data to JSON file"""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def get_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def ensure_dir(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent


def format_metric(metric_value, metric_type="float"):
    """Format metric for display"""
    if metric_type == "float":
        return f"{metric_value:.4f}"
    elif metric_type == "percent":
        return f"{metric_value:.2f}%"
    elif metric_type == "int":
        return f"{metric_value:,}"
    else:
        return str(metric_value)


class ModelMetrics:
    """Container for model metrics"""

    def __init__(self, mae=None, rmse=None, r2=None, mape=None):
        self.mae = mae
        self.rmse = rmse
        self.r2 = r2
        self.mape = mape

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "r2_score": self.r2,
            "mape": self.mape,
        }

    def __repr__(self):
        return f"ModelMetrics(\
                MAE={self.mae:.4f}, \
                RMSE={self.rmse:.4f}, \
                R²={self.r2:.4f} \
            )"


def print_section_header(title, char="=", width=60):
    """Print formatted section header"""
    print()
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_metrics_table(metrics_dict):
    """Print metrics in a formatted table"""
    print("\nMetrics:")
    print("-" * 40)
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:10.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print("-" * 40)
