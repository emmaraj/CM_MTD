"""Configuration management for CM-MTD framework."""
from pathlib import Path
import yaml
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml.
        
    Returns:
        Dictionary with configuration parameters.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_config() -> Dict[str, Any]:
    """Get default configuration."""
    return load_config()
