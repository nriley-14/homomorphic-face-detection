"""Configuration loader for face recognition system.

Loads settings from config.yaml and provides them as a simple namespace.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file.

    Returns:
        Dict of configuration values.

    Raises:
        FileNotFoundError: If config.yaml is missing.
    """
    # Try current directory first
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Try package directory (for installed package)
    pkg_config = Path(__file__).parent.parent.parent / "config.yaml"
    if pkg_config.exists():
        with open(pkg_config) as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(
        f"Config file not found. Looked in:\n"
        f"  - {config_path.absolute()}\n"
        f"  - {pkg_config.absolute()}\n"
        f"Please ensure config.yaml is in your working directory."
    )


# Load config once at module import
_config = load_config()

# File paths
PUB_CTX_FILE = Path(_config["he_public_ctx_file"])
SEC_CTX_FILE = Path(_config["he_secret_ctx_file"])
PROJ_FILE = Path(_config["he_proj_file"])
SERVER_DB_FILE = Path(_config["server_db_file"])

# Dimensions
EMB_DIM = _config["emb_dim"]
PROJ_DIM = _config["proj_dim"]

# Recognition threshold
THRESHOLD = _config["threshold"]

# Server settings
SERVER_HOST = _config["server_host"]
SERVER_PORT = _config["server_port"]
