"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Functions for getting the configuration settings.
"""

import json
import os
from pathlib import Path

CONFIG = None


def _read_config_file() -> dict:
    """Get the configuration data from the user and shared config files.

    Returns:
        dict: dictionary containing the configuration data

    """
    current_dir = Path(__file__).resolve().parent

    # Check if the environment is set for pytest
    if os.getenv("PYTEST_RUNNING") == "1":
        config_dir = current_dir.parent / "tests" / "test_data"
        user_config_path = config_dir / "test_config.json"
    else:
        config_dir = current_dir
        user_config_path = config_dir / "config.json"

    err_msg = f"""
        Please fill in the config file at {user_config_path}.

        REQUIRED:
        'Shared config path': Path to the shared config file on the network drive.

        OPTIONAL - if you want to interact directly with cyclers (e.g. load, eject, submit jobs):
        'SSH private key path': Path to the SSH private key file.
        'Snapshots folder path': Path to a (local) folder to store unprocessed snapshots e.g. 'C:/aurora-shapshots'.

        You can set the 'Shared config path' by running aurora-setup and following the instructions.
    """

    # if there is no user config file, create one
    if not user_config_path.exists():
        with user_config_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "Shared config path": "",
                        "SSH private key path": "",
                        "Snapshots folder path": "",
                    },
                    indent=4,
                ),
            )
            raise FileNotFoundError(err_msg)

    with user_config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    # Check for relative paths and convert to absolute paths
    for key in config:
        if "path" in key.lower() and config[key]:
            if not Path(config[key]).is_absolute():
                config[key] = Path(config_dir / config[key])
            else:
                config[key] = Path(config[key])

    # If there is a shared config file, update with settings from that file
    shared_config_path = config.get("Shared config path")
    if shared_config_path:
        with Path(shared_config_path).open(encoding="utf-8") as f:
            shared_config = json.load(f)

        # Check for relative paths and convert to absolute paths
        shared_config_dir = shared_config_path.parent
        for key in shared_config:
            if "path" in key.lower():
                if not Path(shared_config[key]).is_absolute():
                    shared_config[key] = Path(shared_config_dir / shared_config[key])
                else:
                    shared_config[key] = Path(shared_config[key])
        config.update(shared_config)

    if not config.get("Database path"):
        raise ValueError(err_msg)

    config["User config path"] = user_config_path

    return config


def get_config(reload: bool = False) -> dict:
    """Return global configuration dictionary.

    Only reads the config file once, unless reload is set to True.

    """
    global CONFIG  # noqa: PLW0603
    if CONFIG is None or reload:
        CONFIG = _read_config_file()
    return CONFIG
