# clip_layerwise_alignments/config.py

import os
from pathlib import Path
from dotenv import load_dotenv


def _get_drive_mount_point(drive_path: Path) -> Path:
    """
    Get the Drive mount point to check if Drive is accessible.
    """
    parts = drive_path.parts

    # Windows: Check drive letter (e.g., "G:/")
    if len(parts) > 0 and parts[0].endswith(":"):
        return Path(parts[0] + "\\")  # or Path(parts[0] + '/')

    # Colab/Linux: Check for /content/drive/MyDrive
    if len(parts) >= 3 and "MyDrive" in parts:
        idx = parts.index("MyDrive")
        return Path(*parts[: idx + 1])  # /content/drive/MyDrive

    # Fallback: Check if parent exists
    return drive_path.parent


# Load environment variables
load_dotenv()

# Base paths
HOME_DIR = Path(os.getenv("HOME_DIR")).resolve()
LOCAL_DIR = Path(os.getenv("LOCAL_DIR")).resolve()

# Folder names
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", "data"))
MODELS_FOLDER = Path(os.getenv("MODELS_FOLDER", "models"))

# Local paths
LOCAL_DATA_DIR = LOCAL_DIR / DATA_FOLDER
LOCAL_MODELS_DIR = LOCAL_DIR / MODELS_FOLDER

# Validate local paths exist
for path_name, path in [("HOME_DIR", HOME_DIR), ("LOCAL_DIR", LOCAL_DIR)]:
    if not path.exists():
        raise ValueError(f"{path_name} path '{path}' from .env does not exist.")

# Optional: Drive paths
DRIVE_DIR = os.getenv("DRIVE_DIR")
if DRIVE_DIR:
    DRIVE_DIR = Path(DRIVE_DIR)

    # Check if Drive is mounted (check root/mount point, not final folder)
    drive_mount_point = _get_drive_mount_point(DRIVE_DIR)

    if drive_mount_point.exists():
        DRIVE_DIR = DRIVE_DIR.resolve()
        DRIVE_DATA_DIR = DRIVE_DIR / DATA_FOLDER
        DRIVE_MODELS_DIR = DRIVE_DIR / MODELS_FOLDER

        # Create directories if they don't exist (on mounted drive)
        DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        DRIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        DRIVE_DIR = None
        DRIVE_DATA_DIR = None
        DRIVE_MODELS_DIR = None
else:
    DRIVE_DIR = None
    DRIVE_DATA_DIR = None
    DRIVE_MODELS_DIR = None
