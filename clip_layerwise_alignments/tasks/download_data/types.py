# tasks/download_data/types.py

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DownloadDataContext:
    """Context for downloading datasets."""

    output_data_dir: Path
    datasets: List[str]  # ["flickr30k", "imagenet100"]
    output_storage: str = "local"  # "local" or "drive"

    def __post_init__(self):
        # Validate output directory
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        # Validate storage option
        valid_storages = ["local", "drive"]
        if self.output_storage not in valid_storages:
            raise ValueError(
                f"output_storage must be one of {valid_storages}, " f"got '{self.output_storage}'"
            )

        # Validate datasets
        valid_datasets = ["flickr30k", "imagenet100", "coco"]
        for dataset in self.datasets:
            if dataset not in valid_datasets:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. " f"Valid options: {valid_datasets}"
                )

    @property
    def flickr30k_dir(self) -> Path:
        """Path to flickr30k raw data."""
        return self.output_data_dir / "raw" / "flickr30k"

    @property
    def imagenet100_dir(self) -> Path:
        """Path to imagenet100 raw data."""
        return self.output_data_dir / "raw" / "imagenet100"

    @property
    def coco_dir(self) -> Path:
        """Path to COCO raw data."""
        return self.output_data_dir / "raw" / "coco"
