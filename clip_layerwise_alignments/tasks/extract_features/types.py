# tasks/extract_features/types.py

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ExtractFeaturesContext:
    """Context for extracting CLIP features at multiple layers."""

    # Model config
    model_name: str
    pretrained: str

    # Layer configuration (1-indexed for display)
    text_layers: List[int]
    image_layers: List[int]

    # Layer indices (0-indexed for internal use)
    text_layer_indices: List[int]
    image_layer_indices: List[int]

    # What to extract
    extract_text_final: bool
    extract_image_final: bool

    # Data config
    dataset: str
    splits: List[str]

    # Processing config
    batch_size: int
    num_workers: int
    device: str

    # Paths
    flickr30k_dir: Path
    output_dir: Path

    # Environment
    environment: str

    def __post_init__(self):
        # Convert Path objects if strings were passed
        self.flickr30k_dir = Path(self.flickr30k_dir)
        self.output_dir = Path(self.output_dir)

        # Validate device
        valid_devices = ["cuda", "cpu"]
        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}, got '{self.device}'"
            )

        # Validate dataset
        valid_datasets = ["flickr30k"]
        if self.dataset not in valid_datasets:
            raise ValueError(
                f"dataset must be one of {valid_datasets}, got '{self.dataset}'"
            )

        # Validate splits
        valid_splits = ["train", "val", "test"]
        for split in self.splits:
            if split not in valid_splits:
                raise ValueError(
                    f"Invalid split '{split}'. Valid options: {valid_splits}"
                )

        # Validate layer numbers (1-12 for CLIP ViT-B/32)
        for layer in self.text_layers:
            if not 1 <= layer <= 12:
                raise ValueError(
                    f"Text layer {layer} out of range. Must be 1-12 for ViT-B/32."
                )

        for layer in self.image_layers:
            if not 1 <= layer <= 12:
                raise ValueError(
                    f"Image layer {layer} out of range. Must be 1-12 for ViT-B/32."
                )

        # Validate layer indices match
        expected_text_indices = [l - 1 for l in self.text_layers]
        if self.text_layer_indices != expected_text_indices:
            raise ValueError(
                f"text_layer_indices {self.text_layer_indices} don't match "
                f"text_layers {self.text_layers} (expected {expected_text_indices})"
            )

        expected_image_indices = [l - 1 for l in self.image_layers]
        if self.image_layer_indices != expected_image_indices:
            raise ValueError(
                f"image_layer_indices {self.image_layer_indices} don't match "
                f"image_layers {self.image_layers} (expected {expected_image_indices})"
            )

        # Validate batch size
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        # Validate num_workers
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

        # Validate input path exists
        if not self.flickr30k_dir.exists():
            raise FileNotFoundError(
                f"Flickr30k directory not found: {self.flickr30k_dir}"
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def images_dir(self) -> Path:
        """Path to Flickr30k images directory."""
        return self.flickr30k_dir / "images"

    @property
    def annotations_path(self) -> Path:
        """Path to Flickr30k annotations JSON."""
        return self.flickr30k_dir / "annotations" / "dataset_flickr30k.json"