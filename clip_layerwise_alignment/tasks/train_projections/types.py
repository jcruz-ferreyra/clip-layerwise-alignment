# tasks/train_projections/types.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class TrainProjectionsContext:
    """Context for training projection layers between CLIP encoder layers."""

    # Layer pair to train
    text_layer: Union[int, str]  # 1-12 or "final"
    image_layer: Union[int, str]  # 1-12 or "final"

    # Training parameters (with defaults)
    training_params: Dict[str, Union[float, int, bool]] = field(default_factory=dict)

    # Model architecture
    projection_type: str = "linear"
    use_layernorm: bool = False

    # Data config
    dataset: str = "flickr30k"

    # Processing
    device: Optional[str] = "cuda"
    num_workers: int = 4

    # Paths
    features_dir: Path = None
    checkpoints_dir: Path = None

    # Environment
    environment: str = "local"

    def __post_init__(self):
        # Convert to Path
        self.features_dir = Path(self.features_dir)
        self.checkpoints_dir = Path(self.checkpoints_dir)

        # Fill training params with defaults
        self.training_params = _fill_training_params_with_defaults(self.training_params)

        if self.training_params["optimizer"].lower() not in ["adam", "adamw"]:
            raise ValueError(
                f"optimizer must be 'adam' or 'adamw', got '{self.training_params['optimizer']}'"
            )

        # Validate device
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{self.device}'")

        # Validate environment
        if self.environment not in ["local", "colab"]:
            raise ValueError(f"environment must be 'local' or 'colab', got '{self.environment}'")

        # Validate projection type
        if self.projection_type not in ["linear", "mlp"]:
            raise ValueError(
                f"projection_type must be 'linear' or 'mlp', got '{self.projection_type}'"
            )

        # Validate layers
        for layer_name, layer_val in [
            ("text_layer", self.text_layer),
            ("image_layer", self.image_layer),
        ]:
            if isinstance(layer_val, int):
                if not 1 <= layer_val <= 12:
                    raise ValueError(f"{layer_name} must be 1-12 or 'final', got {layer_val}")
            elif layer_val != "final":
                raise ValueError(f"{layer_name} must be 1-12 or 'final', got '{layer_val}'")

        # Validate features directory exists
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

        # Create checkpoints directory
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    @property
    def features_dir_train(self) -> Path:
        """Path to save/load projection checkpoint."""
        return self.features_dir / "train"

    @property
    def features_dir_val(self) -> Optional[Path]:
        """Path to save/load projection checkpoint."""
        val_dir = self.features_dir / "val"
        return val_dir if val_dir.exists() else None

    @property
    def projection_name(self) -> str:
        """Generate projection name from layer pair."""
        text_str = f"L{self.text_layer}" if isinstance(self.text_layer, int) else "final"
        image_str = f"L{self.image_layer}" if isinstance(self.image_layer, int) else "final"

        return f"text_{text_str}_image_{image_str}"

    @property
    def checkpoint_path(self) -> Path:
        """Path to save/load projection checkpoint."""
        return self.checkpoints_dir / f"{self.projection_name}.pt"


def _fill_training_params_with_defaults(training_params: Dict) -> Dict:
    """Fill missing training parameters with default values."""
    defaults = {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 10,
        "weight_decay": 0.1,
        "temperature": 0.07,
        "learnable_temperature": False,
    }

    # Merge: defaults first, then override with user values
    return {**defaults, **training_params}
