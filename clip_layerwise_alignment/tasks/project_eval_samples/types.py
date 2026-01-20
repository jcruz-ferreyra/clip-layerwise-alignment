# tasks/project_eval_samples/types.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union


@dataclass
class ProjectEvalContext:
    """Context for projecting evaluation samples through trained projections."""

    # Layer pairs to evaluate
    layer_pairs: List[Dict[str, Union[int, str]]]

    # Data config
    dataset: str
    eval_split: str

    # Paths
    features_dir: Path
    projections_dir: Path
    output_dir: Path

    # Model architecture
    projection_type: str = "linear"
    use_layernorm: bool = False

    # Environment
    environment: str = "local"

    def __post_init__(self):
        # Convert to Path
        self.features_dir = Path(self.features_dir)
        self.projections_dir = Path(self.projections_dir)
        self.output_dir = Path(self.output_dir)

        # Validate environment
        if self.environment not in ["local", "colab"]:
            raise ValueError(f"environment must be 'local' or 'colab', got '{self.environment}'")

        # Validate dataset
        if self.dataset != "flickr30k":
            raise ValueError(f"Only 'flickr30k' supported, got '{self.dataset}'")

        # Validate eval split
        if self.eval_split not in ["train", "val", "test"]:
            raise ValueError(f"eval_split must be train/val/test, got '{self.eval_split}'")

        # Validate layer pairs
        if not self.layer_pairs:
            raise ValueError("layer_pairs cannot be empty")

        for pair in self.layer_pairs:
            if "text_layer" not in pair or "image_layer" not in pair:
                raise ValueError(f"Each layer pair must have text_layer and image_layer: {pair}")

        # Validate layers values
        for layer_pair in self.layer_pairs:
            for layer_name, layer_val in layer_pair.items():
                if isinstance(layer_val, int):
                    if not 1 <= layer_val <= 12:
                        raise ValueError(
                            f"{layer_name} must be 1-12 or 'final', got {layer_val} in {layer_pair}"
                        )
                elif layer_val != "final":
                    raise ValueError(
                        f"{layer_name} must be 1-12 or 'final', got '{layer_val}' in {layer_pair}"
                    )

        # Validate projection type
        if self.projection_type not in ["linear", "mlp"]:
            raise ValueError(
                f"projection_type must be 'linear' or 'mlp', got '{self.projection_type}'"
            )

        # Validate paths exist
        eval_features_dir = self.features_dir / self.eval_split
        if not eval_features_dir.exists():
            raise FileNotFoundError(f"Eval features not found: {eval_features_dir}")

        if not self.projections_dir.exists():
            raise FileNotFoundError(f"Projections directory not found: {self.projections_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def eval_features_dir(self) -> Path:
        """Path to evaluation features."""
        return self.features_dir / self.eval_split

    def get_projection_name(
        self, text_layer: Union[int, str], image_layer: Union[int, str]
    ) -> str:
        """Generate projection name from layer pair."""
        text_str = f"L{text_layer}" if isinstance(text_layer, int) else "final"
        image_str = f"L{image_layer}" if isinstance(image_layer, int) else "final"
        return f"text_{text_str}_image_{image_str}"

    def get_projection_path(
        self, text_layer: Union[int, str], image_layer: Union[int, str]
    ) -> Path:
        """Get path to projection checkpoint."""
        return self.projections_dir / f"{self.get_projection_name(text_layer, image_layer)}.pt"

    def get_output_path(self, text_layer: Union[int, str], image_layer: Union[int, str]) -> Path:
        """Get path to save projected embeddings."""
        return (
            self.output_dir / f"{self.get_projection_name(text_layer, image_layer)}_projected.pt"
        )
