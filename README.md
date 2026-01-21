# LayerAlign

Experimental pipeline for investigating cross-modal alignment emergence across CLIP encoder depths.

> **Part of the [LayerAlign Research Project](#)** - See the main article for detailed analysis, findings, and implications for vision-language model architectures.

<br>

## Overview

Multi-stage pipeline for analyzing at which architectural depths vision-language alignment emerges in CLIP models. Extracts features from intermediate transformer layers, trains projection mappings via contrastive learning, and evaluates alignment quality through image-text retrieval metrics.

### Capabilities

- **Multi-layer feature extraction**: Extract CLIP representations at intermediate transformer depths (layers 1, 3, 6, 9, 12) from both vision and text encoders
- **Projection training**: Train linear mappings from intermediate layers to shared embedding space using symmetric contrastive loss
- **Alignment evaluation**: Assess cross-modal alignment quality through bidirectional image-text retrieval metrics
- **Layer-wise analysis**: Compare alignment patterns across encoder depths to identify where semantic correspondence emerges

### Output

- **Intermediate features** - Vision and text representations extracted at multiple encoder depths (separate files per layer)
- **Trained projections** - Learned linear mappings from intermediate layers to shared embedding space
- **Projected embeddings** - Test set features mapped through trained projections for evaluation
- **Retrieval metrics** - Alignment quality scores (Recall@K) for each layer pairing

<br>

## Installation

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- GPU recommended (CUDA-compatible for feature extraction)
- Kaggle API credentials (for dataset download)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/jcruz-ferreyra/clip_layerwise_alignment.git
   cd clip_layerwise_alignment
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your paths:
   # LOCAL_DIR=/path/to/your/project/storage
   # DRIVE_DIR=/path/to/google/drive/storage  # Optional: for Colab workflows
   # DATA_FOLDER=data
   # MODELS_FOLDER=models
   ```

4. **Configure Kaggle credentials** (for dataset download)
   
   Create Kaggle API token and place credentials file:
   
   **Getting credentials:**
   - Create account at https://www.kaggle.com
   - Go to Account > API > Create New API Token
   - Download `kaggle.json` containing your credentials
   
   **Linux/Mac:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
   
   **Windows:**
   ```bash
   mkdir %USERPROFILE%\.kaggle
   move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

5. **Verify installation**
   ```bash
   poetry run python -c "import clip_layerwise_alignment; print('Installation successful!')"
   ```

<br>

## Quick Start

### Task 1: [Download Data](clip_layerwise_alignment/tasks/download_data)

Downloads datasets required for the pipeline (Flickr30k images and Karpathy split annotations).

**Configuration**:

Processing Configuration ([`config.yaml`](clip_layerwise_alignment/tasks/download_data/config.yaml))

YAML file defining which datasets to download and storage location:

```yaml
# Which datasets to download
datasets:
  - flickr30k

# Storage location
output_storage: "drive"  # "local" or "drive"
```

**Run**:
```bash
poetry run python -m clip_layerwise_alignment.tasks.download_data
```

**Output** (saved to `LOCAL_DIR/data/raw/` or `DRIVE_DIR/data/raw/`):
- `flickr30k/images/` - 31,783 images from Flickr30k dataset
- `flickr30k/annotations/` - Karpathy split JSON files (train/val/test splits)
- `flickr30k.zip` - Compressed archive (if `output_storage: "drive"` for Colab workflows)
- Processing logs show download progress and file counts

---

### Task 2: [Extract Features](clip_layerwise_alignment/tasks/extract_features)

Extracts CLIP features at multiple encoder depths from both vision and text transformers using pretrained models.

**Configuration**:

Processing Configuration ([`config.yaml`](clip_layerwise_alignment/tasks/extract_features/config.yaml))

YAML file defining model selection, layer extraction points, and processing settings:

```yaml
# Model configuration
model_name: "ViT-B-32"              # CLIP model architecture (see OpenCLIP model list)
pretrained: "openai"                # Pretrained weights source

# Layer extraction (1-indexed)
text_layers: [1, 3, 6, 9, 12]       # Extract these intermediate text encoder layers
image_layers: [1, 3, 6, 9, 12]      # Extract these intermediate vision encoder layers

extract_text_final: true            # Also extract final text embedding (after ln_final + projection)
extract_image_final: true           # Also extract final image embedding (after ln_post + projection)

# Data
input_dataset: "flickr30k"          # Dataset to process
splits: ["train", "test"]           # Which splits to extract

# Processing
batch_size: 16                      # Batch size for feature extraction
device: "cuda"                      # "cuda" or "cpu"

environment: "local"                # "local" or "colab" (for Drive→SSD extraction)
```

**Note**: Model must be a valid [OpenCLIP model](https://github.com/mlfoundations/open_clip). Different architectures produce different feature dimensions. This research uses `ViT-B-32` (vision: 768-dim intermediate, 512-dim final; text: 512-dim throughout).

**Run**:
```bash
poetry run python -m clip_layerwise_alignment.tasks.extract_features
```

**Output** (saved to `LOCAL_DIR/data/processed/flickr30k/{split}/`):
- `text_layer_1.pt` through `text_layer_12.pt` - Text encoder features [145k, 512] for train
- `text_final.pt` - Final text embeddings after full forward pass [145k, 512]
- `image_layer_1.pt` through `image_layer_12.pt` - Vision encoder features [145k, 768] for train (repeated 5× to match captions)
- `image_final.pt` - Final image embeddings after full forward pass [145k, 512] (repeated 5× to match captions)
- Each `.pt` file contains: `features` tensor, `metadata` dict (image IDs, captions), `layer` name
- Processing logs show extraction progress per split and layer

---

### Task 3: [Train Projections](clip_layerwise_alignment/tasks/train_projections)

Trains linear projection layers to map intermediate encoder features to shared embedding space via contrastive learning.

**Configuration**:

Processing Configuration ([`config.yaml`](clip_layerwise_alignment/tasks/train_projections/config.yaml))

YAML file defining layer pair, training hyperparameters, and model architecture:

```yaml
# Layer pair to train (trains ONE projection per run)
text_layer: 6                       # Text encoder layer (1-12 or "final")
image_layer: "final"                # Image encoder layer (1-12 or "final")

# Model architecture
projection_type: "linear"           # "linear" or "mlp"
use_layernorm: false                # Apply LayerNorm before projection

# Data
dataset: "flickr30k"                # Dataset with pre-extracted features

# Environment
environment: "local"                # "local" or "colab"
```

**Note**: Training parameters (learning rate, batch size, epochs, etc.) are optional and use sensible defaults if not specified. See full config for all available options.

**Run**:
```bash
poetry run python -m clip_layerwise_alignment.tasks.train_projections
```

**Output** (saved to `LOCAL_DIR/models/projections/` or `MODELS_DIR/projections/`):
- `text_L6_image_final.pt` - Trained projection checkpoint containing:
  - `model_state_dict`: Projection weights (and temperature if learnable)
  - `config`: Layer pair, architecture, training hyperparameters
  - `projection_name`: Identifier for this layer pairing
- `text_L6_image_final_history.json` - Training history with per-epoch metrics (train_loss, val_loss, lr)
- Processing logs show training progress, loss curves, and early stopping events

---

### Task 4: [Project Eval Samples](clip_layerwise_alignment/tasks/project_eval_samples)

Projects evaluation set features through trained projection layers to generate embeddings for downstream retrieval evaluation.

**Configuration**:

Processing Configuration ([`config.yaml`](clip_layerwise_alignment/tasks/project_eval_samples/config.yaml))

YAML file defining which layer pairs to project and evaluation settings:

```yaml
# Layer pairs to project (must match trained projections)
layer_pairs:
  - text_layer: 1
    image_layer: "final"
  - text_layer: 3
    image_layer: "final"
  - text_layer: 6
    image_layer: "final"
  # Add all pairs you want to evaluate

# Projection type
projection_type: "linear"           # Must match training configuration

# Data
dataset: "flickr30k"                # Dataset with pre-extracted features
eval_split: "test"                  # Split to project ("test" or "val")

# Paths
projections_subdir: "projections"   # Subdirectory containing trained projections

# Environment
environment: "local"                # "local" or "colab"
```

**Run**:
```bash
poetry run python -m clip_layerwise_alignment.tasks.project_eval_samples
```

**Output** (saved to `LOCAL_DIR/results/flickr30k/{eval_split}/`):
- `text_L1_image_final_projected.pt` - Projected text embeddings containing:
  - `text_embeddings`: Projected features [N_samples, 512]
  - `layer_pair`: Source and target layer configuration
  - `modality`: Which encoder was projected ("text" or "image")
  - `metadata`: Image IDs and caption mappings
  - `eval_split`: Which split was processed
- One file per layer pair in `layer_pairs` configuration
- Processing logs show projection progress per pair and file sizes

---

### Bonus: [Analysis Notebooks](notebooks/)

Jupyter notebooks for evaluating and comparing alignment quality across different layer pairings using projected embeddings.

**Purpose**:

After obtaining projected test features from Task 4, use these notebooks to:
- Evaluate image-text retrieval metrics (Recall@K) for each layer pairing
- Compare alignment quality across encoder depths
- Visualize alignment emergence patterns
- Analyze cross-modal correspondence at different architectural levels

**Location**: `notebooks/` directory contains interactive analysis tools for exploring experimental results.

<br>

## Structure

### Task Architecture

Each task within `clip_layerwise_alignment/tasks/` folder follows a consistent structure:

```
extract_features/
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point - handles config loading and orchestration
├── config.yaml                 # Processing configuration (user's working copy)
├── types.py                    # Context dataclass definition with validation
├── extract_features.py         # Core processing logic (called from __main__.py)
└── *.py                        # Modular helper functions (called from extract_features.py)
```

**Context Pattern**:

All tasks use a context object to eliminate parameter passing complexity:

```python
@dataclass
class ExtractFeaturesContext:
    # Configuration from YAML
    model_name: str
    pretrained: str
    text_layers: List[int]
    image_layers: List[int]
    batch_size: int
    device: str
    ...
    
    # Computed paths (using @property decorators)
    flickr30k_dir: Path
    output_dir: Path
    
    # Runtime objects (initialized during setup)
    text_layer_indices: List[int]  # Converted to 0-indexed
    image_layer_indices: List[int]
    ...
```

This pattern provides:
- Centralized configuration and state management
- Automated path computation using `@property` decorators
- Type validation and conversion in `__post_init__` method
- Clear separation between user-facing config (1-indexed layers) and internal representation (0-indexed)

<br>

## How It Works

### Task 1: [Download Data](clip_layerwise_alignment/tasks/download_data)

Downloads datasets required for the pipeline with automatic resume capability and optional compression for Colab workflows.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. Initialization
   - Authenticate Kaggle API using credentials from `~/.kaggle/kaggle.json`
   - Validate output directories (LOCAL_DIR or DRIVE_DIR based on storage setting)
   - Check for existing downloads to enable resume capability
2. Dataset Download Loop
   - For each dataset in configuration (currently supports Flickr30k):
     - Check if already downloaded (31k+ images present)
     - Download dataset from Kaggle if needed
     - Download Karpathy split annotations from external source
     - Verify download integrity
3. Flickr30k Specific Processing
   - **Images**: Download via Kaggle API (`hsankesara/flickr-image-dataset`)
   - **Annotations**: Fetch Karpathy splits (standard train/val/test JSON files)
   - **Verification**: Check image count (31,783 expected) and annotation files exist
4. Drive Storage Preparation (if `output_storage: "drive"`)
   - Create compressed `.zip` archive of entire dataset
   - Save to DRIVE_DIR for future Colab workflows
   - Enables efficient extraction in subsequent tasks (extract_features, train_projections)
   - Original uncompressed files saved for local development

**Key Features**:
- **Resume capability**: Skips already downloaded datasets (checks image count and annotation presence)
- **Kaggle integration**: Automated API authentication and dataset download
- **Drive storage option**: Saves compressed datasets to Google Drive for Colab usage in later tasks
- **Modular design**: Easy to extend with additional datasets (ImageNet100, COCO interfaces defined)

</details>

---

### Task 2: [Extract Features](clip_layerwise_alignment/tasks/extract_features)

Extracts CLIP features at multiple encoder depths from both vision and text transformers using OpenCLIP's intermediate layer access.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. Environment Setup
   - **Colab mode** (`environment: "colab"`): Extract dataset from Drive `.zip` to local SSD (`/content/data/raw/`) for fast access
   - **Local mode** (`environment: "local"`): Use dataset directly from LOCAL_DIR
   - Load pretrained CLIP model (ViT-B-32) with specified weights
2. Model Loading
   - Initialize OpenCLIP model with `create_model_and_transforms()`
   - Load preprocessing pipeline (resize, normalize)
   - Load tokenizer for text encoding
   - Move model to specified device (GPU recommended)
   - Set to evaluation mode (no gradient computation)
3. Feature Extraction Per Split (train/test)
   - Load Flickr30k annotations (Karpathy split JSON)
   - **Vision features**:
     - Extract intermediate layers using `forward_intermediates()` API
     - Extract CLS token from each layer [batch_size, 768] (pre-normalization)
     - Extract final embeddings via full forward pass [batch_size, 512] (post-projection)
   - **Text features**:
     - Extract intermediate layers using `forward_intermediates()` API  
     - Extract EOS token from each layer [batch_size, 512] (pre-normalization)
     - Extract final embeddings via full forward pass [batch_size, 512] (post-projection)
4. Feature Pairing and Storage
   - Repeat image features 5× to match caption count (29k images → 145k entries)
   - Ensures direct correspondence: `text_features[i]` ↔ `image_features[i]`
   - Save each layer to separate `.pt` file for memory-efficient loading
   - Each file contains: `features` tensor, `metadata` dict, `layer` name

**Key Algorithms**:
- **forward_intermediates()**: OpenCLIP method for extracting multiple layer outputs in single forward pass
- **Token extraction**: CLS token for vision (class token), EOS token for text (end-of-sequence marker)
- **Feature pairing**: Robust image-text alignment via feature repetition (eliminates manual indexing)
- **Colab optimization**: Drive→SSD extraction enables fast I/O during processing (10-20× faster than reading from Drive)

**Technical Details**:
- Vision features: 768-dim at intermediate layers (pre-ln_post), 512-dim final (post-projection)
- Text features: 512-dim throughout (consistent dimensionality across all layers)
- Intermediate features are **unnormalized** (raw layer outputs) for studying natural representations
- Final features are **normalized and projected** (standard CLIP embeddings)

</details>

---

### Task 3: [Train Projections](clip_layerwise_alignment/tasks/train_projections)

Trains linear projection layers to map intermediate encoder features to shared embedding space via symmetric contrastive learning.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. Feature Loading
   - Load pre-extracted features for specified layer pair (text_layer → image_layer)
   - Features loaded directly from `.pt` files (no unzipping needed, works on Drive or local)
   - Creates train/val dataloaders with contrastive sampling
   - Batch size typically 256 (sufficient negatives for contrastive loss)
2. Model Initialization
   - Create projection architecture based on config:
     - **Linear**: `nn.Linear(d_in, 512, bias=False)` (default)
     - **MLP**: Two-layer with GELU activation
   - Optional LayerNorm before projection (recommended for intermediate layers)
   - Learnable temperature parameter (initialized to 0.07, CLIP default)
   - Wrap in ProjectionWrapper for symmetric processing
3. Training Loop
   - **Optimizer**: AdamW (matches CLIP paper, decoupled weight decay)
   - **Scheduler**: Linear warmup → constant plateau → cosine decay
   - **Loss**: Symmetric contrastive loss (image→text + text→image cross-entropy)
   - **Early stopping**: Monitor validation loss with patience and min_delta thresholds
   - **Checkpointing**: Save best model based on validation performance
4. Model Saving
   - Save checkpoint containing:
     - `model_state_dict`: Projection weights (and temperature if learnable)
     - `config`: Layer pair, architecture, all hyperparameters
     - `projection_name`: Identifier (e.g., "text_L6_image_final")
   - Save training history as JSON with per-epoch metrics

**Key Algorithms**:
- **Symmetric contrastive loss**: Computes both image→text and text→image similarities, applies cross-entropy on both directions
- **Learning rate scheduling**: Warmup prevents unstable early training, plateau allows convergence, cosine decay fine-tunes
- **Early stopping**: Prevents overfitting by monitoring validation loss plateaus
- **Feature pairing**: Pre-paired features (from Task 2) enable efficient batch construction without runtime indexing

**Technical Details**:
- Input dimensions vary by layer: 768-dim (vision intermediate), 512-dim (text all layers, vision final)
- Output dimension always 512 (shared embedding space)
- Temperature learned during training (typically stays near 0.07)
- LayerNorm helps align distribution statistics between intermediate and final features

</details>

---

### Task 4: [Project Eval Samples](clip_layerwise_alignment/tasks/project_eval_samples)

Projects evaluation set features through trained projection layers to generate embeddings for downstream retrieval evaluation.

<details>
<summary><b>Details</b></summary>
<br>

**Processing Pipeline**:
1. Layer Pair Iteration
   - Process each layer pair specified in configuration sequentially
   - Load corresponding trained projection checkpoint
   - Load evaluation split features (test or val) for the specific layer pair
2. Checkpoint Loading
   - Load trained projection weights from `models/projections/`
   - Restore model architecture (Linear/MLP + optional LayerNorm)
   - Load learned temperature parameter
   - Set model to evaluation mode (no gradient computation)
3. Feature Projection
   - Load intermediate features for source layer (text or image)
   - Project through trained model to shared 512-dim space
   - Uses dummy tensors for unused modality (e.g., dummy image when projecting text)
   - Single forward pass per sample (no batching needed, fast inference)
4. Embedding Storage
   - Save projected embeddings with metadata:
     - Embeddings tensor [N_samples, 512]
     - Layer pair configuration (source and target layers)
     - Modality identifier (text or image)
     - Metadata (image IDs, caption mappings)
     - Evaluation split name
   - One output file per layer pair

**Key Features**:
- **Batch processing**: Handles multiple layer pairs in single run (processes all trained projections)
- **Memory efficient**: Loads only necessary features per pair, no batching overhead
- **Fast inference**: Single forward pass with no gradient computation
- **Self-contained outputs**: Each file includes all metadata needed for evaluation

**Technical Details**:
- Dummy tensors used for ProjectionWrapper compatibility (model expects both modalities)
- Output embeddings always 512-dim regardless of source layer dimensions
- Preserves exact feature ordering from extraction for retrieval evaluation
- Files saved to `results/` directory, separate from training checkpoints

</details>

<br>

## Additional Resources

For detailed analysis, findings, and implications for vision-language model architectures, see the **[LayerAlign Research Article](#)**.

### Related Technologies

- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** - Open source CLIP implementation with intermediate layer access
- **[Flickr30k Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)** - Image-caption pairs used in experiments
- **[Karpathy Splits](http://cs.stanford.edu/people/karpathy/deepimagesent/)** - Standard train/val/test splits for image-caption datasets

### Support

For questions or issues:
- **GitHub Issues**: [clip_layerwise_alignment/issues](https://github.com/yourusername/clip_layerwise_alignment/issues)

### Citation

If you use this pipeline in your research, please cite:
```bibtex
@software{layeralign2025,
  title={LayerAlign: Investigating Cross-Modal Alignment Across CLIP Encoder Depths},
  author={Your Name},
  institution={Northeastern University},
  year={2025},
  url={https://github.com/yourusername/clip_layerwise_alignment}
}
```

### License

MIT License - see [LICENSE](LICENSE) file for details.


