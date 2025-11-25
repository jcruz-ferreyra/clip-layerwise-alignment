# tasks/download_data/download_data.py

import logging
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile
import shutil

from .types import DownloadDataContext

logger = logging.getLogger(__name__)


# ============================================================================
# Helper functions (top to bottom in order they're called)
# ============================================================================


def _validate_output_directories(ctx: DownloadDataContext) -> None:
    """Ensure output directories exist and are writable."""
    logger.info("Validating output directories")

    raw_dir = ctx.output_data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Test write permissions
    test_file = raw_dir / ".write_test"
    test_file.touch()
    test_file.unlink()

    logger.info(f"✓ Output directory validated: {raw_dir}")


def _download_flickr30k_images(images_dir: Path) -> None:
    """Download Flickr30k images using Kaggle API."""
    logger.info("Downloading Flickr30k images via Kaggle API")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

    except OSError as e:
        # Kaggle raises OSError when credentials not found
        logger.error("Kaggle API credentials not found!")
        logger.error("Please set up Kaggle API:")
        logger.error("1. Create account at https://www.kaggle.com")
        logger.error("2. Go to Account > API > Create New API Token")
        logger.error(
            "3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)"
        )
        logger.error("4. Run: chmod 600 ~/.kaggle/kaggle.json (Linux/Mac)")
        raise RuntimeError("Kaggle API credentials not configured") from e

    # Download dataset
    dataset = "hsankesara/flickr-image-dataset"
    logger.info(f"Downloading dataset: {dataset}")
    logger.info("This may take several minutes (images are ~2GB)...")

    # Download to temporary directory
    temp_dir = images_dir.parent / "temp_download"
    temp_dir.mkdir(exist_ok=True)

    try:
        api.dataset_download_files(dataset, path=str(temp_dir), unzip=True)

        # Move images to correct location
        downloaded_files = list(temp_dir.rglob("*.jpg"))

        if not downloaded_files:
            raise RuntimeError("No image files found in downloaded dataset")

        logger.info(f"Found {len(downloaded_files)} images, moving to {images_dir}")
        images_dir.mkdir(exist_ok=True)

        # Move all images to images_dir
        for img_file in tqdm(downloaded_files, desc="Moving images"):
            shutil.move(str(img_file), str(images_dir / img_file.name))

        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        logger.info(f"✓ Downloaded {len(downloaded_files)} images")

    except Exception as e:
        # Cleanup on failure
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def _download_flickr30k_annotations(annotations_dir: Path) -> None:
    """Download Flickr30k annotations from GitHub."""
    logger.info("Downloading Flickr30k annotations")

    # Download captions from Karpathy's splits (standard for CLIP evaluation)
    captions_url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

    temp_zip = annotations_dir.parent / "temp_captions.zip"

    try:
        logger.info(f"Downloading from: {captions_url}")
        response = requests.get(captions_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(temp_zip, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="Annotations") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Extract
        logger.info(f"Extracting to: {annotations_dir}")
        annotations_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(annotations_dir)

        temp_zip.unlink()
        logger.info("✓ Annotations downloaded")

    except Exception as e:
        if temp_zip.exists():
            temp_zip.unlink()
        raise


def _download_flickr30k(ctx: DownloadDataContext) -> None:
    """Download Flickr30k dataset (images + annotations)."""
    logger.info("=" * 60)
    logger.info("Downloading Flickr30k dataset")
    logger.info("=" * 60)

    output_dir = ctx.flickr30k_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"

    # Check if already downloaded
    if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) > 30000:
        logger.info("✓ Flickr30k images already exist, skipping image download")
    else:
        _download_flickr30k_images(images_dir)

    # Download annotations
    if annotations_dir.exists() and any(annotations_dir.iterdir()):
        logger.info("✓ Flickr30k annotations already exist, skipping annotation download")
    else:
        _download_flickr30k_annotations(annotations_dir)

    logger.info("=" * 60)
    logger.info(f"✓ Flickr30k complete: {output_dir}")
    logger.info("=" * 60)

    # Create zip if using Drive storage
    if ctx.output_storage == "drive":
        _zip_dataset(output_dir, "flickr30k")


def _download_imagenet100(ctx: DownloadDataContext) -> None:
    """ImageNet-100 not implemented - not needed for this project phase."""
    raise NotImplementedError(
        "ImageNet-100 download not implemented. "
        "For this project, we're using Flickr30k for training and evaluation. "
        "You can optionally use CIFAR-100 (auto-downloaded by PyTorch) for zero-shot classification."
    )


def _download_coco(ctx: DownloadDataContext) -> None:
    """COCO download not implemented - not needed for this project phase."""
    raise NotImplementedError(
        "COCO download not implemented. "
        "For this project, we're using Flickr30k for training and evaluation. "
        "COCO can be added later if needed for cross-dataset evaluation."
    )


def _zip_dataset(dataset_dir: Path, dataset_name: str) -> None:
    """
    Create a compressed zip archive of the dataset for Drive storage.

    Args:
        dataset_dir: Path to dataset directory (e.g., .../raw/flickr30k)
        dataset_name: Name of dataset (e.g., "flickr30k")
    """
    logger.info("=" * 60)
    logger.info(f"Creating zip archive for {dataset_name}")
    logger.info("=" * 60)

    # Create zip in parent directory
    zip_path = dataset_dir.parent / f"{dataset_name}.zip"

    # Skip if zip already exists and is valid
    if zip_path.exists():
        try:
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zf:
                if zf.testzip() is None:  # Zip is valid
                    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                    logger.info(f"✓ Valid zip already exists: {zip_path} ({zip_size_mb:.1f} MB)")
                    logger.info("  Skipping zip creation")
                    return
        except Exception:
            logger.info("Existing zip is invalid, recreating...")
            zip_path.unlink()

    try:
        import shutil
        import time

        logger.info(f"Source: {dataset_dir}")
        logger.info(f"Target: {zip_path}")

        # Calculate original size
        original_size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
        original_size_mb = original_size / (1024 * 1024)
        logger.info(f"Compressing {original_size_mb:.1f} MB...")

        # Create zip archive
        start = time.time()
        shutil.make_archive(
            base_name=str(zip_path.with_suffix("")),  # Remove .zip (added automatically)
            format="zip",
            root_dir=str(dataset_dir.parent),
            base_dir=dataset_dir.name,
        )
        zip_time = time.time() - start

        # Log statistics
        zip_size = zip_path.stat().st_size
        zip_size_mb = zip_size / (1024 * 1024)
        compression_ratio = (1 - zip_size / original_size) * 100 if original_size > 0 else 0

        logger.info("=" * 60)
        logger.info("✓ Dataset archived successfully")
        logger.info(f"  Archive: {zip_path}")
        logger.info(f"  Original size: {original_size_mb:.1f} MB")
        logger.info(f"  Compressed size: {zip_size_mb:.1f} MB")
        logger.info(f"  Compression: {compression_ratio:.1f}%")
        logger.info(f"  Time: {zip_time:.1f}s ({original_size_mb/zip_time:.1f} MB/s)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to create zip archive: {e}")
        raise


# ============================================================================
# Main public function
# ============================================================================


def download_data(ctx: DownloadDataContext) -> None:
    """
    Download all configured datasets.

    Args:
        ctx: DownloadDataContext containing dataset list and output paths
    """
    logger.info("Starting dataset download process")

    _validate_output_directories(ctx)

    for dataset_name in ctx.datasets:
        logger.info(f"Processing dataset: {dataset_name}")

        try:
            if dataset_name == "flickr30k":
                _download_flickr30k(ctx)
            elif dataset_name == "imagenet100":
                _download_imagenet100(ctx)
            elif dataset_name == "coco":
                _download_coco(ctx)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}, skipping")

        except NotImplementedError as e:
            logger.warning(f"Skipping {dataset_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            raise

    logger.info("✓ Dataset download completed successfully")
