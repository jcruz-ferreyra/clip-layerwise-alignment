import logging
from pathlib import Path
import shutil
import time
import zipfile

logger = logging.getLogger(__name__)


def retrieve_and_unzip_data(dataset_dir: Path, dataset_folder: str):
    """Download and extract dataset from drive to colab local storage."""
    logger.info("Starting dataset retrieval and extraction for Colab environment")

    dataset_name = dataset_dir.name
    zipfile_name = f"{dataset_name}.zip"

    colab_data_dir = Path("/content/data")
    colab_dataset_dir = colab_data_dir / dataset_folder
    colab_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy zip file to colab
    src = dataset_dir.parent / zipfile_name
    dst = colab_dataset_dir.parent / zipfile_name

    # Validate source file exists
    if not src.exists():
        logger.error(f"Source zip file not found: {src}")
        raise FileNotFoundError(f"Dataset zip file not found: {src}")

    try:
        logger.info(f"Copying dataset from drive: {src}")
        start = time.time()
        shutil.copy2(src, dst)
        copy_time = time.time() - start

        # Log copy statistics
        file_size_mb = src.stat().st_size / (1024 * 1024)
        logger.info(
            f"Copied {file_size_mb:.1f} MB in {copy_time:.2f} seconds "
            f"({file_size_mb/copy_time:.1f} MB/s)"
        )

    except Exception as e:
        logger.error(f"Failed to copy dataset zip file: {e}")
        raise

    try:
        logger.info(f"Extracting dataset to: {colab_data_dir}")
        start = time.time()

        with zipfile.ZipFile(dst, "r") as zip_ref:
            zip_ref.extractall(colab_dataset_dir)

        extract_time = time.time() - start

        # Count extracted files
        extracted_files = len(list(colab_data_dir.rglob("*")))
        logger.info(f"Extracted {extracted_files} files in {extract_time:.2f} seconds")

    except zipfile.BadZipFile as e:
        logger.error(f"Corrupted zip file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        raise

    try:
        # Clean up zip file to save space
        dst.unlink()
        logger.info("Cleaned up zip file to save space")

    except Exception as e:
        logger.warning(f"Failed to clean up zip file: {e}")

    logger.info("Dataset retrieval and extraction completed successfully")

    # Update context dataset directory
    return colab_data_dir


def retrieve_and_unzip_data(
    source_zip: Path, extract_to: Path = Path("/content/data/raw"), dataset_name: str = "flickr30k"
) -> Path:
    """
    Extract dataset from Drive to Colab local storage (Colab only).

    Handles both zip structures:
    - With top folder: flickr30k.zip → flickr30k/images/, flickr30k/annotations/
    - Without top folder: flickr30k.zip → images/, annotations/

    Args:
        source_zip: Path to zip file on Drive
        extract_to: Directory to extract to (default: /content/data/raw)
        dataset_name: Name of dataset folder (default: "flickr30k")

    Returns:
        Path to extracted dataset on Colab local storage
    """
    logger.info("=" * 80)
    logger.info("Setting up dataset for Colab environment")
    logger.info("=" * 80)

    if not source_zip.exists():
        raise FileNotFoundError(
            f"Dataset zip not found on Drive: {source_zip}\n"
            f"Please upload {dataset_name}.zip to Drive"
        )

    extracted_dir = extract_to / dataset_name

    # Check if already extracted
    if extracted_dir.exists():
        num_files = len(list(extracted_dir.rglob("*")))
        if num_files > 1000:
            logger.info(f"✓ Dataset already extracted at {extracted_dir}")
            logger.info(f"  Found {num_files} existing files")
            return extracted_dir

    extract_to.mkdir(parents=True, exist_ok=True)
    temp_zip = extract_to / f"{dataset_name}_temp.zip"

    try:
        # Copy from Drive to Colab
        logger.info(f"Copying from Drive: {source_zip}")
        file_size_mb = source_zip.stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {file_size_mb:.1f} MB")

        start = time.time()
        shutil.copy2(source_zip, temp_zip)
        copy_time = time.time() - start

        logger.info(f"✓ Copied in {copy_time:.1f}s ({file_size_mb/copy_time:.1f} MB/s)")

        # Peek at zip structure (cheap!)
        logger.info("Checking zip structure...")
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            file_list = zip_ref.namelist()

            # Check if top-level folder exists
            has_top_folder = file_list[0].startswith(f"{dataset_name}/")
            logger.info(f"  Zip has top-level '{dataset_name}/' folder: {has_top_folder}")
            logger.info(f"  First file in zip: {file_list[0]}")

        # Extract
        start = time.time()

        if has_top_folder:
            # Extract to parent (zip contains flickr30k/ already)
            logger.info(f"Extracting to: {extract_to}")
            with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            # Extract directly into dataset folder (zip has no top folder)
            logger.info(f"Extracting to: {extracted_dir}")
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                zip_ref.extractall(extracted_dir)

        extract_time = time.time() - start
        num_files = len(list(extracted_dir.rglob("*")))

        logger.info(f"✓ Extracted {num_files} files in {extract_time:.1f}s")

        # Cleanup
        temp_zip.unlink()
        logger.info("✓ Cleaned up temporary zip")

    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        if temp_zip.exists():
            temp_zip.unlink()
        raise

    logger.info("=" * 80)
    logger.info(f"✓ Dataset ready at: {extracted_dir}")
    logger.info("=" * 80)

    return extracted_dir
