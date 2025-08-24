"""
Utility to convert **SigMF (Signal Metadata Format)** datasets into **COCO format**
for object detection and recognition tasks in RF signal processing.

SigMF is the standard format for RF signal datasets, containing signal data and metadata.
This tool converts SigMF annotations into COCO format suitable for training YOLO and
other object detection models on RF spectrograms.

Input Structure
---------------

Expected SigMF dataset structure:

.. code-block::

    <SIGMF_ROOT>
    └── data/
        ├── signal_001.sigmf-data
        ├── signal_001.sigmf-meta
        ├── signal_002.sigmf-data
        └── signal_002.sigmf-meta


Output Structure
----------------

Produces COCO-formatted dataset:

.. code-block::

    <OUT_DIR>
    ├── train/
    │   ├── _annotations.coco.json
    │   ├── 000000000001.png
    │   └── 000000000002.png
    ├── valid/
    │   ├── _annotations.coco.json
    │   ├── 000000000003.png
    │   └── 000000000004.png
    └── test/
        ├── _annotations.coco.json
        ├── 000000000005.png
        └── 000000000006.png


Conversion Modes
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - **detection**
     - Single-class object detection. All RF signals mapped to class ``signal`` for binary detection tasks.
   * - **recognition**
     - Multi-class recognition. Preserves original signal categories (e.g., WiFi, Bluetooth, LTE).
   * - **modulation**
     - Modulation-specific classification. Groups signals by modulation type (QPSK, OFDM, etc.).

CLI Usage
---------

.. code-block:: bash

    # Convert for signal detection (single class)
    python sigmf_to_coco.py \\
           --sigmf_dir ./sigmf_dataset \\
           --out_dir ./coco_detection \\
           --mode detection \\
           --train_split 0.8

    # Convert for signal recognition (multi-class)
    python sigmf_to_coco.py \\
           --sigmf_dir ./sigmf_dataset \\
           --out_dir ./coco_recognition \\
           --mode recognition \\
           --train_split 0.7

"""
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from copy import deepcopy
import json

from matplotlib import pyplot as plt
import numpy as np
from sigmf import SigMFFile, sigmffile
from tqdm import tqdm
from scipy.signal import get_window
from PIL import Image


def _get_capture_for_sample(
    sigmf_file: sigmffile.SigMFFile,
    sample_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the most relevant capture metadata for a given sample index from SigMF file.

    :param sigmf_file: SigMF file object containing capture metadata
    :type sigmf_file: sigmffile.SigMFFile
    :param sample_index: Sample index for which to find corresponding capture
    :type sample_index: int
    :returns: Capture dictionary with greatest sample_start ≤ sample_index, or None if no captures
    :rtype: Optional[Dict[str, Any]]
    """

    captures = sigmf_file.get_captures()
    if not captures:
        return None

    # Find the capture with the greatest sample_start that is ≤ sample_index
    chosen_capture = captures[0]
    chosen_start = chosen_capture.get(SigMFFile.START_INDEX_KEY, 0)

    for cap in captures[1:]:
        cap_start = cap.get(SigMFFile.START_INDEX_KEY, 0)
        if cap_start <= sample_index and cap_start > chosen_start:
            chosen_capture = cap
            chosen_start = cap_start

    return chosen_capture


def _generate_spectrogram(
    iq_data: np.ndarray,
    nfft: int,
    hop_length_div: int = 4,
    db_scale: bool = True,
    normalize: bool = True,
    invert: bool = False,
    resize: Optional[Tuple[int, int]] = (512, 512),
) -> np.ndarray:
    """
    Generates spectrogram from complex IQ data using STFT with post-processing options.

    :param iq_data: Input 1D array of complex IQ samples
    :type iq_data: np.ndarray
    :param nfft: FFT window size
    :type nfft: int
    :param hop_length_div: Divisor for hop length calculation
    :type hop_length_div: int
    :param db_scale: Whether to convert power spectrum to decibel scale
    :type db_scale: bool
    :param normalize: Whether to normalize spectrogram to [0, 1] before dB scaling
    :type normalize: bool
    :param invert: Whether to invert spectrogram values (1.0 - x)
    :type invert: bool
    :param resize: Target dimensions (height, width) for output spectrogram
    :type resize: Optional[Tuple[int, int]]
    :returns: 2D spectrogram array with values in [0, 1]
    :rtype: np.ndarray
    """
    # --- 1. STFT -------------------------------------------------------------
    hop_length = nfft // hop_length_div
    win = get_window("blackman", nfft, fftbins=True).astype(np.float32)

    n_frames = 1 + (len(iq_data) - nfft) // hop_length

    shape = (n_frames, nfft)
    strides = (iq_data.strides[0] * hop_length, iq_data.strides[0])
    frames = np.lib.stride_tricks.as_strided(iq_data, shape, strides)

    # Apply window → FFT → power-spectrum (power=2)
    windowed = frames * win
    spec = np.fft.fft(windowed, n=nfft, axis=1, norm=None).astype(np.complex64)
    x = (spec.real**2 + spec.imag**2).T  # → [freq, time] like torchaudio

    # --- 2. post-processing --------------------------------------------------
    if normalize:
        max_val = np.abs(x).max()
        if max_val < 1e-12:  # Avoid division by zero if max_val is zero or very small
            max_val = 1e-12
        x /= max_val

    # fftshift
    x = np.fft.fftshift(x, axes=0)

    if db_scale:
        # Use np.maximum to avoid log(0)
        x = 10.0 * np.log10(np.maximum(x, 1e-12))

    # min–max to [0,1]
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min) < 1e-12:  # Avoid division by zero if x is flat
        x = np.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min)

    if invert:
        x = 1.0 - x

    # --- 3. resize (bilinear, antialiased) -----------------------------------
    if resize is not None:
        # Pillow expects H×W, float32 in 0-255.  Convert, resize, convert back.
        img = Image.fromarray((x * 255.0).astype(np.float32), mode="F")
        img = img.resize(
            resize[::-1], resample=Image.Resampling.BILINEAR, reducing_gap=3.0
        )
        x = np.asarray(img, dtype=np.float32) / 255.0

    return x


def _convert_sigmf_ann_to_bbox(
    norm_ann: Dict[str, Any],
    freq_low_abs: float,
    freq_high_abs: float,
    center_freq: float,
    img_width: int,
    img_height: int,
    fs: float,
    frame_size: int,
) -> Optional[List[float]]:
    """
    Converts SigMF annotation to bounding box in pixel coordinates with frequency band clipping.

    :param norm_ann: SigMF annotation dictionary with START_INDEX_KEY and LENGTH_INDEX_KEY
    :type norm_ann: Dict[str, Any]
    :param freq_low_abs: Absolute lower frequency bound in Hz
    :type freq_low_abs: float
    :param freq_high_abs: Absolute upper frequency bound in Hz
    :type freq_high_abs: float
    :param center_freq: Center frequency of recording in Hz
    :type center_freq: float
    :param img_width: Target image width in pixels
    :type img_width: int
    :param img_height: Target image height in pixels
    :type img_height: int
    :param fs: Sampling frequency in Hz
    :type fs: float
    :param frame_size: Number of samples per frame for time normalization
    :type frame_size: int
    :returns: Bounding box as [x, y, w, h] in pixel coordinates, or None if outside visible band
    :rtype: Optional[List[float]]
    """

    # base-band offsets
    f_low = freq_low_abs - center_freq
    f_high = freq_high_abs - center_freq

    if f_low > f_high:
        # Swap if low > high
        f_low, f_high = f_high, f_low

    # intersection with the visible band
    band_low, band_high = -fs/2, +fs/2
    f_low_clipped = max(f_low,  band_low)
    f_high_clipped = min(f_high, band_high)

    # completely outside? → skip
    if f_low_clipped >= f_high_clipped:
        return None

    # convert to 0…1 normalised coordinates
    norm_low = (f_low_clipped + fs/2) / fs
    norm_high = (f_high_clipped + fs/2) / fs

    # time → pixels
    pixel_x = (norm_ann[SigMFFile.START_INDEX_KEY] / frame_size) * img_width
    pixel_w = (norm_ann[SigMFFile.LENGTH_INDEX_KEY] / frame_size) * img_width

    # frequency → pixels
    pixel_y = (1.0 - norm_high) * img_height
    pixel_h = (norm_high - norm_low) * img_height

    # final clipping
    pixel_x = max(0, min(pixel_x, img_width - 1))
    pixel_y = max(0, min(pixel_y, img_height - 1))
    pixel_w = max(1, min(pixel_w, img_width - pixel_x))
    pixel_h = max(1, min(pixel_h, img_height - pixel_y))

    return [float(pixel_x), float(pixel_y), float(pixel_w), float(pixel_h)]


def _get_annotations_in_frame(
    sigmf_annotations_list: List[dict],
    frame_sample_start: int,
    frame_sample_count: int
) -> List[dict]:
    """
    Filters SigMF annotations to include only those overlapping with specified frame.

    :param sigmf_annotations_list: List of all annotations from SigMF file
    :type sigmf_annotations_list: List[dict]
    :param frame_sample_start: Starting sample index of current frame
    :type frame_sample_start: int
    :param frame_sample_count: Total number of samples in current frame
    :type frame_sample_count: int
    :returns: List of annotation dictionaries that overlap with the frame
    :rtype: List[dict]
    """
    annotations_in_frame = []
    frame_end = frame_sample_start + frame_sample_count

    for ann in sigmf_annotations_list:
        ann_start_abs = ann.get(SigMFFile.START_INDEX_KEY)
        ann_count = ann.get(SigMFFile.LENGTH_INDEX_KEY)

        if ann_start_abs is None or ann_count is None:
            print(
                f"Warning: Annotation missing sample_start or sample_count: {ann.get(SigMFFile.LABEL_KEY, 'N/A')}")
            continue

        ann_end_abs = ann_start_abs + ann_count

        # Check for overlap: (StartA < EndB) and (EndA > StartB)
        if (ann_start_abs < frame_end) and (ann_end_abs > frame_sample_start):
            annotations_in_frame.append(deepcopy(ann))  # Return a copy
    return annotations_in_frame


def _normalize_annotation_to_frame(
    annotation: dict,
    frame_sample_start: int,
    frame_sample_count: int
) -> dict:
    """
    Adjusts annotation's time coordinates to be relative to specified frame with boundary clipping.

    :param annotation: Original annotation dictionary containing SigMF time fields
    :type annotation: dict
    :param frame_sample_start: Absolute sample index where frame begins
    :type frame_sample_start: int
    :param frame_sample_count: Number of samples in the frame
    :type frame_sample_count: int
    :returns: Annotation dictionary with normalized time fields relative to frame start, or None if no overlap
    :rtype: dict
    """
    norm_ann = deepcopy(annotation)
    original_ann_start_abs = norm_ann.get(SigMFFile.START_INDEX_KEY)
    original_ann_count = norm_ann.get(SigMFFile.LENGTH_INDEX_KEY)

    frame_end_abs = frame_sample_start + frame_sample_count

    # Clip annotation to frame boundaries
    effective_start_abs = max(original_ann_start_abs, frame_sample_start)
    effective_end_abs = min(original_ann_start_abs +
                            original_ann_count, frame_end_abs)

    if effective_start_abs >= effective_end_abs:  # No overlap or zero length after clipping
        return None

    norm_ann[SigMFFile.START_INDEX_KEY] = effective_start_abs - \
        frame_sample_start
    norm_ann[SigMFFile.LENGTH_INDEX_KEY] = effective_end_abs - \
        effective_start_abs

    return norm_ann


def _prepare_coco_image_entry(file_name: str, width: int, height: int) -> Dict[str, Any]:
    """
    Creates COCO image dictionary entry with basic metadata.

    :param file_name: Name of the image file
    :type file_name: str
    :param width: Image width in pixels
    :type width: int
    :param height: Image height in pixels
    :type height: int
    :returns: COCO-formatted image dictionary
    :rtype: Dict[str, Any]
    """
    return {
        "file_name": file_name, "width": width, "height": height,
    }


def _prepare_coco_annotation_entry(bbox: List[float], category_label_name: str, area: float) -> Dict[str, Any]:
    """
    Creates COCO annotation dictionary entry before ID assignment.

    :param bbox: Bounding box coordinates as [x, y, width, height]
    :type bbox: List[float]
    :param category_label_name: Category label name for the annotation
    :type category_label_name: str
    :param area: Area of the bounding box
    :type area: float
    :returns: COCO-formatted annotation dictionary without assigned IDs
    :rtype: Dict[str, Any]
    """
    return {
        "category_label_name": category_label_name, "bbox": bbox, "area": area, "iscrowd": 0
    }


def _setup_coco_directories(sigmf_input_dir_path: Path) -> Path:
    """
    Creates necessary directory structure for COCO dataset output.

    :param sigmf_input_dir_path: Input directory path containing SigMF files
    :type sigmf_input_dir_path: Path
    :returns: Base path for COCO output
    :rtype: Path
    :raises OSError: If directory creation fails
    """
    base_path = sigmf_input_dir_path / "coco"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def _compute_frame_sample_count(
    nfft: int,
    hop_div: int,
    img_width: int = 512
) -> int:
    """
    Computes number of IQ samples required to generate spectrogram with specified time bins.

    :param nfft: FFT size used for computing spectrogram
    :type nfft: int
    :param hop_div: Divisor used to calculate hop length
    :type hop_div: int
    :param img_width: Number of time bins in output spectrogram
    :type img_width: int
    :returns: Total number of IQ samples needed to produce img_width time bins
    :rtype: int
    """
    hop_length = nfft // hop_div
    return hop_length * img_width


def process_single_sigmf_file(
    sigmf_meta_path_str: str,
    frame_sample_count: int,
    frame_step_size: int,
    nfft: int,
    hop_length_div: int
):
    """
    Processes single SigMF file and converts it to COCO format with spectrogram generation.

    :param sigmf_meta_path_str: Path to SigMF metadata file
    :type sigmf_meta_path_str: str
    :param frame_sample_count: Number of samples per frame to process
    :type frame_sample_count: int
    :param frame_step_size: Step size between frames in samples
    :type frame_step_size: int
    :param nfft: Number of FFT points for spectrogram generation
    :type nfft: int
    :param hop_length_div: Hop length divisor for spectrogram generation
    :type hop_length_div: int
    :returns: Tuple of (COCO images list, COCO annotations list, category labels set, spectrograms)
    :rtype: tuple
    :raises FileNotFoundError: If SigMF metadata or data files cannot be found
    """
    sigmf_meta_path = Path(sigmf_meta_path_str)

    coco_images = []
    coco_annotations = []
    category_labels_found = set()
    spectrograms = []

    sigmf_file = sigmffile.fromfile(str(sigmf_meta_path), skip_checksum=True)
    original_file_name_stem = sigmf_meta_path.stem
    sample_rate = sigmf_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    all_annotations = sigmf_file.get_annotations()

    frame_idx_in_file = 0
    frame_range = range(0, sigmf_file.sample_count, frame_step_size)

    with tqdm(frame_range, desc=f"Processing {original_file_name_stem}",
              unit="frame", leave=False) as pbar:

        for frame_abs_start in pbar:
            # Skip last frame if it's not full
            if frame_abs_start + frame_sample_count > sigmf_file.sample_count:
                break
            frame_samples = sigmf_file.read_samples(
                start_index=frame_abs_start,
                count=frame_sample_count
            )

            spectrogram = _generate_spectrogram(
                frame_samples,
                nfft=nfft,
                hop_length_div=hop_length_div,
            )

            # Get dynamic image dimensions
            img_height, img_width = spectrogram.shape

            frame_image_file_name = f"{original_file_name_stem}_frame{frame_idx_in_file}.png"

            image_data = _prepare_coco_image_entry(
                frame_image_file_name, int(img_width), int(img_height))
            coco_images.append(image_data)

            annotations_in_frame = _get_annotations_in_frame(
                all_annotations, frame_abs_start, len(frame_samples))

            coco_annotations_in_frame = []
            for annotation in annotations_in_frame:
                annotation_start_abs = annotation[SigMFFile.START_INDEX_KEY]

                center_freq = _get_capture_for_sample(sigmf_file, annotation_start_abs)[
                    SigMFFile.FREQUENCY_KEY]

                label = annotation[SigMFFile.LABEL_KEY]
                freq_low = annotation[SigMFFile.FLO_KEY]
                freq_high = annotation[SigMFFile.FHI_KEY]

                norm_anno = _normalize_annotation_to_frame(
                    annotation, frame_abs_start, len(frame_samples))

                category_labels_found.add(label)
                bbox = _convert_sigmf_ann_to_bbox(
                    norm_anno, freq_low, freq_high, center_freq,
                    int(img_width), int(
                        img_height), sample_rate, len(frame_samples)
                )

                if bbox is None:
                    continue

                annotation_data = _prepare_coco_annotation_entry(
                    bbox, label, bbox[2] * bbox[3])
                coco_annotations_in_frame.append(annotation_data)

            coco_annotations.append(coco_annotations_in_frame)
            spectrograms.append(spectrogram)
            frame_idx_in_file += 1

            # Update progress bar with stats
            pbar.set_postfix({
                'frames': frame_idx_in_file,
                'annotations': sum(len(ann_list) for ann_list in coco_annotations)
            })

    return coco_images, coco_annotations, category_labels_found, spectrograms


def _initialize_coco_output(mode: Literal["detection", "recognition"]) -> Tuple[Dict, Dict, Dict, Dict, int]:
    """
    Initializes COCO format output structures for training, validation and test datasets.

    :param mode: Operating mode for category mapping setup
    :type mode: Literal["detection", "recognition"]
    :returns: Tuple of (train_coco_dict, val_coco_dict, test_coco_dict, category_map, next_category_id)
    :rtype: Tuple[Dict, Dict, Dict, Dict, int]
    :raises ValueError: If mode is not "detection" or "recognition"
    """

    coco_template = {
        "info": {},
        "categories": [],
        "images": [],
        "annotations": []
    }

    coco_output_train = deepcopy(coco_template)
    coco_output_val = deepcopy(coco_template)
    coco_output_test = deepcopy(coco_template)

    # Initialize category mapping based on mode - START AT 0
    if mode == "detection":
        global_category_map = {"signal": 0}  # Changed from 1 to 0
        next_global_category_id = 1
    elif mode == "recognition":
        global_category_map = {}
        next_global_category_id = 0  # Start at 0 for recognition mode too
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Choose 'detection' or 'recognition'.")

    return coco_output_train, coco_output_val, coco_output_test, global_category_map, next_global_category_id


def _assign_category_id(
        mode: Literal["detection", "recognition"],
        original_label: str,
        global_category_map: Dict[str, int]
) -> Optional[int]:
    """
    Assigns category ID based on operating mode and original label mapping.

    :param mode: Operating mode determining category assignment strategy
    :type mode: Literal["detection", "recognition"]
    :param original_label: Original label string to map to category ID
    :type original_label: str
    :param global_category_map: Dictionary mapping category names to IDs
    :type global_category_map: Dict[str, int]
    :returns: Category ID if valid mapping found, None otherwise
    :rtype: Optional[int]
    """
    if mode == "detection":
        return global_category_map.get("signal")
    elif mode == "recognition":
        if original_label and original_label in global_category_map:
            return global_category_map[original_label]
    return None


def _process_annotations_for_split(
    image_data_list: List[Dict],
    annotations_for_images_list: List[List[Dict]],
    spectrograms_list: List[np.ndarray],
    indices: List[int],
    coco_output: Dict,
    split_dir: Path,
    global_category_map: Dict[str, int],
    mode: Literal["detection", "recognition"],
    image_id_counter: int,
    annotation_id_counter: int
) -> Tuple[int, int]:
    """
    Processes annotations for specific dataset split with ID assignment and category mapping.

    :param image_data_list: List of image metadata dictionaries
    :type image_data_list: List[Dict]
    :param annotations_for_images_list: List of annotation lists per image
    :type annotations_for_images_list: List[List[Dict]]
    :param spectrograms_list: List of spectrogram arrays
    :type spectrograms_list: List[np.ndarray]
    :param indices: List of indices for this split
    :type indices: List[int]
    :param coco_output: COCO output dictionary to populate
    :type coco_output: Dict
    :param split_dir: Directory for this split
    :type split_dir: Path
    :param global_category_map: Mapping from category names to IDs
    :type global_category_map: Dict[str, int]
    :param mode: Processing mode for category assignment
    :type mode: Literal["detection", "recognition"]
    :param image_id_counter: Current image ID counter
    :type image_id_counter: int
    :param annotation_id_counter: Current annotation ID counter
    :type annotation_id_counter: int
    :returns: Updated (image_id_counter, annotation_id_counter)
    :rtype: Tuple[int, int]
    """

    # Create split directory
    split_dir.mkdir(parents=True, exist_ok=True)

    for item_idx in indices:
        # Process image
        image_data = deepcopy(image_data_list[item_idx])
        image_data["id"] = image_id_counter
        coco_output["images"].append(image_data)

        # Save spectrogram image
        spectrogram = spectrograms_list[item_idx]
        image_path = split_dir / image_data["file_name"]
        plt.imsave(str(image_path), spectrogram, cmap='gray')

        # Process annotations for this image
        if item_idx < len(annotations_for_images_list):
            annotations_for_this_image = annotations_for_images_list[item_idx]

            for ann_data in annotations_for_this_image:
                ann_data_copy = deepcopy(ann_data)
                ann_data_copy["id"] = annotation_id_counter
                ann_data_copy["image_id"] = image_data["id"]

                # Extract and assign category
                original_label_name = ann_data_copy.pop(
                    "category_label_name", None)
                assigned_category_id = _assign_category_id(
                    mode, original_label_name, global_category_map)

                if assigned_category_id is not None:
                    ann_data_copy["category_id"] = assigned_category_id
                    coco_output["annotations"].append(ann_data_copy)

                annotation_id_counter += 1

        image_id_counter += 1

    return image_id_counter, annotation_id_counter


def _split_and_process_file_data(
    image_data_list: List[Dict],
    annotations_for_images_list: List[List[Dict]],
    spectrograms_list: List[np.ndarray],
    train_split_ratio: float,
    val_test_split_ratio: float,
    base_path: Path,
    coco_output_train: Dict,
    coco_output_val: Dict,
    coco_output_test: Dict,
    global_category_map: Dict[str, int],
    mode: Literal["detection", "recognition"],
    image_id_counter: int,
    annotation_id_counter: int,
    include_test: bool = True
) -> Tuple[int, int]:
    """
    Splits file data into train/validation/test sets and processes annotations for all splits.

    :param image_data_list: List of image metadata dictionaries to split
    :type image_data_list: List[Dict]
    :param annotations_for_images_list: List of annotation lists per image
    :type annotations_for_images_list: List[List[Dict]]
    :param spectrograms_list: List of spectrogram arrays
    :type spectrograms_list: List[np.ndarray]
    :param train_split_ratio: Fraction of data for training set
    :type train_split_ratio: float
    :param val_test_split_ratio: Fraction of remaining data for validation (rest goes to test)
    :type val_test_split_ratio: float
    :param base_path: Base directory path for COCO output
    :type base_path: Path
    :param coco_output_train: COCO dictionary for training data
    :type coco_output_train: Dict
    :param coco_output_val: COCO dictionary for validation data  
    :type coco_output_val: Dict
    :param coco_output_test: COCO dictionary for test data
    :type coco_output_test: Dict
    :param global_category_map: Mapping from category names to IDs
    :type global_category_map: Dict[str, int]
    :param mode: Processing mode for category assignment
    :type mode: Literal["detection", "recognition"]
    :param image_id_counter: Current image ID counter
    :type image_id_counter: int
    :param annotation_id_counter: Current annotation ID counter
    :type annotation_id_counter: int
    :param include_test: Whether to create test split
    :type include_test: bool
    :returns: Updated (image_id_counter, annotation_id_counter) after processing all splits
    :rtype: Tuple[int, int]
    """

    num_items = len(image_data_list)
    if num_items == 0:
        return image_id_counter, annotation_id_counter

    # Split indices
    item_indices = list(range(num_items))
    random.shuffle(item_indices)

    train_split_point = int(num_items * train_split_ratio)
    train_indices = item_indices[:train_split_point]
    remaining_indices = item_indices[train_split_point:]

    if include_test and len(remaining_indices) > 0:
        # Split remaining data between validation and test
        val_split_point = int(len(remaining_indices) * val_test_split_ratio)
        val_indices = remaining_indices[:val_split_point]
        test_indices = remaining_indices[val_split_point:]
    else:
        # All remaining data goes to validation
        val_indices = remaining_indices
        test_indices = []

    # Process training split
    train_dir = base_path / "train"
    image_id_counter, annotation_id_counter = _process_annotations_for_split(
        image_data_list, annotations_for_images_list, spectrograms_list, train_indices,
        coco_output_train, train_dir, global_category_map, mode,
        image_id_counter, annotation_id_counter
    )

    # Process validation split
    val_dir = base_path / "valid"
    image_id_counter, annotation_id_counter = _process_annotations_for_split(
        image_data_list, annotations_for_images_list, spectrograms_list, val_indices,
        coco_output_val, val_dir, global_category_map, mode,
        image_id_counter, annotation_id_counter
    )

    # Process test split if requested
    if include_test and len(test_indices) > 0:
        test_dir = base_path / "test"
        image_id_counter, annotation_id_counter = _process_annotations_for_split(
            image_data_list, annotations_for_images_list, spectrograms_list, test_indices,
            coco_output_test, test_dir, global_category_map, mode,
            image_id_counter, annotation_id_counter
        )

    return image_id_counter, annotation_id_counter


def _update_categories(mode: str, category_labels: set, global_category_map: Dict[str, int], next_id: int) -> int:
    """
    Updates global category mapping with new category labels for recognition mode.

    :param mode: Operation mode, only processes when mode is "recognition"
    :type mode: str
    :param category_labels: Set of category labels to add to mapping
    :type category_labels: set
    :param global_category_map: Dictionary mapping category labels to IDs
    :type global_category_map: Dict[str, int]
    :param next_id: Next available ID to assign to new categories
    :type next_id: int
    :returns: Updated next available ID after processing new categories
    :rtype: int
    """
    if mode == "recognition":
        for label in category_labels:
            if label not in global_category_map:
                global_category_map[label] = next_id
                next_id += 1
    return next_id


def _finalize_coco_output(
    coco_output_train: Dict,
    coco_output_val: Dict,
    coco_output_test: Dict,
    global_category_map: Dict[str, int],
    base_path: Path,
    include_test: bool = True
) -> None:
    """
    Finalizes and saves COCO format output files for training, validation and test datasets.

    :param coco_output_train: COCO format dictionary for training data
    :type coco_output_train: Dict
    :param coco_output_val: COCO format dictionary for validation data
    :type coco_output_val: Dict
    :param coco_output_test: COCO format dictionary for test data
    :type coco_output_test: Dict
    :param global_category_map: Mapping from category labels to IDs
    :type global_category_map: Dict[str, int]
    :param base_path: Base directory path for COCO output
    :type base_path: Path
    :param include_test: Whether test split was created
    :type include_test: bool
    :raises IOError: If JSON files cannot be written
    """

    # Build categories list (starting from 0)
    master_categories_list = [
        {"id": cat_id, "name": label, "supercategory": "signal"}
        for label, cat_id in global_category_map.items()
    ]

    coco_output_train["categories"] = master_categories_list
    coco_output_val["categories"] = master_categories_list
    if include_test:
        coco_output_test["categories"] = master_categories_list

    # Save files in split directories
    train_dir = base_path / "train"
    val_dir = base_path / "valid"

    train_json_path = train_dir / "_annotations.coco.json"
    val_json_path = val_dir / "_annotations.coco.json"

    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output_train, f, indent=4)
    print(f"COCO Train JSON saved to: {train_json_path}")

    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output_val, f, indent=4)
    print(f"COCO Val JSON saved to: {val_json_path}")

    if include_test and len(coco_output_test["images"]) > 0:
        test_dir = base_path / "test"
        test_json_path = test_dir / "_annotations.coco.json"
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_output_test, f, indent=4)
        print(f"COCO Test JSON saved to: {test_json_path}")

    print(f"Images saved in split directories under: {base_path}")


def sigmf_dir_to_coco(
    sigmf_input_dir: str,
    frame_overlap_ratio: float = 0.5,
    nfft: int = 512,
    hop_length_div: int = 4,
    train_split_ratio: float = 0.8,
    val_test_split_ratio: float = 0.5,
    mode: Literal["detection", "recognition"] = "detection",
    include_test: bool = True,
):
    """
    Converts directory of SigMF files to COCO format for object detection or recognition tasks.

    :param sigmf_input_dir: Path to directory containing SigMF files
    :type sigmf_input_dir: str
    :param frame_overlap_ratio: Overlap ratio between consecutive frames
    :type frame_overlap_ratio: float
    :param nfft: Number of FFT points for spectrogram generation
    :type nfft: int
    :param hop_length_div: Divisor for hop length calculation
    :type hop_length_div: int
    :param train_split_ratio: Ratio of data for training vs (validation + test)
    :type train_split_ratio: float
    :param val_test_split_ratio: Ratio of remaining data for validation (rest goes to test)
    :type val_test_split_ratio: float
    :param mode: Processing mode determining annotation format
    :type mode: Literal["detection", "recognition"]
    :param include_test: Whether to create test split
    :type include_test: bool
    :raises FileNotFoundError: If no SigMF files found in input directory
    :raises ValueError: If parameters are outside valid ranges
    """
    sigmf_input_dir_path = Path(sigmf_input_dir)
    base_path = _setup_coco_directories(sigmf_input_dir_path)

    # Initialize COCO structures
    coco_output_train, coco_output_val, coco_output_test, global_category_map, next_global_category_id = _initialize_coco_output(
        mode)

    # Initialize counters
    global_coco_image_id_counter = 1
    global_coco_annotation_id_counter = 1

    # Get SigMF files
    sigmf_meta_files = list(sigmf_input_dir_path.glob('*.sigmf-meta'))
    if not sigmf_meta_files:
        print(f"No .sigmf-meta files found in {sigmf_input_dir_path}")
        return

    # Calculate frame parameters
    frame_sample_count = _compute_frame_sample_count(nfft, hop_length_div)
    frame_step_size = int(frame_sample_count * (1 - frame_overlap_ratio))

    # Process each SigMF file
    for sigmf_meta_path_obj in tqdm(sigmf_meta_files, desc="Processing SigMF files"):
        # Process single file
        image_data_list, annotations_for_images_list, category_labels, spectrograms_list = process_single_sigmf_file(
            str(sigmf_meta_path_obj),
            frame_sample_count,
            frame_step_size,
            nfft,
            hop_length_div,
        )

        # Update categories
        next_global_category_id = _update_categories(
            mode, category_labels, global_category_map, next_global_category_id)

        # Split and process data
        global_coco_image_id_counter, global_coco_annotation_id_counter = _split_and_process_file_data(
            image_data_list,
            annotations_for_images_list,
            spectrograms_list,
            train_split_ratio,
            val_test_split_ratio,
            base_path,
            coco_output_train,
            coco_output_val,
            coco_output_test,
            global_category_map,
            mode,
            global_coco_image_id_counter,
            global_coco_annotation_id_counter,
            include_test
        )

    # Finalize output
    _finalize_coco_output(
        coco_output_train,
        coco_output_val,
        coco_output_test,
        global_category_map,
        base_path,
        include_test
    )
