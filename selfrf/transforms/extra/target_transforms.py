"""
Custom target transformations for RF signal metadata processing in object detection pipelines.

This module extends TorchSig's target transform framework with RF-specific metadata transformations required for object detection and classification tasks. It provides transforms to convert RF signal annotations into standardized formats like bounding boxes, class labels, and family names that are compatible with computer vision frameworks. The transforms handle the conversion between RF domain concepts (frequency bands, time windows, signal types) and machine learning labels (normalized coordinates, class indices). These transforms are essential for bridging the gap between RF signal datasets and detection models like YOLO or Detectron2. The module integrates with the broader selfRF pipeline by providing the target preprocessing required for supervised fine-tuning of detection models on RF spectrograms.
"""
from torchsig.transforms.target_transforms import TargetTransform


class Identity(TargetTransform):
    """
    Pass-through transform that returns metadata unchanged for testing and debugging purposes.

    Useful as a placeholder or for maintaining consistency in transform pipelines without modification.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, metadata):
        """
        Returns input metadata without any modifications.

        :param metadata: Input metadata dictionary to pass through unchanged
        :returns: The same metadata dictionary without modifications
        :rtype: Any
        """
        return metadata


class BBOXLabel(TargetTransform):
    """
    Converts RF signal temporal and spectral boundaries into normalized bounding box coordinates.

    Transforms signal start time, duration, bandwidth, and center frequency into XYWH format suitable for object detection models.

    :param kwargs: Additional keyword arguments passed to parent TargetTransform
    """

    output_list = ["list", "dict"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Include "duration" since it's used in __apply__
        self.required_metadata = [
            "class_index", "start", "duration", "bandwidth", "center_freq", "sample_rate"]
        self.targets_metadata = ["bbox"]

    def __apply__(self, metadata):
        """
        Computes normalized bounding box from RF signal parameters and adds to metadata.

        :param metadata: Dictionary containing RF signal parameters including start, duration, bandwidth, center_freq, and sample_rate
        :type metadata: dict
        :returns: Updated metadata dictionary with added "bbox" field containing (x_min, y_min, width, height) tuple
        :rtype: dict
        """
        # Extract required metadata
        # X_min is the starting time
        x_min = metadata["start"]
        # Width is the duration, normalized
        width = metadata["duration"]
        # Height is bandwidth normalized by sample rate
        height = metadata["bandwidth"] / metadata["sample_rate"]
        # Compute y_center as in YOLO for consistency
        y_center = 1 - ((metadata["sample_rate"] / 2.0) +
                        metadata["center_freq"]) / metadata["sample_rate"]
        # Y_min is the top edge: y_center - height / 2
        y_min = y_center - height / 2
        # Create the XYHW label tuple
        xyhw_label = (x_min, y_min, width, height)
        # Add to metadata
        metadata["bbox"] = xyhw_label

        return metadata


class ConstantSignalName(TargetTransform):
    """
    Adds a fixed signal name to metadata for single-class detection tasks.

    Useful when all RF signals should be labeled with the same class name regardless of their actual type.

    :param signal_name: The constant signal name to assign to all samples
    :type signal_name: str
    :param kwargs: Additional keyword arguments passed to parent TargetTransform
    """

    def __init__(self, signal_name: str, **kwargs):
        super().__init__(**kwargs)
        self.signal_name = signal_name
        self.targets_metadata = ["const_signal_name"]

    def __apply__(self, metadata):
        """
        Adds the constant signal name to the metadata dictionary.

        :param metadata: Input metadata dictionary to modify
        :type metadata: dict
        :returns: Updated metadata with "const_signal_name" field set to the configured value
        :rtype: dict
        """
        metadata["const_signal_name"] = self.signal_name
        return metadata


class ConstantSignalIndex(TargetTransform):
    """
    Assigns a fixed class index to all signals for single-class detection scenarios.

    Commonly used in binary detection tasks where all signals are treated as the same class.

    :param index: The constant class index to assign to all samples
    :type index: int
    :param kwargs: Additional keyword arguments passed to parent TargetTransform
    """

    def __init__(self, index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.targets_metadata = ["const_signal_index"]
        self.index = index

    def __apply__(self, metadata):
        """
        Adds the constant signal index to the metadata dictionary.

        :param metadata: Input metadata dictionary to modify
        :type metadata: dict
        :returns: Updated metadata with "const_signal_index" field set to the configured value
        :rtype: dict
        """
        metadata["const_signal_index"] = self.index
        return metadata


class ConstantFamilyName(TargetTransform):
    """
    Assigns a fixed family name to all signals for hierarchical classification tasks.

    Used when grouping all signals under a common super-category regardless of their specific type.

    :param family_name: The constant family name to assign to all samples
    :type family_name: str
    :param kwargs: Additional keyword arguments passed to parent TargetTransform
    """

    def __init__(self, family_name: str, **kwargs):
        super().__init__(**kwargs)
        self.family_name = family_name
        self.targets_metadata = ["const_family_name"]

    def __apply__(self, metadata):
        """
        Adds the constant family name to the metadata dictionary.

        :param metadata: Input metadata dictionary to modify
        :type metadata: dict
        :returns: Updated metadata with "const_family_name" field set to the configured value
        :rtype: dict
        """
        metadata["const_family_name"] = self.family_name
        return metadata
