"""
Export Detectron2 model to ONNX format.
"""
import logging

import torch


logger = logging.getLogger("detectron2")


def export_onnx_model(cfg, model, export_path: str, input_shape: tuple = (1, 1, 512, 512)):
    """
    Exports Detectron2 model to ONNX format using TracingAdapter.

    :param cfg: Detectron2 configuration object
    :type cfg: detectron2.config.CfgNode
    :param model: Trained Detectron2 model
    :type model: torch.nn.Module
    :param export_path: Path to save ONNX model
    :type export_path: str
    :param input_shape: Input tensor shape (batch_size, channels, height, width)
    :type input_shape: tuple
    :raises RuntimeError: If ONNX export or tracing fails
    """
    # Set model to evaluation mode
    model.eval()

    # Create dummy input in the format Detectron2 expects
    batch_size, channels, height, width = input_shape
    device = next(model.parameters()).device

    from detectron2.export import TracingAdapter
    from detectron2.modeling.meta_arch import GeneralizedRCNN

    # Create single image tensor (not batched for TracingAdapter)
    image = torch.randn(channels, height, width).to(device)

    # Create input in Detectron2 format
    inputs = [{"image": image}]  # Remove other unused keys, keep only image

    # Define inference function for GeneralizedRCNN (like Faster R-CNN)
    if isinstance(model, GeneralizedRCNN):
        def inference(model, inputs):
            inst = model.inference(inputs, do_postprocess=True)[0]
            return [{"instances": inst}]
    else:
        inference = None  # Assume that we just call the model directly

    # Create TracingAdapter with the custom inference function
    traceable_model = TracingAdapter(model, inputs, inference)

    # Export to ONNX - pass only the image tensor, not the full input dict
    torch.onnx.export(
        traceable_model,
        (image,),  # Pass image tensor as tuple
        export_path,
        export_params=True,
        opset_version=16,  # Use stable ONNX opset version
        do_constant_folding=True,
        input_names=['image'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch_size'} if batch_size > 1 else {},
            'output': {0: 'batch_size'} if batch_size > 1 else {}
        }
    )

    logger.info(f"Model successfully exported to ONNX: {export_path}")
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))
