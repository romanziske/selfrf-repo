from functools import partial
import warnings
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from selfrf.finetuning.detection.detectron2.vitdet.coco_loader_lsj import dataloader

# -------------------------------------------------------------------
# Build ViTDet with a ViT-Small /16 backbone (embed_dim=384, heads=6)
# -------------------------------------------------------------------
model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.backbone.net.embed_dim = 384
model.backbone.net.num_heads = 6             # 6 × 64 = 384
# depth remains 12; patch_size remains 16
model.backbone.net.drop_path_rate = 0.3      # slightly lower than ViT-B default
model.backbone.net.in_chans = 1              # grayscale spectrogram input

# -------------------------------------------------------------------
# ROI heads: class-only (no masks) for 10 signal classes
# -------------------------------------------------------------------
model.roi_heads.mask_head = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_in_features = None
model.roi_heads.num_classes = 10

# -------------------------------------------------------------------
# Input scaling: loader yields 0‒255 floats; model divides by 255
# -------------------------------------------------------------------
model.pixel_mean = [0.0]     # length must match in_chans
model.pixel_std = [1.0]


# -------------------------------------------------------------------
# Training setup
# -------------------------------------------------------------------
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    ""
)

# -------------------------------------------------------------------
# LR schedule: warm-up + multi-step
# -------------------------------------------------------------------
train.max_iter = 250_000
milestones = [train.max_iter * 0.9, train.max_iter * 0.96]

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=milestones,
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# -------------------------------------------------------------------
# Optimiser: AdamW with layer-wise LR decay (12 ViT blocks)
# -------------------------------------------------------------------
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# -------------------------------------------------------------------
# Suppress AMP matmul precision warning
# -------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning,
)
