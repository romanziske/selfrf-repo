from functools import partial
import warnings
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from selfrf.finetuning.detection.detectron2.vitdet.coco_loader_lsj import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# dont build mask head
model.roi_heads.mask_head = None
model.roi_heads.mask_pooler = None
model.roi_heads.mask_in_features = None     # or []

# Set number of classes and input channels
model.roi_heads.num_classes = 10
model.pixel_mean = [0.0]
model.pixel_std = [1.0]
model.backbone.net.in_chans = 1


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)

train.max_iter = 250000  # 10 epchs
train.eval_period = 25000
train.checkpointer.period = 25000
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

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*",
    category=FutureWarning
)
