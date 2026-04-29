from pytorch_lightning.callbacks import LearningRateMonitor
from core.configs import builds, MainStore


MainStore.store(name="pl", node=builds(LearningRateMonitor), group="callbacks/lr_monitor")
