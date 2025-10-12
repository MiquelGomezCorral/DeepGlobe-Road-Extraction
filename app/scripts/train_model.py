"""Train model script.

Script to train a model and save it.
"""
import time

import pytorch_lightning as pl
from maikol_utils.print_utils import print_log, print_separator
from maikol_utils.time_tracker import print_time
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from src.config.config import Configuration
from src.models import RoadSegmentationModel, RoadSegmentationModelVIT
from src.script_refactor import get_data_loaders, visualize_model_predictions
from src.utils import get_device


class VisualizePredictionsCallback(Callback):
    """Callback to visualize model predictions at the end of each epoch."""

    def __init__(self, CONFIG, test_dataloader):
        super().__init__()
        self.CONFIG = CONFIG
        self.test_dataloader = test_dataloader

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize model predictions at the end of each epoch."""
        visualize_model_predictions(self.CONFIG, pl_module, self.test_dataloader)


early_stop = EarlyStopping(
    monitor="val_loss",  # metric to watch
    patience=Configuration.patience,  # stop if no improvement after 3 checks
    mode="min",
)


def train_model(CONFIG: Configuration):
    """Train a model and save it to disk.

    Args:
        CONFIG (Configuration): Configuration.
    """
    # ====================================================================
    #                             CONFIGURATION
    # ====================================================================
    print_separator("TRAINING MODEL", sep_type="START")
    CONFIG.print_config()
    logger = CSVLogger(CONFIG.LOGS_FOLDER, name=CONFIG.model_name)

    # ====================================================================
    #                             DATASET
    # ====================================================================
    print_separator("GETTING DATASET", sep_type="LONG")
    train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(CONFIG)

    # ====================================================================
    #                             TRAIN MODEL
    # ====================================================================
    print_separator("TRAINING MODEL", sep_type="LONG")
    print_log(" - Getting model...")
    if CONFIG.architecture == "ViT":
        model = RoadSegmentationModelVIT(CONFIG)
    else:
        model = RoadSegmentationModel(CONFIG)
    model = model.to(get_device())

    trainer = pl.Trainer(
        max_epochs=CONFIG.epochs,
        log_every_n_steps=1,
        logger=logger,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[early_stop, VisualizePredictionsCallback(CONFIG, test_dataloader)],
    )

    # Resume from checkpoint if exists
    # ckpt_path = CONFIG.model_path
    # if os.path.exists(ckpt_path):
    #     print(f"⏯ Resuming from checkpoint: {ckpt_path}")
    # else:
    #     ckpt_path = None
    #     print("▶ No checkpoint found, starting from scratch.")

    # ========================== ACTUAL TRAINING =====================================
    t0 = time.time()
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
        # ckpt_path=CONFIG.model_path,
    )
    t1 = time.time()
    print_time(sec=t1 - t0, n_files=len(train_dataloader), prefix=" - Training time")

    print_log(f" - Saving model at {CONFIG.model_path}...")
    trainer.save_checkpoint(CONFIG.model_path)

    # ====================================================================
    #                             VALIDATE MODEL
    # ====================================================================
    print_separator("TESTING MODEL", sep_type="LONG")
    test_metrics = trainer.validate(model, dataloaders=test_dataloader, verbose=False)
    print_log(f" - Test metrics: {test_metrics}")
    visualize_model_predictions(CONFIG, model, test_dataloader)

    with open(CONFIG.test_metrics_file, "w") as f:
        print(f" - Test metrics: {test_metrics}", file=f)
        print_time(
            sec=t1 - t0, n_files=len(train_dataloader), prefix=" - Training time", out_file=f
        )

    print_separator("DONE!", sep_type="START")
