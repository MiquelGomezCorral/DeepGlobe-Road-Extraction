"""Train model script.

Script to train a model and save it.
"""
import time

import pytorch_lightning as pl
from maikol_utils.print_utils import print_log, print_separator
from maikol_utils.time_tracker import print_time
from pytorch_lightning.loggers import CSVLogger
from src.config.config import Configuration
from src.models import RoadSegmentationModel
from src.script_refactor import get_data_loaders, visualize_model_predictions
from src.utils import get_device


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
    logger = CSVLogger(CONFIG.log_folder, name=CONFIG.model_name)

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
    model = RoadSegmentationModel(CONFIG)
    model = model.to(get_device())

    t0 = time.time()
    trainer = pl.Trainer(
        max_epochs=CONFIG.epochs,
        log_every_n_steps=1,
        logger=logger,
        accelerator="auto",
        devices="auto",
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    t1 = time.time()
    print_time(sec=t1 - t0, files=len(train_dataloader), prefix=" - Training time")

    print_log(f" - Saving model at {CONFIG.model_path}...")
    trainer.save_checkpoint(CONFIG.model_path)

    # ====================================================================
    #                             VALIDATE MODEL
    # ====================================================================
    print_separator("TESTING MODEL", sep_type="LONG")
    test_metrics = trainer.validate(model, dataloaders=test_dataloader, verbose=False)
    print_log(f" - Test metrics: {test_metrics[0]}")

    visualize_model_predictions(CONFIG, model, test_dataloader)

    print_separator("DONE!", sep_type="START")
