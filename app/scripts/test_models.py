"""Test all models.

Script to test all the models.
"""
import os

from maikol_utils.file_utils import list_dir_files, save_json
from maikol_utils.print_utils import print_separator
from src.config import Configuration
from src.models import (
    RoadSegmentationDataset,
    RoadSegmentationModel,
    RoadSegmentationModelVIT,
    test_model,
)
from src.utils import PathParser
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_all_models(CONFIG: Configuration):
    """Load and test all the models"""
    # ============================================================================
    #                               LOAD MODELS
    # ============================================================================
    print_separator("LOADING MODELS", sep_type="LONG")
    files, n = list_dir_files(CONFIG.LOGS_FOLDER, recursive=True)
    models = [f for f in files if "checkpoints" in f]
    model_obj = [PathParser(m) for m in models]

    vit_models = [mod for mod in model_obj if "ViT" == mod.arc]
    cnn_models = [mod for mod in model_obj if "ViT" != mod.arc]
    vit_models_scores = []
    cnn_models_scores = []

    print_separator("TESTING MODELS", sep_type="LONG")
    print_separator("TESTING ViT MODELS")
    n_cpu = max(os.cpu_count() // 2, 1)
    for model in tqdm(vit_models):
        #  =================== LOAD MODEL ===================
        model = RoadSegmentationModelVIT.load_from_checkpoint(
            checkpoint_path=model.path, CONFIG=model.get_config()
        )

        #  =================== LOAD DATASET ===================
        test_dataset = RoadSegmentationDataset(CONFIG.test_folder, CONFIG, verbose=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=n_cpu
        )

        #  =================== TEST MODEL ===================
        score = test_model(model, test_dataloader)
        vit_models_scores.append((model, score))

    print_separator("TESTING CNN MODELS")
    for model in tqdm(cnn_models):
        #  =================== LOAD MODEL ===================
        model = RoadSegmentationModel.load_from_checkpoint(
            checkpoint_path=model.path, CONFIG=model.get_config()
        )

        #  =================== LOAD DATASET ===================
        test_dataset = RoadSegmentationDataset(CONFIG.test_folder, CONFIG, verbose=False)
        test_dataloader = DataLoader(
            test_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=n_cpu
        )

        #  =================== TEST MODEL ===================
        score = test_model(model, test_dataloader)
        cnn_models_scores.append((model, score))

    # ============================================================================
    #                               SAVE SCORES
    # ============================================================================
    print_separator("SAVING SCORES", sep_type="LONG")
    vit_scores = {
        mod_obs.name: (mod_obs.path, score)
        for mod_obs, (mod, score) in zip(vit_models, vit_models_scores)
    }
    cnn_scores = {
        mod_obs.name: (mod_obs.path, score)
        for mod_obs, (mod, score) in zip(cnn_models, cnn_models_scores)
    }
    save_json(CONFIG.vit_model_scores, vit_scores)
    save_json(CONFIG.cnn_model_scores, cnn_scores)
