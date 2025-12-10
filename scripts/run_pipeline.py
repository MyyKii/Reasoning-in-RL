import mlp_data_collector

import logging
from pathlib import Path

def setup_logging(log_level: int = logging.INFO, log_to_file: bool = False) -> None:
    """
    Richtet das Python-Logging ein.
    - Gibt alle messages >= log_level auf der Konsole aus.
    - Optional zusätzlich in eine Log-Datei.
    """
    #log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_format = "%(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_to_file:
        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler("logs/pipeline.log", mode="a", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )


def start_pipeline():
    # ------------------------------ MLP Data Collection ------------------------------
    logging.info("Starting the counterfactual agent training pipeline...")
    logging.info("Step 1: Collecting Random Samples from Mujoco environment for MLP...")
    mlp_data_collector.run_mlp_data_collection()
    logging.info(f' Data saved in {mlp_data_collector.DATA_PATH}')
    # ------------------------------ MLP Model ----------------------------------------
    logging.info("Step 2: Training MLP Model...")
    import mlp_model
    mlp_model.train_mlp_model()
    logging.info(f' MLP Model saved in {mlp_model.MLP_MODEL_PATH}')
    

if __name__ == "__main__":
    setup_logging(log_level=logging.INFO, log_to_file=False)
    start_pipeline()