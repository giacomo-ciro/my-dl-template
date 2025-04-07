import json
import os
import sys
import time

import lightning as pl
import wandb
from datamodule import myDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from model import myModel
from utils import check_config

# Load config.json file
if len(sys.argv) == 2:
    CONFIG_PATH = sys.argv[1]
else:
    print("No path/to/config.json provided, defaulting to './config.json'")
    CONFIG_PATH = "./config.json"

# Configuration file
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Check validity
check_config(config)

# Initalize the save folder (and save current config)
dirpath = config["save_dir"] + time.strftime("ymd_%y%m%d_HMS_%H_%M_%S")
os.mkdir(dirpath)
with open(os.path.join(dirpath, "config.json"), "w") as f:
    json.dump(config, f)

# Setup logging
if config["wandb"]:
    wandb.login()
    run = wandb.init(
        project=config["project_name"],
        config=config,
        name=config["run_name"] if config["run_name"] else None,
    )
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/ce", summary="min", step_metric="epoch")
    logger = WandbLogger(project=config["project_name"])
else:
    logger = CSVLogger(save_dir=".")

# Load the datamodule
datamodule = myDataModule(CONFIG_PATH)

# Instantiate a model
model = myModel(CONFIG_PATH)

# Checkpointers
checkpointer = ModelCheckpoint(
    dirpath=dirpath,  # Directory to save checkpoints
    filename="epoch_{epoch}_ce_{valid/ce:.2f}",  # Checkpoint filename format
    save_top_k=-1,  # Save the 3 best models
    monitor="valid/ce",  # Metric to monitor
    mode="min",  # Mode ('min' for loss, 'max' for accuracy)
    auto_insert_metric_name=False,  # To avoid issues when "/" in metric name
)

# Early Stopping
early_stopping = EarlyStopping(
    monitor="valid/ce",  # Monitor validation cross-entropy loss
    patience=2,  # Number of validation checks with no improvement after which training will stop
    min_delta=0.001,  # Minimum change in monitored value to qualify as improvement
    mode="min",  # We want to minimize the loss
    verbose=True,  # Print message when early stopping is triggered
    check_on_train_epoch_end=False,  # Check at validation time, not training end
)

# Init the trainer
trainer = pl.Trainer(
    max_epochs=config["n_epochs"] if config["n_epochs"] > 0 else None,  # stops when either one is met
    max_steps=config["n_steps"] if config["n_steps"] > 0 else None,
    accelerator="auto",  # recognizes device
    devices="auto",  # how many devices to use
    precision="16-mixed",  # to use amp 16
    logger=logger,
    log_every_n_steps=1,
    val_check_interval=config["val_check_interval"],  # after how many train batches to check val
    enable_checkpointing=True,  # saves the most recent model after each epoch (default True)
    callbacks=[checkpointer, early_stopping],
    enable_progress_bar=True,
    gradient_clip_val=config["gradient_clip_val"],  # Maximum norm of the gradients
    gradient_clip_algorithm="norm",  # 'norm' or 'value'
    accumulate_grad_batches=config["accumulate_grad_batches"],
)

# Train
trainer.fit(
    model,
    datamodule,
    # ckpt_path=config["init_from"] if config["resume_training"] else None,
)
