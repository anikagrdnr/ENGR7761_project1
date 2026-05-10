"""
MLCV Project 1 — main.py
Entry point for training the waste classification CNN.

Reference: A Study of Garbage Classification with CNN
https://www.sciencedirect.com/science/article/pii/S2773186324000768

TODO - get initial dataset train test split
- check transforms
- consider small subset 
"""

import torch
import torch.nn as nn
import torch.optim as optim

from scraps.project1_gd.config_gd        import *
from scraps.project1_gd.model_gd         import CNN
from dataset       import WasteData, get_transforms, get_dataloader
from scraps.project1_gd.trainer_gd       import Trainer
from scraps.project1_gd.visualisation_gd import run_all


if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #transforms - turn on flags 
    train_tf = get_transforms(IMG_SIZE, train=True)
    val_tf   = get_transforms(IMG_SIZE, train=False)
    #test_tf = get_transforms(IMG_SIZE, test=True) #TODO adjsut test transform

    # NOTE: assumes data split into test train val run split_data.py to split og dataset 70/15/15
    ds_train = WasteData(DATA_DIR / "train", transform=train_tf)
    ds_val   = WasteData(DATA_DIR / "val",   transform=val_tf)
    #ds_test = WasteData(DATA_DIR/"test", transform=test_tf) #test_df add severe transforms to induce domain shift 

    dl_train = get_dataloader(ds_train, BATCH_SIZE, train=True,  num_workers=NUM_WORKERS)
    dl_val   = get_dataloader(ds_val,   BATCH_SIZE, train=False, num_workers=NUM_WORKERS)

    model = CNN(HIDDEN_SIZE, NUM_CLASSES).to(device)
    print(f"Trainable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimiser = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=N_EPOCHS)

    trainer = Trainer(
        model     = model,
        optimiser = optimiser,
        criterion = criterion,
        dl_train  = dl_train,
        dl_val    = dl_val,
        device    = device,
        scheduler = scheduler,
        log_dir   = LOG_DIR,
    )

    train_losses, val_losses, train_accs, val_accs = trainer.fit(
        N_EPOCHS, save_path=SAVE_PATH)

    #saves params and saves visualisations. TODO add loogging to visualise whilst running.. 
    #method in visualisation 
    run_all(
        model        = model,
        optimiser    = optimiser,
        criterion    = criterion,
        scheduler    = scheduler,
        config       = {
            "IMG_SIZE":    IMG_SIZE,
            "BATCH_SIZE":  BATCH_SIZE,
            "N_EPOCHS":    N_EPOCHS,
            "LR":          LR,
            "HIDDEN_SIZE": HIDDEN_SIZE,
            "NUM_CLASSES": NUM_CLASSES,
        },
        train_losses = train_losses,
        val_losses   = val_losses,
        train_accs   = train_accs,
        val_accs     = val_accs,
        dl_val       = dl_val,
        train_dataset= ds_train,
        device       = device,
    )