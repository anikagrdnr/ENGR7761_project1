"""
MLCV Project 1 — trainer.py
Encapsulates training, validation, and evaluation logic.
#reference docs 
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 

"""

import time
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Manages the training and validation loop for the waste classification CNN.

    Args:
        model     : CNN instance
        optimiser : torch.optim optimiser
        criterion : loss function (CrossEntropyLoss)
        scheduler : lr scheduler (optional, call step per epoch)
        dl_train  : training DataLoader
        dl_val    : validation DataLoader
        device    : torch.device
        log_dir   : TensorBoard log directory
    """

    def __init__(self, model, optimiser, criterion, dl_train, dl_val,
                 device, scheduler=None, log_dir="runs/experiment"):
        self.model     = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.scheduler = scheduler
        self.dl_train  = dl_train
        self.dl_val    = dl_val
        self.device    = device
        self.writer    = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def calculate_accuracy(preds, labels):
        return (preds.argmax(dim=1) == labels).float().mean().item()

    @staticmethod
    def epoch_time(start, end):
        elapsed = end - start
        return int(elapsed // 60), int(elapsed % 60)

    def train(self):
        """
        One full pass over the training DataLoader.
        -> WeightedRandomSampler in DataLoader ensures IID mini-batches
        -> Loss and gradients computed per batch
        -> Returns mean loss and accuracy over the epoch
        """
        self.model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for imgs, labels in self.dl_train:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            preds = self.model(imgs)
            loss  = self.criterion(preds, labels)
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item()
            total_acc  += self.calculate_accuracy(preds, labels)
            n          += 1

        return total_loss / n, total_acc / n

#adjusted evaluation loop
    def evaluate(self):
        """
        One full pass over the validation DataLoader.
        -> No gradient computation
        -> No augmentation (handled by DataLoader transform)
        -> Returns mean loss and accuracy over the epoch
        """
        self.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0

        with torch.no_grad():
            for imgs, labels in self.dl_val:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(imgs)
                loss  = self.criterion(preds, labels)

                total_loss += loss.item()
                total_acc  += self.calculate_accuracy(preds, labels)
                n          += 1

        return total_loss / n, total_acc / n

    def fit(self, n_epochs, save_path="best_model.pt"):
        """
        Full training loop over n_epochs.
        -> Logs loss and accuracy to TensorBoard each epoch
        -> Saves best model weights based on validation loss
        -> Steps scheduler each epoch if provided
        """
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            start = time.monotonic()

            train_loss, train_acc = self.train()
            val_loss,   val_acc   = self.evaluate()

            if self.scheduler:
                self.scheduler.step()

            #for visualisation
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # TensorBoard logging
            self.writer.add_scalars("Loss", {
                "train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {
                "train": train_acc, "val": val_acc}, epoch)
            if self.scheduler:
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar("LR", lr, epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)

            mins, secs = self.epoch_time(start, time.monotonic())
            print(f"Epoch {epoch+1:02d}/{n_epochs} | "
                  f"Time: {mins}m {secs}s | "
                  f"Train Loss: {train_loss:.3f}  Acc: {train_acc*100:.1f}% | "
                  f"Val Loss:   {val_loss:.3f}  Acc: {val_acc*100:.1f}%")

        self.writer.close()
        print(f"\nBest val loss: {best_val_loss:.3f} — saved to {save_path}")
        print(f"Trainable parameters: {self.count_parameters(self.model):,}")
        return train_losses, val_losses, train_accs, val_accs