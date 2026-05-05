"""
WavLM + LoRA Training Module

This module implements a training and evaluation pipeline for a two-head
speech model designed for age classification and spoof detection.

Overview
--------
The model is built on a pretrained WavLM encoder and adapted using LoRA
to enable efficient fine-tuning with a reduced number of trainable parameters.

The architecture consists of:
- A shared encoder (WavLM + LoRA)
- Two task-specific heads:
  • Age head   → minor vs adult
  • Spoof head → real vs spoof

The model jointly learns both tasks to support robust decision-making
under noisy and adversarial conditions.

Workflow
--------
1. Load audio segments from CSV manifests
2. Apply preprocessing (normalization / optional transforms)
3. Extract features using WavLM
4. Compute predictions from both heads
5. Compute weighted loss
6. Update model parameters
7. Evaluate performance on validation/test sets

Experiment Modes
----------------
- Clean baseline (age-only training)
- Noise augmentation (robustness to noise)
- Multi-task training (age + spoof)
- Robustness evaluation under noise and spoof conditions

Notes
-----
- Audio is preprocessed (mono, 16 kHz) and segmented into fixed-length chunks
- The same model structure is reused across experiments
- Behavior is controlled through configuration and data setup
"""

import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchaudio

from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel
from peft import LoraConfig, get_peft_model, TaskType

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    recall_score
)

# Set random seed for reproducible experiments
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# Dataset
# =========================================================

class SpeechManifestDataset(Dataset):
    """
    Speech Dataset

    Loads audio segments and labels from a CSV manifest and prepares them
    for model input.

    Overview
    --------
    Each sample is read from disk, optionally normalized, and passed through
    an optional transform function before being returned to the model.

    Behavior
    --------
    - If spoof labels are not provided → all samples are treated as clean
    - If a transform is provided → it returns waveform, noise_type, and SNR_db
    - Labels are converted to integer format for training

    Output
    ------
    Returns a dictionary containing:
    - input_values : waveform tensor
    - age_label    : 0 (minor) or 1 (adult)
    - spoof_label  : 0 (clean) or 1 (spoof)
    """

    def __init__(self, datasets, audio_transform=None, global_mean=None, global_std=None):

        # Store all samples from the provided CSV files in one internal list.
        self.data = []
        self.global_mean = global_mean
        self.global_std = global_std

        # Optional transform used for experiment-specific changes, such as noise injection.
        self.audio_transform = audio_transform

        if self.global_mean is None or self.global_std is None:
            raise ValueError("global_mean and global_std must be provided for global normalization.")

        for d in datasets:
            df = pd.read_csv(d["csv_path"])

            # Columns for this dataset (may differ between CSVs)
            audio_col = d["audio_col"]
            age_col = d["age_col"]
            spoof_col = d.get("spoof_col", None)
            segment_id_col = d.get("segment_id_col", None)

            for _, row in df.iterrows():

                if spoof_col is None:
                    spoof = 0
                else:
                    value = row[spoof_col]
                    spoof = int(value) if isinstance(value, (int, float)) else (0 if value == "real" else 1) # 0 if clean

                self.data.append({
                  **row.to_dict(),

                  "__segment_id_model": row[segment_id_col],
                  "__audio_path_model": row[audio_col],

                  "__age_model": row[age_col],
                  "__spoof_model": spoof
              })


    def __len__(self):
        # The dataset size is the number of rows in the manifest.
        return len(self.data)

    def __getitem__(self, idx):
        # Select one row from the manifest.
        item = self.data[idx]

        # Load the audio segment path stored in the manifest.
        waveform, _ = torchaudio.load(item["__audio_path_model"])

        # Remove the channel dimension so the model receives shape (audio_length,).
        waveform = waveform.squeeze(0).float()

        # Apply global normalization using training statistics
        waveform = (waveform - self.global_mean) / (self.global_std + 1e-6)

        # Apply optional augmentation after normalization
        if self.audio_transform is not None:
            waveform, noise_type, snr = self.audio_transform(waveform, item)
        else:
            noise_type, snr = "none", "clean"

        # Convert the age label into the integer format expected by CrossEntropyLoss.
        age_label = self._encode_age(item["__age_model"])
        spoof_label = item["__spoof_model"]

        return {
            "input_values": waveform.float(),
            "age_label": torch.tensor(age_label, dtype=torch.long),
            "spoof_label": torch.tensor(spoof_label, dtype=torch.long),

            "segment_id": item["__segment_id_model"],
            "path": item["__audio_path_model"],
            "metadata": item,
            "noise_type": noise_type,
            "SNR_db": snr,
        }


    @staticmethod
    def _encode_age(value):
        """Convert age class label into integer format."""
        # Normalize label text before comparison.
        value = str(value).lower().strip()

        # Minor-related labels are mapped to class 0.
        if value in ["minor", "child", "teen", "0"]:
            return 0

        # Adult-related labels are mapped to class 1.
        if value in ["adult", "1"]:
            return 1

        # Raise an error if an unexpected label appears in the manifest.
        raise ValueError(f"Unknown age label: {value}")


# =========================================================
# Model
# =========================================================

class BaseWavLM(nn.Module):
    """
    Base WavLM model using optional LoRA adapters.

    This base class contains the shared components used by different model
    architectures. The WavLM encoder is loaded from HuggingFace and can be
    adapted using LoRA, where the original pretrained weights remain frozen
    and only low-rank adapter parameters are trained.

    The encoder produces frame-level speech representations. These
    representations are then mean-pooled into one utterance-level
    representation that can be passed to task-specific classification heads.

    Parameters
    ----------
    config : dict
        Dictionary containing model, LoRA, and training configuration.
    """

    def __init__(self, config):
        super().__init__()

        # Load the pretrained WavLM encoder from HuggingFace.
        base_encoder = WavLMModel.from_pretrained(config["wavlm_model_name"])

        # Optionally add LoRA adapters.
        if config.get("use_lora", True):
            # Define where and how LoRA adapters are inserted into WavLM.
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["lora_target_modules"],
                bias="none",
            )

            # Wrap the pretrained encoder with LoRA.
            self.encoder = get_peft_model(base_encoder, lora_config)

        else:
            # Use the pretrained backbone without LoRA.
            self.encoder = base_encoder

        # Optionally freeze the backbone and train only the heads.
        if config.get("freeze_backbone", False):
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _build_head(hidden_size, head_hidden_size, dropout_rate):
        """
        Build a simple binary classification head.

        This lightweight MLP head is reused by both the age-only and
        multi-task architectures.
        """
        return nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_size, 2),
        )

    def _forward_encoder(self, input_values):
        """
        Run the shared WavLM encoder and pool its output.

        Parameters
        ----------
        input_values : torch.Tensor
            Batch of raw audio waveforms with shape (batch_size, audio_length).

        Returns
        -------
        torch.Tensor
            Utterance-level representation with shape (batch_size, hidden_size).
        """
        # Extract frame-level representations from WavLM.
        hidden_states = self.encoder(
            input_values=input_values,
            return_dict=True
        ).last_hidden_state

        # Convert frame-level sequence into one utterance-level representation.
        pooled = hidden_states.mean(dim=1)

        return pooled


class MultiTaskWavLM(BaseWavLM):
    """
    Multi-task speech classifier using WavLM with LoRA adapters.

    The WavLM encoder is adapted using LoRA, where the original pretrained
    weights remain frozen and only low-rank adapter parameters are trained.
    The pooled encoder representation is passed to two classification heads:
    one for age classification and one for spoof detection.

    Parameters
    ----------
    config : dict
        Dictionary containing model and LoRA configuration.
    """

    def __init__(self, config):
        super().__init__(config)

        # Classification head for minor/adult prediction.
        self.age_head = self._build_head(
            hidden_size=config["hidden_size"],
            head_hidden_size=config["head_hidden_size"],
            dropout_rate=config["dropout_rate"],
        )

        # Classification head for clean/spoof prediction.
        self.spoof_head = self._build_head(
            hidden_size=config["hidden_size"],
            head_hidden_size=config["head_hidden_size"],
            dropout_rate=config["dropout_rate"],
        )

    def forward(self, input_values):
        """
        Forward pass.

        Parameters
        ----------
        input_values : torch.Tensor
            Batch of raw audio waveforms with shape (batch_size, audio_length).

        Returns
        -------
        dict
            Dictionary containing age_logits and spoof_logits.
        """
        # Run the shared encoder and obtain utterance-level representation.
        pooled = self._forward_encoder(input_values)

        # Return logits from both task-specific heads.
        return {
            "age_logits": self.age_head(pooled),
            "spoof_logits": self.spoof_head(pooled),
        }


class AgeOnlyWavLM(BaseWavLM):
    """
    Single-task speech classifier using WavLM with LoRA adapters.

    The WavLM encoder is adapted using LoRA, where the original pretrained
    weights remain frozen and only low-rank adapter parameters are trained.
    The pooled encoder representation is passed to one classification head
    for age classification.

    Parameters
    ----------
    config : dict
        Dictionary containing model and LoRA configuration.
    """

    def __init__(self, config):
        super().__init__(config)

        # Classification head for minor/adult prediction.
        self.age_head = self._build_head(
            hidden_size=config["hidden_size"],
            head_hidden_size=config["head_hidden_size"],
            dropout_rate=config["dropout_rate"],
        )

    def forward(self, input_values):
        """
        Forward pass.

        Parameters
        ----------
        input_values : torch.Tensor
            Batch of raw audio waveforms with shape (batch_size, audio_length).

        Returns
        -------
        dict
            Dictionary containing age_logits.
        """
        # Run the shared encoder and obtain utterance-level representation.
        pooled = self._forward_encoder(input_values)

        # Return logits from the age classification head only.
        return {
            "age_logits": self.age_head(pooled),
        }


# =========================================================
# Utility Functions
# =========================================================

def count_parameters(model):
    """
    Print total, trainable, and frozen parameter counts.
    """
    # Count all parameters in the model.
    total = sum(p.numel() for p in model.parameters())

    # Count only parameters that will receive gradients.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Frozen parameters are mostly the pretrained WavLM backbone.
    frozen = total - trainable

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}")
    print(f"Trainable ratio      : {100 * trainable / total:.2f}%")


def compute_train_stats(datasets):
    """
    Compute global mean and standard deviation from training audio only.

    This function iterates over all training audio files and computes:
    - mean: average amplitude across all samples
    - std : standard deviation across all samples

    These values are later used for global normalization.
    """

    # Accumulators to compute statistics over the whole dataset
    total_sum = 0.0        # sum of all waveform values
    total_sq_sum = 0.0     # sum of squared waveform values
    total_count = 0        # total number of samples (not files!)

    for d in datasets:
        df = pd.read_csv(d["csv_path"])
        audio_col = d["audio_col"]

        # Loop over each audio file path
        for i, path in enumerate(df[audio_col]):

            # Load waveform (shape: [1, num_samples])
            waveform, _ = torchaudio.load(path)

            # Remove channel dimension → shape becomes (num_samples,)
            waveform = waveform.squeeze(0).float()

            # Add sum of all values in this waveform
            total_sum += waveform.sum().item()

            # Add sum of squared values (needed for variance)
            total_sq_sum += (waveform ** 2).sum().item()

            # Count how many samples we have (length of waveform)
            total_count += waveform.numel()

    # Compute global mean
    mean = total_sum / total_count

    # Compute variance using formula:
    # Var = E[x^2] - (E[x])^2
    variance = (total_sq_sum / total_count) - (mean ** 2)

    # Standard deviation = sqrt(variance)
    std = variance ** 0.5

    return mean, std


def build_loader(datasets, config, shuffle=False, audio_transform=None,
                 global_mean=None, global_std=None):
    """
    Build a DataLoader from a CSV manifest.

    This function keeps the loading logic consistent across train, validation,
    and test sets while allowing experiment-specific options such as noise
    transforms or spoof labels.
    """

    # Create a dataset object using the selected manifest and experiment options.
    dataset = SpeechManifestDataset(
        datasets=datasets,
        audio_transform=audio_transform,
        global_mean=global_mean,
        global_std=global_std
    )

    # Wrap the dataset in a DataLoader for batching and optional shuffling.
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
    )


def compute_loss(outputs, age_labels, spoof_labels, criterion,
                 age_weight=1.0 , spoof_weight=1.0):
    """
    Compute weighted loss for both single-task and multi-task models.

    For age-only training, the model returns only age_logits, so only age loss
    is computed. For multi-task training, both age and spoof losses are computed
    and combined using task weights.
    """

    # Compute the age classification loss.
    age_loss = criterion(outputs["age_logits"], age_labels)

    # Compute the spoof detection loss only when the model has a spoof head.
    if "spoof_logits" in outputs:
        spoof_loss = criterion(outputs["spoof_logits"], spoof_labels)
    else:
        spoof_loss = torch.tensor(0.0, device=age_labels.device)

    # Combine available losses using task weights.
    total_loss = age_weight * age_loss + spoof_weight * spoof_loss

    return total_loss, age_loss, spoof_loss


# =========================================================
# Training
# =========================================================

def train_model(config, train_datasets, val_datasets, base_run_dir, experiment_name,
                num_epochs=10, age_weight=1.0, spoof_weight=1.0, train_transform=None,
                val_transform=None, global_mean=None, global_std=None, seed=42):
    """
    Train WavLM + LoRA using a flexible architecture setup.

    This function supports clean-only training, noise training, and multitask
    spoof-aware training. The architecture is selected through the config
    dictionary and can use either an age-only model or a multi-task model.

    For age-only training, only the age classification head is used.
    For multi-task training, both age and spoof heads are used, and the
    contribution of each task is controlled through loss weights.

    Examples
    --------
    Age-only training:
        architecture="age_only"

    Spoof-aware multi-task training:
        architecture="multitask", age_weight=0.75, spoof_weight=0.25,
        spoof_col="authenticity"

    Noise training:
        pass a train_transform function that adds noise to the waveform.
    """

    set_seed(seed)

    # Use GPU if available; otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(base_run_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    # Build training loader with optional transform/labels depending on experiment.
    train_loader = build_loader(
        train_datasets,
        config,
        shuffle=True,
        audio_transform=train_transform,
        global_mean=global_mean,
        global_std=global_std
    )

    # Build validation loader without shuffling.
    val_loader = build_loader(
        val_datasets,
        config,
        shuffle=False,
        audio_transform=val_transform,
        global_mean=global_mean,
        global_std=global_std
    )

    # ── Model Selection ─────────────────────────────────────────────

    # Choose architecture based on config
    if config.get("architecture", "multitask") == "age_only":
        model = AgeOnlyWavLM(config).to(device)
    else:
        model = MultiTaskWavLM(config).to(device)

    # Print parameter summary to verify LoRA freezing/training behavior.
    count_parameters(model)

    # CrossEntropyLoss is used for both binary classification heads.
    criterion = nn.CrossEntropyLoss()

    # =========================================================
    # Optimizer with Separate Learning Rates for LoRA and Heads
    # =========================================================

    # Select LoRA parameters if LoRA is enabled.
    lora_params = [
        p for n, p in model.named_parameters()
        if "lora" in n and p.requires_grad
    ]

    # Select classification head parameters.
    head_params = [
        p for n, p in model.named_parameters()
        if ("age_head" in n or "spoof_head" in n) and p.requires_grad
    ]

    # Build optimizer depending on the experiment setting.
    if config.get("use_lora", True):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": lora_params,
                    "lr": config["lora_lr"],
                    "weight_decay": config["weight_decay"],
                },
                {
                    "params": head_params,
                    "lr": config["head_lr"],
                    "weight_decay": config["weight_decay"],
                },
            ]
        )
    else:
        # No LoRA: train only the heads if the backbone is frozen.
        optimizer = torch.optim.AdamW(
            head_params,
            lr=config["head_lr"],
            weight_decay=config["weight_decay"],
        )

    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    # Track the best validation loss for checkpoint saving.
    best_val_loss = float("inf")
    best_epoch = None

    patience = config.get("early_stopping_patience", 3)
    epochs_no_improve = 0

    # Create checkpoint directory if it does not exist.
    save_path = f"{run_dir}/best_model.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\n🚀 Starting training...")
    for epoch in range(num_epochs):
        # Set model to training mode.
        model.train()

        # Running sums for epoch-level logging.
        noise_counts = {}
        total_train_samples = 0

        train_loss = 0.0
        train_age_loss = 0.0
        train_spoof_loss = 0.0

        train_preds = []
        train_labels_all = []

        spoof_preds = []
        spoof_labels_all = []

        for batch_idx, batch in enumerate(train_loader):

            for snr in batch["SNR_db"]:
                snr = str(snr)
                noise_counts[snr] = noise_counts.get(snr, 0) + 1

            total_train_samples += len(batch["SNR_db"])

            # Move batch tensors to GPU/CPU.
            waveforms = batch["input_values"].to(device)
            age_labels = batch["age_label"].to(device)
            spoof_labels = batch["spoof_label"].to(device)

            # Clear gradients from the previous step.
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):

                # Forward pass through WavLM + heads.
                outputs = model(waveforms)

                # Compute weighted multi-task loss.
                total_loss, age_loss, spoof_loss = compute_loss(
                    outputs,
                    age_labels,
                    spoof_labels,
                    criterion,
                    age_weight,
                    spoof_weight,
                )

            # Backpropagate gradients.
            scaler.scale(total_loss).backward()

            # Update trainable parameters.
            scaler.step(optimizer)
            scaler.update()

            # Accumulate losses for reporting.
            train_loss += total_loss.item()
            train_age_loss += age_loss.item()
            train_spoof_loss += spoof_loss.item()

            age_pred = torch.argmax(outputs["age_logits"], dim=1)
            train_preds.extend(age_pred.detach().cpu().numpy())
            train_labels_all.extend(age_labels.detach().cpu().numpy())

            # Spoof prediction (only if exists)
            if "spoof_logits" in outputs:
                spoof_pred = torch.argmax(outputs["spoof_logits"], dim=1)
                spoof_preds.extend(spoof_pred.detach().cpu().numpy())
                spoof_labels_all.extend(spoof_labels.detach().cpu().numpy())

        # Convert accumulated sums to average losses.
        train_loss /= len(train_loader)
        train_age_loss /= len(train_loader)
        train_spoof_loss /= len(train_loader)

        train_acc = accuracy_score(train_labels_all, train_preds)

        if len(spoof_preds) > 0:
            train_spoof_acc = accuracy_score(spoof_labels_all, spoof_preds)
        else:
            train_spoof_acc = None

        # Evaluate on validation data after each epoch.
        val_loss, val_age_loss, val_spoof_loss, val_acc, val_spoof_acc = validate_model(
            model,
            val_loader,
            criterion,
            device,
            age_weight,
            spoof_weight,
        )

        if train_spoof_acc is not None and val_spoof_acc is not None:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Age Acc: {train_acc:.4f} | "
                f"Train Spoof Acc: {train_spoof_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Age Acc: {val_acc:.4f} | "
                f"Val Spoof Acc: {val_spoof_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        epoch_dir = f"{run_dir}/epochs"
        os.makedirs(epoch_dir, exist_ok=True)

        epoch_log = {
            "epoch": epoch + 1,

            "metrics": {
                "train_loss": train_loss,
                "train_age_loss": train_age_loss,
                "train_spoof_loss": train_spoof_loss,
                "train_acc": train_acc,
                "train_spoof_acc": train_spoof_acc,   

                "val_loss": val_loss,
                "val_age_loss": val_age_loss,
                "val_spoof_loss": val_spoof_loss,
                "val_acc": val_acc,
                "val_spoof_acc": val_spoof_acc,       
            },

            "noise_distribution": {
                snr: {
                    "count": count,
                    "percentage": round(100 * count / total_train_samples, 2)
                }
                for snr, count in noise_counts.items()
            }
        }

        with open(f"{epoch_dir}/epoch_{epoch+1:03d}.json", "w") as f:
            json.dump(epoch_log, f, indent=2)

        # Save model only when validation loss improves + reset counter.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0

            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_spoof_acc": val_spoof_acc,   # ← أضيفي هذا
                "config": config
            }, save_path)

        # No improvement → increase counter
        else:
            epochs_no_improve += 1

        # Stop training if no improvement for 'patience' epochs
        if epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping at epoch {epoch+1}")
            break

    print("\n=== Best Model Summary ===")
    print(f"Best Epoch : {best_epoch}")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Saved at : {save_path}")

    return model


def validate_model(model, val_loader, criterion, device,
                   age_weight=1.0, spoof_weight=1.0):
    """
    Run validation and return average losses, age accuracy, and spoof accuracy.

    This function supports both architectures:
    - AgeOnlyWavLM: returns only age_logits
    - MultiTaskWavLM: returns age_logits and spoof_logits

    For age-only models, spoof loss and spoof accuracy are not applicable
    and will be returned as None.
    """

    # Switch the model to evaluation mode.
    model.eval()

    # Initialize loss accumulators.
    total_loss_sum = 0.0
    age_loss_sum = 0.0
    spoof_loss_sum = 0.0

    # Lists used to compute validation age accuracy.
    val_preds = []
    val_labels_all = []

    # Lists used to compute validation spoof accuracy.
    spoof_preds = []
    spoof_labels_all = []

    # Track whether the model returns spoof predictions.
    has_spoof_head = False

    # Disable gradient computation during validation.
    with torch.no_grad():
        for batch in val_loader:

            # Move input waveforms and labels to GPU/CPU.
            waveforms = batch["input_values"].to(device)
            age_labels = batch["age_label"].to(device)
            spoof_labels = batch["spoof_label"].to(device)

            # Forward pass only; no parameter updates.
            outputs = model(waveforms)

            # Check once whether spoof output exists for this model.
            has_spoof_output = "spoof_logits" in outputs
            has_spoof_head = has_spoof_head or has_spoof_output

            # Compute validation loss.
            total_loss, age_loss, spoof_loss = compute_loss(
                outputs,
                age_labels,
                spoof_labels,
                criterion,
                age_weight,
                spoof_weight,
            )

            # Accumulate total and age losses.
            total_loss_sum += total_loss.item()
            age_loss_sum += age_loss.item()

            # Compute age predictions.
            age_pred = torch.argmax(outputs["age_logits"], dim=1)
            val_preds.extend(age_pred.detach().cpu().numpy())
            val_labels_all.extend(age_labels.detach().cpu().numpy())

            # Compute spoof loss and predictions only when spoof output exists.
            if has_spoof_output:
                spoof_loss_sum += spoof_loss.item()

                spoof_pred = torch.argmax(outputs["spoof_logits"], dim=1)
                spoof_preds.extend(spoof_pred.detach().cpu().numpy())
                spoof_labels_all.extend(spoof_labels.detach().cpu().numpy())

    # Number of validation batches.
    n_batches = len(val_loader)

    # Compute average total and age losses.
    avg_total_loss = total_loss_sum / n_batches
    avg_age_loss = age_loss_sum / n_batches

    # Compute validation age accuracy.
    val_acc = accuracy_score(val_labels_all, val_preds)

    # Compute spoof metrics only for multi-task models.
    if has_spoof_head:
        avg_spoof_loss = spoof_loss_sum / n_batches
        val_spoof_acc = accuracy_score(spoof_labels_all, spoof_preds)
    else:
        avg_spoof_loss = None
        val_spoof_acc = None

    return (
        avg_total_loss,
        avg_age_loss,
        avg_spoof_loss,
        val_acc,
        val_spoof_acc
    )


# =========================================================
# Evaluation
# =========================================================

def evaluate_model(config, test_datasets, checkpoint_path, test_transform=None,
                   global_mean=None, global_std=None, eval_spoof=True,
                   run_dir=None, experiment_name=None,
                   age_col="mapped_age_class", spoof_col="authenticity"):
    """
    Evaluate model on full test set and compute all metrics.

    Workflow:
    1) Run inference on all test samples
    2) Store predictions + probabilities + metadata
    3) Build DataFrame from results
    4) Compute:
        - Age head metrics
        - Spoof head metrics (if available)
        - Allow/Block metrics (system-level decision)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build test loader from manifests
    test_loader = build_loader(
        test_datasets,
        config,
        shuffle=False,
        audio_transform=test_transform,
        global_mean=global_mean,
        global_std=global_std
    )

    # Load trained model checkpoint
    if config.get("architecture", "multitask") == "age_only":
        model = AgeOnlyWavLM(config).to(device)
    else:
        model = MultiTaskWavLM(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    sample_records = []

    # Check if spoof labels exist (depends on experiment setup)
    has_spoof = eval_spoof and any(d.get("spoof_col") is not None for d in test_datasets)

    # Run inference on all test samples
    with torch.no_grad():
        for batch in test_loader:

            waveforms = batch["input_values"].to(device)
            outputs = model(waveforms)

            # Age always computed
            age_probs = torch.softmax(outputs["age_logits"], dim=1)
            age_pred = torch.argmax(age_probs, dim=1)

            if has_spoof:
                spoof_probs = torch.softmax(outputs["spoof_logits"], dim=1)
                spoof_pred = torch.argmax(spoof_probs, dim=1)

            # Store results per sample
            for i in range(len(batch["segment_id"])):

                metadata = {
                    k: v[i]
                    for k, v in batch["metadata"].items()
                    if not k.startswith("__")
                }

                record = {
                    **metadata,

                    "predicted_age": int(age_pred[i].item()),
                    "prob_age_minor": float(age_probs[i, 0].item()),
                    "prob_age_adult": float(age_probs[i, 1].item()),
                }

                if has_spoof:
                    record.update({
                        "predicted_spoof": int(spoof_pred[i].item()),
                        "prob_spoof_real": float(spoof_probs[i, 0].item()),
                        "prob_spoof_spoof": float(spoof_probs[i, 1].item()),
                    })

                sample_records.append(record)

    # Convert to DataFrame for metric computation
    df = pd.DataFrame(sample_records)

    results = {
        "experiment_name": experiment_name,
        "metrics": {},
        "samples": sample_records,
        "num_samples": len(sample_records)
    }

    # Age classification performance
    results["metrics"]["age_head"] = compute_age_metrics(df, age_col=age_col)

    # System-level decision (Allow = adult AND real if spoof is used)
    results["metrics"]["allow_block"] = {
        "accuracy": allow_block_accuracy(
            df,
            age_col=age_col,
            spoof_col=spoof_col if has_spoof else None,
            pred_spoof_col="predicted_spoof" if has_spoof else None
        ),
        "false_allow_rate": false_allow_rate(
            df,
            age_col=age_col,
            spoof_col=spoof_col if has_spoof else None,
            pred_spoof_col="predicted_spoof" if has_spoof else None
        ),
        "false_block_rate": false_block_rate(
            df,
            age_col=age_col,
            spoof_col=spoof_col if has_spoof else None,
            pred_spoof_col="predicted_spoof" if has_spoof else None
        )
    }

    # Spoof detection performance (if available)
    if has_spoof:
        results["metrics"]["spoof_head"] = compute_spoof_metrics(df, spoof_col=spoof_col)

        # Additional analysis on specific subsets
        results["metrics"]["allow_block"].update({

            # Errors where spoof samples were incorrectly allowed
            "false_allow_rate_on_all_spoofed_samples": false_allow_rate(
                df[df[spoof_col] == "spoof"],
                age_col=age_col,
                spoof_col=spoof_col,
                pred_spoof_col="predicted_spoof"
            ),

            # Errors where real minors were incorrectly allowed
            "false_allow_rate_on_real_minors": false_allow_rate(
                df[(df[spoof_col] == "real") & (df[age_col] == "minor")],
                age_col=age_col
            )
        })

        # Cross-age spoof (minor → adult)
        if "target_age_class" in df.columns:
            results["metrics"]["allow_block"]["false_allow_rate_on_minor_to_adult_spoof"] = false_allow_rate(
                df[
                    (df[spoof_col] == "spoof") &
                    (df[age_col] == "minor") &
                    (df["target_age_class"] == "adult")
                ],
                age_col=age_col,
                spoof_col=spoof_col,
                pred_spoof_col="predicted_spoof"
            )

    # Print results clearly
    print("\n=== Age Head Results ===")
    print(results["metrics"]["age_head"])

    if has_spoof:
        print("\n=== Spoof Head Results ===")
        print(results["metrics"]["spoof_head"])

    print("\n=== Adult-Only Online Gaming Chatting Evaluation ===")
    print(results["metrics"]["allow_block"])

    # Save results as JSON
    if run_dir is not None and experiment_name is not None:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, f"test_results_{experiment_name}.json")

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results → {path}")

    return results


def compute_age_metrics(df, age_col, pred_age_col="predicted_age"):
    """
    Compute age classification metrics (minor vs adult).

    Inputs:
    - df: DataFrame with ground truth and predictions
    - age_col: name of true age column ("minor"/"adult")
    - pred_age_col: predicted labels (0=minor, 1=adult)

    Returns:
    - accuracy, macro_f1, balanced_accuracy
    - adult_recall, minor_recall
    - confusion_matrix
    """

    y_true = df[age_col].map({"minor": 0, "adult": 1}).astype(int)
    y_pred = df[pred_age_col].astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "adult_recall": recall_score(y_true, y_pred, pos_label=1),
        "minor_recall": recall_score(y_true, y_pred, pos_label=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_spoof_metrics(df, spoof_col, pred_spoof_col="predicted_spoof"):
    """
    Compute spoof detection metrics (real vs spoof).

    Inputs:
    - spoof_col: true labels ("real"/"spoof")
    - pred_spoof_col: predictions (0=real, 1=spoof)

    Returns:
    - accuracy, macro_f1
    - real_recall, spoof_recall
    - false_spoof_acceptance_rate
    - confusion_matrix
    """

    y_true = df[spoof_col].map({"real": 0, "spoof": 1}).astype(int)
    y_pred = df[pred_spoof_col].astype(int)

    return {
        "spoof_accuracy": accuracy_score(y_true, y_pred),
        "spoof_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "real_recall": recall_score(y_true, y_pred, pos_label=0),
        "spoof_recall": recall_score(y_true, y_pred, pos_label=1),
        "false_spoof_acceptance_rate": ((y_true == 1) & (y_pred == 0)).sum() / max(1, (y_true == 1).sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def false_allow_rate(df, age_col, pred_age_col="predicted_age",
                     spoof_col=None, pred_spoof_col=None):
    """
    Compute false allow rate.

    Definition:
    - Allow:
        • Without spoof → predicted_age == adult
        • With spoof → predicted_age == adult AND predicted_spoof == real

    - False Allow:
        • Without spoof → allowed sample where true age = minor
        • With spoof → allowed sample where (minor OR spoof)

    Parameters
    ----------
    df : DataFrame
        Prediction results.

    age_col : str
        True age column ("minor"/"adult").

    pred_age_col : str
        Predicted age column (0=minor, 1=adult).

    spoof_col : str, optional
        True spoof column ("real"/"spoof").

    pred_spoof_col : str, optional
        Predicted spoof column (0=real, 1=spoof).

    Returns
    -------
    float
        False allow rate on the given subset.
    """

    true_age = df[age_col].map({"minor": 0, "adult": 1}).astype(int)
    pred_age = df[pred_age_col].astype(int)

    if spoof_col is None or pred_spoof_col is None:
        allow = pred_age == 1
        should_block = (true_age == 0)
    else:
        true_spoof = df[spoof_col].map({"real": 0, "spoof": 1}).astype(int)
        pred_spoof = df[pred_spoof_col].astype(int)

        allow = (pred_age == 1) & (pred_spoof == 0)
        should_block = (true_age == 0) | (true_spoof == 1)

    false_allow = allow & should_block

    return false_allow.sum() / max(1, should_block.sum())


def false_block_rate(df, age_col, pred_age_col="predicted_age",
                     spoof_col=None, pred_spoof_col=None):
    """
    Compute false block rate.

    False Block:
    - Sample should be allowed, but model blocks it.
    - Without spoof: true adult is blocked.
    - With spoof: true adult + real is blocked.
    """

    true_age = df[age_col].map({"minor": 0, "adult": 1}).astype(int)
    pred_age = df[pred_age_col].astype(int)

    if spoof_col is None or pred_spoof_col is None:
        allow = pred_age == 1
        should_allow = true_age == 1
    else:
        true_spoof = df[spoof_col].map({"real": 0, "spoof": 1}).astype(int)
        pred_spoof = df[pred_spoof_col].astype(int)

        allow = (pred_age == 1) & (pred_spoof == 0)
        should_allow = (true_age == 1) & (true_spoof == 0)

    block = ~allow
    false_block = block & should_allow

    return false_block.sum() / max(1, should_allow.sum())


def allow_block_accuracy(df, age_col, pred_age_col="predicted_age",
                         spoof_col=None, pred_spoof_col=None):
    """
    Compute allow/block accuracy.
    Allow = adult (and real if spoof is used).
    """

    true_age = df[age_col].map({"minor": 0, "adult": 1}).astype(int)
    pred_age = df[pred_age_col].astype(int)

    if spoof_col is None or pred_spoof_col is None:
        allow = pred_age == 1
        should_allow = true_age == 1
    else:
        true_spoof = df[spoof_col].map({"real": 0, "spoof": 1}).astype(int)
        pred_spoof = df[pred_spoof_col].astype(int)

        allow = (pred_age == 1) & (pred_spoof == 0)
        should_allow = (true_age == 1) & (true_spoof == 0)

    return (allow == should_allow).mean()
