import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
import time
from datetime import datetime
import config
from model import DOATransformer
from dataset import get_dataloader


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)

        # Decoder input is the target sequence shifted right (starts with START_TOKEN)
        tgt_input = tgt[:, :-1]

        # Decoder target is the target sequence (ends with END_TOKEN)
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        output = model(src, tgt_input)

        # Reshape for loss calculation
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Validating"):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_log(log_data, log_path):
    """Save training log to JSON file"""
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def load_log(log_path):
    """Load existing training log"""
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            return json.load(f)
    return {"training_history": [], "best_val_loss": float('inf'), "best_epoch": 0}


def main():
    # Setup
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = config.DEVICE
    print(f"Using device: {device}")

    # Log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/training_log_{timestamp}.json"

    # DataLoaders
    train_loader = get_dataloader(config.TRAIN_DATA_PATH, config.BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(config.VAL_DATA_PATH, config.BATCH_SIZE, shuffle=False)

    # Model, Optimizer, Criterion
    model = DOATransformer(
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.D_FF,
        dropout=config.DROPOUT,
        vocab_size=config.VOCAB_SIZE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Ignore PAD_TOKEN in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN)

    # Initialize logging
    log_data = {
        "training_history": [],
        "best_val_loss": float('inf'),
        "best_epoch": 0,
        "config": {
            "d_model": config.D_MODEL,
            "n_head": config.N_HEAD,
            "num_encoder_layers": config.NUM_ENCODER_LAYERS,
            "num_decoder_layers": config.NUM_DECODER_LAYERS,
            "dim_feedforward": config.D_FF,
            "dropout": config.DROPOUT,
            "vocab_size": config.VOCAB_SIZE,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS
        },
        "start_time": datetime.now().isoformat(),
        "device": str(device)
    }

    # Training Loop
    best_val_loss = float('inf')
    best_epoch = 0

    print("Starting training...")
    print(f"Log file: {log_path}")
    print("-" * 50)

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start_time

        # Log epoch results
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": epoch_time,
            "timestamp": datetime.now().isoformat()
        }

        log_data["training_history"].append(epoch_log)

        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Update best model info in log
            log_data["best_val_loss"] = best_val_loss
            log_data["best_epoch"] = best_epoch

            # Save model
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"✓ New best model saved! Val Loss: {val_loss:.4f} (Epoch {best_epoch})")
        else:
            print(f"  No improvement (Best: {best_val_loss:.4f} at Epoch {best_epoch})")

        # Save log after each epoch
        save_log(log_data, log_path)
        print("-" * 50)

    # Final summary
    log_data["end_time"] = datetime.now().isoformat()
    log_data["total_epochs"] = config.NUM_EPOCHS

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {config.MODEL_PATH}")
    print(f"Training log saved to: {log_path}")

    # Save final log
    save_log(log_data, log_path)


if __name__ == "__main__":
    main()