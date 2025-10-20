#!/usr/bin/env python3
"""
Text-only baseline fine-tuning for Gemma-2-2b
==============================================
Train Gemma on caption data WITHOUT visual tokens for baseline comparison.
Uses next utterance prediction task, matching the social token training setup.
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from tqdm import tqdm


class TextOnlyDataset(Dataset):
    """Dataset for text-only caption training."""

    def __init__(self, captions_csv, tokenizer, max_length=512):
        """
        Args:
            captions_csv: Path to captions CSV with 'caption' column
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load captions
        df = pd.read_csv(captions_csv)
        if 'caption' not in df.columns:
            raise ValueError(f"captions_csv must have 'caption' column, found: {df.columns.tolist()}")

        self.captions = df['caption'].dropna().tolist()
        print(f"Loaded {len(self.captions)} captions from {captions_csv}")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]

        # Tokenize with BOS (model will shift for next-token prediction)
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate with padding."""
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    attention_mask = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        # Right padding
        input_ids.append(
            torch.cat([item['input_ids'], torch.full((pad_len,), 0, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask)
    }


def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Create labels (shift by 1)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding in loss

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Count actual tokens (not padding)
            num_tokens = (attention_mask == 1).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity, avg_loss


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval=10,
                gradient_accumulation_steps=1, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Create labels (shift by 1 is done internally by the model)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step with gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # Track metrics
        num_tokens = (attention_mask == 1).sum().item()
        total_loss += loss.item() * num_tokens * gradient_accumulation_steps
        total_tokens += num_tokens

        if step % log_interval == 0:
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            pbar.set_postfix({'loss': f'{loss.item()*gradient_accumulation_steps:.4f}', 'ppl': f'{ppl:.2f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Text-only baseline training for Gemma-2-2b")

    # Data
    parser.add_argument('--captions-csv', type=str, default='data/captions.csv',
                        help='Path to captions CSV file')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Fraction of data to use for validation')

    # Model
    parser.add_argument('--model-id', type=str, default='google/gemma-2-2b',
                        help='HuggingFace model ID')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')

    # Training
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision (fp16) training')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/text_only_baseline',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N steps')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 70)
    print("TEXT-ONLY BASELINE TRAINING")
    print("=" * 70)
    print(f"Model: {args.model_id}")
    print(f"Captions: {args.captions_csv}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Load model with appropriate dtype
    dtype = torch.float16 if args.mixed_precision else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    model = model.to(args.device)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"Using dtype: {dtype}")

    # Create datasets
    print("\nCreating datasets...")
    full_dataset = TextOnlyDataset(args.captions_csv, tokenizer, max_length=args.max_length)

    # Split into train/val
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Setup mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    if args.mixed_precision:
        print("Mixed precision (fp16) enabled")

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    metrics_history = []
    best_val_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 70)

        # Train
        train_loss, train_ppl = train_epoch(
            model, train_loader, optimizer, scheduler, args.device, epoch, args.log_interval,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scaler=scaler
        )

        print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")

        # Validation
        val_ppl, val_loss = compute_perplexity(model, val_loader, args.device)
        print(f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        # Save metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ppl': train_ppl,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'lr': scheduler.get_last_lr()[0]
        }
        metrics_history.append(metrics)

        # Save metrics to file
        metrics_df = pd.DataFrame(metrics_history)
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)

        # Save checkpoint
        if epoch % args.save_every == 0 or val_ppl < best_val_ppl:
            checkpoint_dir = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            print(f"Saved checkpoint to {checkpoint_dir}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"New best model! Val PPL: {val_ppl:.2f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation perplexity: {best_val_ppl:.2f}")
    print(f"Final validation perplexity: {val_ppl:.2f}")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
