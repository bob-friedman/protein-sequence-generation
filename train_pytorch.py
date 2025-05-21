import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from functools import partial
import argparse
import os
import math # For math.sqrt in model, but not directly here. Good to have if example code were moved.

# Assuming pytorch_transformer_model.py and protein_dataset.py are in the same directory
# or accessible via PYTHONPATH.
from pytorch_transformer_model import ProteinTransformerModel
from protein_dataset import ProteinDataset, collate_fn

def train(args):
    """
    Main training function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.model_output_dir, exist_ok=True)
    print(f"Model checkpoints will be saved to: {args.model_output_dir}")

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
        return
    
    vocab_size = tokenizer.get_vocab_size()
    # ProteinDataset will load and store pad_token_id from the tokenizer.
    # We fetch it from the dataset instance later.

    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    # Create Dataset and DataLoader
    try:
        dataset = ProteinDataset(
            tokenizer_path=args.tokenizer_path,
            dataset_file_path=args.dataset_path,
            block_size=args.block_size
        )
    except Exception as e:
        print(f"Failed to create ProteinDataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Please check dataset_path and its content.")
        return

    pad_token_id = dataset.pad_token_id # Get pad_token_id from the dataset instance
    print(f"Using PAD token ID: {pad_token_id}")

    _collate_fn = partial(collate_fn, pad_token_id=pad_token_id)
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=_collate_fn,
        shuffle=True,
        num_workers=args.num_workers # Added num_workers from args
    )

    # Instantiate Model
    # max_len for positional encoding should be at least block_size.
    # Adding a bit of buffer (e.g., +50) is fine, or just use block_size if sequences
    # are strictly truncated to block_size.
    model = ProteinTransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        dropout=args.dropout,
        max_len=args.block_size + 50 
    )
    model.to(device)
    print(f"Model instantiated with d_model={args.d_model}, nhead={args.nhead}, nlayers={args.nlayers}.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


    # Optimizer and Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    print(f"Optimizer: AdamW, Learning Rate: {args.learning_rate}")
    print(f"Loss Function: CrossEntropyLoss (ignoring index {pad_token_id})")

    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for i, batch in enumerate(data_loader):
            # input_ids shape (batch_size, seq_len), target_ids shape (batch_size, seq_len)
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Permute input_ids to (seq_len, batch_size) for the model
            src = input_ids.permute(1, 0) 
            targets = target_ids # Shape (batch_size, seq_len)

            # Generate causal mask for the source sequence length
            # Mask needs to be on the same device as the model input
            src_mask = model._generate_square_subsequent_mask(src.size(0)).to(device)

            optimizer.zero_grad()
            
            # Output logits shape (seq_len, batch_size, vocab_size)
            output_logits = model(src, src_mask)
            
            # Reshape for loss calculation:
            # Output: (batch_size * seq_len, vocab_size)
            # Target: (batch_size * seq_len)
            output_for_loss = output_logits.permute(1, 0, 2).reshape(-1, vocab_size)
            targets_for_loss = targets.reshape(-1)
            
            loss = criterion(output_for_loss, targets_for_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm) # Use arg
            optimizer.step()

            total_loss += loss.item()
            num_batches +=1

            if (i + 1) % args.log_every == 0: # Use arg
                avg_batch_loss = loss.item() # Current batch loss
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(data_loader)}], Loss: {avg_batch_loss:.4f}")

        avg_epoch_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Epoch Loss: {avg_epoch_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.model_output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss, # Save average epoch loss
                'args': args # Save training arguments
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Protein Transformer Model")
    
    # Paths
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to protein_tokenizer.json")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset.fas (training data)")
    parser.add_argument("--model_output_dir", type=str, default="./protein_model_checkpoints", help="Directory to save trained model checkpoints")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. 0 for main process.") # Added

    # Model hyperparameters
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length for the model (max_len for positional encoding will be block_size + 50)")
    parser.add_argument("--d_model", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--d_hid", type=int, default=512, help="Hidden dimension in feedforward layers (dim_feedforward in Transformer)")
    parser.add_argument("--nlayers", type=int, default=3, help="Number of Transformer decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Checkpointing and Logging
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=10, help="Log training progress every N batches") # Added
    parser.add_argument("--clip_grad_norm", type=float, default=0.5, help="Max norm for gradient clipping") # Added

    args = parser.parse_args()
    
    train(args)
