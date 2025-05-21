import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
# from functools import partial # Not strictly needed for definition, but for usage example

class ProteinDataset(Dataset):
    """
    A PyTorch Dataset for protein sequences, suitable for causal language modeling
    with a Transformer model.
    """
    def __init__(self, tokenizer_path: str, dataset_file_path: str, block_size: int):
        """
        Args:
            tokenizer_path: Path to the saved Hugging Face Tokenizer JSON file.
            dataset_file_path: Path to the dataset file. Each line should contain
                               one protein sequence, already formatted with spaces
                               between amino acids and wrapped with <|endoftext|> tokens.
                               Example: "<|endoftext|> M F V ... L P <|endoftext|>"
            block_size: The fixed sequence length for model input. Sequences will be
                        truncated or padded (by collate_fn) to this length.
        """
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.block_size = block_size
        
        # Ensure PAD token is part of the tokenizer's vocabulary
        pad_token_str = "[PAD]"
        if self.tokenizer.token_to_id(pad_token_str) is None:
            # This case should ideally not happen if tokenizer.py added it.
            # If it can, one might need to add it: self.tokenizer.add_special_tokens([pad_token_str])
            raise ValueError(f"PAD token '{pad_token_str}' not found in tokenizer. Please ensure it's added during tokenizer training.")
        self.pad_token_id = self.tokenizer.token_to_id(pad_token_str)

        # EOS/BOS token
        eos_bos_token_str = "<|endoftext|>"
        if self.tokenizer.token_to_id(eos_bos_token_str) is None:
            raise ValueError(f"EOS/BOS token '{eos_bos_token_str}' not found in tokenizer.")
        self.eos_token_id = self.tokenizer.token_to_id(eos_bos_token_str) # Used as both BOS and EOS

        self.tokenized_sequences = []
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Tokenize the line. The tokenizer should handle special tokens like <|endoftext|>
                # if they were part of its training and vocabulary.
                encoded = self.tokenizer.encode(line) # Returns an Encoding object
                token_ids = encoded.ids
                
                # Filter out sequences that are too short for causal LM (input/target pair)
                if len(token_ids) < 2:
                    # print(f"Warning: Skipping sequence of length {len(token_ids)}: {line[:30]}...")
                    continue 
                
                self.tokenized_sequences.append(token_ids)

    def __len__(self) -> int:
        """Returns the total number of sequences in the dataset."""
        return len(self.tokenized_sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single tokenized sequence and prepares it for causal language modeling.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            A tuple (input_ids, target_ids), both as PyTorch tensors.
            - input_ids: Sequence tokens, truncated to block_size.
            - target_ids: Sequence tokens shifted by one, truncated to block_size.
        """
        token_ids = self.tokenized_sequences[idx]

        # For Causal LM: input is current token, target is the next token.
        # Example: if token_ids is [BOS, t1, t2, EOS]
        # input_ids_full should be [BOS, t1, t2]
        # target_ids_full should be [t1, t2, EOS]
        input_ids_full = token_ids[:-1]
        target_ids_full = token_ids[1:]

        # Truncate to block_size.
        # Note: Slicing up to self.block_size handles sequences shorter than block_size naturally.
        # The collate_fn will handle padding them up to block_size later.
        current_input_ids = input_ids_full[:self.block_size]
        current_target_ids = target_ids_full[:self.block_size]
        
        return torch.tensor(current_input_ids, dtype=torch.long), \
               torch.tensor(current_target_ids, dtype=torch.long)

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """
    Collates a batch of (input_ids, target_ids) tensors into padded batch tensors.

    Args:
        batch: A list of tuples, where each tuple is (input_ids_tensor, target_ids_tensor)
               as returned by ProteinDataset.__getitem__.
        pad_token_id: The ID to use for padding sequences.

    Returns:
        A dictionary containing:
        - "input_ids": A tensor of padded input sequences (batch_size, max_seq_len_in_batch).
        - "target_ids": A tensor of padded target sequences (batch_size, max_seq_len_in_batch).
    """
    # Separate input_ids and target_ids from the batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Pad input sequences
    # torch.nn.utils.rnn.pad_sequence expects a list of Tensors and pads them to the longest Tensor in the list.
    # batch_first=True makes the output (batch_size, seq_len)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        inputs, 
        batch_first=True, 
        padding_value=float(pad_token_id) # pad_sequence expects float for padding_value
    )

    # Pad target sequences
    padded_targets = torch.nn.utils.rnn.pad_sequence(
        targets, 
        batch_first=True, 
        padding_value=float(pad_token_id)
    )
    
    return {"input_ids": padded_inputs.long(), "target_ids": padded_targets.long()}


if __name__ == '__main__':
    # This block is for demonstration and basic testing.
    # It requires a dummy tokenizer and dataset file.
    
    # 1. Create a dummy tokenizer file (protein_tokenizer.json)
    try:
        from tokenizers import ByteLevelBPETokenizer, AddedToken
        dummy_tokenizer_path = "dummy_protein_tokenizer.json"
        
        # Check if dummy tokenizer already exists
        try:
            Tokenizer.from_file(dummy_tokenizer_path)
            print(f"Dummy tokenizer '{dummy_tokenizer_path}' already exists.")
        except Exception: # Catch more general exception if file is invalid or not found
            print(f"Creating dummy tokenizer '{dummy_tokenizer_path}'...")
            tokenizer = ByteLevelBPETokenizer()
            eos_bos_token = AddedToken("<|endoftext|>", single_word=True, special=True)
            pad_token = AddedToken("[PAD]", single_word=True, special=True)
            # Train with minimal data and include special tokens
            tokenizer.train_from_iterator(["M F V L", "A G C T"], 
                                          min_frequency=1,
                                          special_tokens=[str(eos_bos_token), str(pad_token)])
            tokenizer.save(dummy_tokenizer_path, pretty=True)
            print(f"Dummy tokenizer '{dummy_tokenizer_path}' created and saved.")

        # 2. Create a dummy dataset file (dummy_dataset.fas)
        dummy_dataset_path = "dummy_dataset.fas"
        with open(dummy_dataset_path, 'w', encoding='utf-8') as f:
            f.write("<|endoftext|> M F V F L V L L P L <|endoftext|>\n") # len 12 tokens after encode (example)
            f.write("<|endoftext|> S Q C V N L T R T Q L P P A Y T N S F T R G V Y Y P D K V F R S S V L H S I Q D L F L P F F S N V <|endoftext|>\n") # A longer one
            f.write("<|endoftext|> A G C <|endoftext|>\n") # A short one (5 tokens)
            f.write("<|endoftext|> M <|endoftext|>\n") # Very short (3 tokens) -> input [bos, M], target [M, eos]
            f.write("<|endoftext|><|endoftext|>\n") # Shortest valid (2 tokens) -> input [bos], target [eos]
            f.write("M F V\n") # Line without BOS/EOS tokens (will be tokenized as is)
            f.write("\n") # Empty line (should be skipped)
            f.write("<|endoftext|> L <|endoftext|>\n") # Another short one

        # 3. Test ProteinDataset
        print("\nTesting ProteinDataset...")
        block_size = 10 
        try:
            dataset = ProteinDataset(
                tokenizer_path=dummy_tokenizer_path,
                dataset_file_path=dummy_dataset_path,
                block_size=block_size
            )
            print(f"Dataset size: {len(dataset)}")
            if len(dataset) > 0:
                input_ids, target_ids = dataset[0]
                print(f"Sample 0 - input_ids (len {len(input_ids)}): {input_ids}")
                print(f"Sample 0 - target_ids (len {len(target_ids)}): {target_ids}")
                
                # Verify token IDs for special tokens
                print(f"PAD token ID from dataset: {dataset.pad_token_id}")
                print(f"EOS/BOS token ID from dataset: {dataset.eos_token_id}")

                # Test with a small DataLoader
                from torch.utils.data import DataLoader
                from functools import partial

                # Need to use partial to pass pad_token_id to collate_fn
                custom_collate_fn = partial(collate_fn, pad_token_id=dataset.pad_token_id)
                
                data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
                
                print("\nTesting DataLoader and collate_fn...")
                for i, batch in enumerate(data_loader):
                    print(f"Batch {i}:")
                    print(f"  Input IDs shape: {batch['input_ids'].shape}") # Expected (batch_size, block_size or less if all are shorter)
                    print(f"  Input IDs: {batch['input_ids']}")
                    print(f"  Target IDs shape: {batch['target_ids'].shape}")
                    print(f"  Target IDs: {batch['target_ids']}")
                    if i >= 1: # Print a couple of batches
                        break
            else:
                print("Dataset is empty, cannot show sample or test DataLoader.")

        except Exception as e:
            print(f"Error during ProteinDataset or DataLoader test: {e}")
            import traceback
            traceback.print_exc()

        # 4. Cleanup dummy files (optional)
        # import os
        # os.remove(dummy_tokenizer_path)
        # os.remove(dummy_dataset_path)
        # print("\nCleaned up dummy files.")

    except ImportError:
        print("Skipping __main__ example: `tokenizers` or `functools` import failed.")
    except Exception as e:
        print(f"An error occurred in __main__: {e}")
