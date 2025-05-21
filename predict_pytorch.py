import torch
import argparse
from tokenizers import Tokenizer
import sys # For potential path adjustments
import os  # For potential path adjustments

# Attempt to import local modules.
# This structure assumes that predict_pytorch.py might be in a 'scripts' directory,
# and pytorch_transformer_model.py & protein_utils.py are in the parent directory (project root).
try:
    # If predict_pytorch.py is in the root with other .py files:
    from pytorch_transformer_model import ProteinTransformerModel
    from protein_utils import format_sequence_with_spaces
except ImportError:
    # If predict_pytorch.py is in a subdirectory like 'scripts/':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from pytorch_transformer_model import ProteinTransformerModel
    from protein_utils import format_sequence_with_spaces


def generate_sequence(args):
    """
    Loads a trained model and generates a sequence based on a prompt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    try:
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {args.tokenizer_path}: {e}")
        return

    vocab_size = tokenizer.get_vocab_size()
    eos_token_str = "<|endoftext|>"
    pad_token_str = "[PAD]" # For cleaning output

    eos_token_id = tokenizer.token_to_id(eos_token_str)
    if eos_token_id is None:
        print(f"Error: EOS token '{eos_token_str}' not found in tokenizer.")
        return
    
    print(f"Tokenizer loaded. Vocab size: {vocab_size}. EOS token ID: {eos_token_id}.")

    # Instantiate Model
    # max_len for positional encoding context. args.block_size is appropriate.
    # Using args.block_size + 50 as in training for consistency if sequences could exceed block_size
    # during generation before max_length is hit, though typically generation is within block_size.
    model = ProteinTransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        dropout=args.dropout, # Dropout is typically not active in eval mode, but good to match.
        max_len=args.block_size + 50 
    )
    print(f"Model instantiated with d_model={args.d_model}, nhead={args.nhead}, nlayers={args.nlayers}.")

    # Load Model Checkpoint
    try:
        checkpoint = torch.load(args.model_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model checkpoint loaded from {args.model_checkpoint_path} (epoch {checkpoint.get('epoch', 'N/A')}).")
    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        return
        
    model.to(device)
    model.eval() # Set model to evaluation mode

    # Format and Tokenize Prompt
    if not args.prompt_sequence:
        print("Error: Prompt sequence cannot be empty.")
        return

    formatted_prompt_seq = format_sequence_with_spaces(args.prompt_sequence)
    # format_sequence_with_spaces might add a trailing space if the last char is a letter.
    # This should be fine for the tokenizer. Example: "M F V "
    
    # Prepend BOS token <|endoftext|> to the formatted sequence for tokenization
    input_string_for_tokenizer = f"{eos_token_str}{formatted_prompt_seq.strip()}" 
    # .strip() ensures no leading/trailing spaces from formatted_prompt_seq affect BOS attachment.
    # The space after each amino acid inside formatted_prompt_seq is preserved.
    
    print(f"Input string for tokenizer: '{input_string_for_tokenizer}'")
    
    tokenized_prompt = tokenizer.encode(input_string_for_tokenizer).ids
    
    # Remove any additional BOS/EOS tokens if encode added them redundantly,
    # keeping only the first BOS and structure.
    # Typically, tokenizer.encode on a string already containing special tokens handles them correctly if they are known.
    # For BPE, "<|endoftext|>M F V" -> might be tokenized as ["<|endoftext|>", "M", " F", " V"]
    # If format_sequence_with_spaces was changed to *not* add trailing space, then "M F V"
    # and input_string_for_tokenizer would be "<|endoftext|>M F V"
    # The important part is that tokenized_prompt starts with the ID for eos_token_str.
    if not tokenized_prompt or tokenized_prompt[0] != eos_token_id:
         # If format_sequence_with_spaces or tokenizer behaves unexpectedly,
         # force the beginning with eos_token_id
         raw_formatted_prompt_ids = tokenizer.encode(formatted_prompt_seq.strip()).ids
         tokenized_prompt = [eos_token_id] + raw_formatted_prompt_ids
         print(f"Adjusted tokenized_prompt to ensure BOS: {tokenized_prompt}")


    generated_ids = list(tokenized_prompt) # Start with the tokenized prompt
    print(f"Initial tokenized prompt IDs: {generated_ids} (length {len(generated_ids)})")


    # Generation Loop (Greedy Decoding)
    print("Generating sequence...")
    with torch.no_grad():
        for _ in range(args.max_length - len(tokenized_prompt)):
            if len(generated_ids) >= args.max_length:
                print("Reached max_length.")
                break

            # Prepare current sequence as input for the model
            # Shape: (current_seq_len, batch_size=1)
            current_input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device).permute(1, 0)
            
            # Generate causal mask for the current input length
            src_mask = model._generate_square_subsequent_mask(current_input_tensor.size(0)).to(device)
            
            output_logits = model(current_input_tensor, src_mask)
            # output_logits shape: (current_seq_len, 1, vocab_size)
            
            next_token_logits = output_logits[-1, 0, :] # Logits for the last token position
            next_token_id = torch.argmax(next_token_logits).item() # Greedy decoding
            
            if next_token_id == eos_token_id:
                print("EOS token generated. Stopping generation.")
                generated_ids.append(next_token_id) # Append EOS to see it in raw output if desired
                break
            
            generated_ids.append(next_token_id)
        
    # Decode and Print Output
    # skip_special_tokens=True will remove <|endoftext|> and [PAD]
    output_sequence_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Further manual cleaning if tokenizer's skip_special_tokens is not enough
    # (e.g., if format_sequence_with_spaces leaves artifacts not handled by tokenizer)
    # The output of BPE with space formatting should be "M F V L P" which is fine.
    # If there are unwanted internal spaces, they might need to be removed.
    # For now, assume tokenizer.decode handles it well.
    # Example: output_sequence_text = output_sequence_text.replace(" ", "") # To get condensed "MFVLP"

    print("\n--- Generated Sequence ---")
    print(f"Raw generated token IDs: {generated_ids}")
    print(f"Decoded and cleaned sequence (length {len(output_sequence_text.replace(' ', ''))}):") # Length of actual AAs
    print(output_sequence_text)
    print("--- End of Sequence ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein sequences using a trained ProteinTransformerModel")

    # Paths
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to protein_tokenizer.json")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to a trained .pt model checkpoint")

    # Generation parameters
    parser.add_argument("--prompt_sequence", type=str, required=True, help="Initial amino acid sequence string to start generation (e.g., 'MFVFL')")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the generated sequence (including prompt)")

    # Model hyperparameters (must match the loaded checkpoint)
    # These are needed to instantiate the model structure before loading weights.
    parser.add_argument("--block_size", type=int, required=True, help="Max sequence length the model was trained with (for positional encoding context)")
    parser.add_argument("--d_model", type=int, required=True, help="Model embedding dimension")
    parser.add_argument("--nhead", type=int, required=True, help="Number of attention heads")
    parser.add_argument("--d_hid", type=int, required=True, help="Hidden dimension in feedforward layers")
    parser.add_argument("--nlayers", type=int, required=True, help="Number of Transformer decoder layers")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout rate used during training (for model instantiation)")
    
    args = parser.parse_args()
    generate_sequence(args)
