"""
Converts a file of protein sequences (one sequence per line, potentially
in Method #1 format) to Method #2 format. Each output line is space-delimited
and wrapped with <|endoftext|> tokens. Reads from a file or standard input.

Usage:
  python prepare_dataset.py [input_file]
  cat input.txt | python prepare_dataset.py
"""
import argparse
import sys
import os

# Add the parent directory to sys.path to allow imports from protein_utils
# This makes the script runnable both as `python scripts/prepare_dataset.py` from root
# and potentially if `scripts` is in PYTHONPATH.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from protein_utils import format_sequence_with_spaces
except ImportError:
    # Fallback for scenarios where the script might be run in a way that
    # protein_utils is directly in PYTHONPATH or the CWD is the project root.
    from protein_utils import format_sequence_with_spaces


def main():
    """
    Main function to parse arguments, process input lines, and print formatted sequences.
    """
    parser = argparse.ArgumentParser(
        description="Converts protein sequences to space-delimited, token-wrapped format."
    )
    parser.add_argument(
        "input_file",
        nargs="?",  # Makes the argument optional
        type=str,
        default=None, # Default to None if not provided
        help="Path to the input file. If not provided, reads from stdin."
    )
    args = parser.parse_args()

    input_source = None
    try:
        if args.input_file:
            input_source = open(args.input_file, 'r')
        else:
            input_source = sys.stdin

        for line in input_source:
            # format_sequence_with_spaces already handles \n, \r, and <|endoftext|>
            # It also adds spaces after each alphabet character.
            # The original line might have leading/trailing whitespace,
            # format_sequence_with_spaces does not strip arbitrary leading/trailing
            # whitespace from the whole line, only specific characters.
            # However, the description implies "one sequence per line".
            # Let's strip the line first to be safe before processing.
            stripped_line = line.strip()
            if not stripped_line: # Skip empty lines
                continue

            processed_line = format_sequence_with_spaces(stripped_line)
            
            # The processed_line will have a trailing space if the last char was an alphabet.
            # Example: "ABC" -> "A B C ".
            # We should strip this trailing space before wrapping with tokens,
            # or ensure the tokens are directly adjacent to the content.
            # Let's strip it:
            final_processed_sequence = processed_line.strip()

            # Only print if the final processed sequence is not empty
            # (e.g. if input was just "<|endoftext|>" or "\n")
            if final_processed_sequence:
                 print(f"<|endoftext|>{final_processed_sequence}<|endoftext|>")

    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if args.input_file and input_source:
            input_source.close()

if __name__ == "__main__":
    main()
