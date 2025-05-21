"""
Deduplicates sequences from a Fasta file based on their MD5 hash.

This script reads a Fasta file, calculates the MD5 hash of each sequence
(after removing all whitespace), and prints only the unique sequences
to standard output. If multiple sequences have the same content, only the
first one encountered (based on its header and sequence) is printed.

Usage:
  python deduplicate_fasta.py <input_fasta_file>
"""
import argparse
import hashlib
import sys

def parse_fasta(file_handle):
    """
    Parses a Fasta file and yields (header, sequence) tuples.

    Args:
        file_handle: An open file object for the Fasta file.

    Yields:
        A tuple (header, sequence) where header is the string without '>',
        and sequence is the potentially multi-line sequence string.
    """
    header = None
    sequence_lines = []
    for line in file_handle:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(sequence_lines)
            header = line[1:]
            sequence_lines = []
        elif header is not None: # Only collect sequence lines if a header has been found
            sequence_lines.append(line)
    
    # Yield the last sequence in the file
    if header is not None:
        yield header, "".join(sequence_lines)

def main():
    """
    Main function to parse arguments, process Fasta file, and print unique sequences.
    """
    parser = argparse.ArgumentParser(
        description="Deduplicate a Fasta file based on sequence MD5 hashes."
    )
    parser.add_argument(
        "input_fasta_file",
        help="Path to the input Fasta file."
    )
    args = parser.parse_args()

    seen_digests = set()

    try:
        with open(args.input_fasta_file, 'r') as f_in:
            for header, sequence_original in parse_fasta(f_in):
                # For hashing, remove all whitespace from the sequence,
                # similar to Perl's s/\s//g
                sequence_for_hashing = "".join(sequence_original.split())

                if not sequence_for_hashing: # Skip if sequence is empty after cleaning
                    # If original sequence was also empty or just whitespace,
                    # we might want to print its header if no other identical empty sequence was seen.
                    # However, typical Fasta has non-empty sequences.
                    # For now, let's assume empty sequences (after cleaning) don't get printed
                    # unless we define specific behavior for them.
                    # The Perl script would hash an empty string in this case.
                    pass # Or handle as per specific requirement for empty sequences

                md5_hash = hashlib.md5(sequence_for_hashing.encode('utf-8')).hexdigest()

                # Only process and print if the sequence (after whitespace removal) is not empty
                if sequence_for_hashing:
                    if md5_hash not in seen_digests:
                        seen_digests.add(md5_hash)
                        sys.stdout.write(f">{header}\n")
                        # Print the original sequence, which might be multi-line
                        # The current `parse_fasta` joins all sequence lines into one string.
                        # So, we print that single string, followed by a newline.
                        sys.stdout.write(f"{sequence_original}\n")
                # If sequence_for_hashing is empty, we effectively skip printing this entry.

    except FileNotFoundError:
        print(f"Error: File not found: {args.input_fasta_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
