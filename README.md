# Generative Model for Protein Sequences

## Finding the Sequence Data
A common format of protein sequences is "Fasta". In general, protein sequence files are available from the major genetic database sites, such as at NCBI, EMBL, Ensembl, and UniProt. The UniProt site is of particular interest since it has sequence data with sequence redundancy removed at various levels. UniProt labels these versions as UniRef datasets, such as UniRef50. They use software to process sequence-level redundancy, such as the MMseqs2 algorithm. This kind of filter of the data may lead to better results where using a generative model of sequence construction. However, it is also possible to write scripts, such as in the Perl language, to filter the data by simpler criteria.

A Python script, `scripts/deduplicate_fasta.py`, is available to remove redundant sequences from a Fasta formatted file. It uses an MD5 hash algorithm to identify identical sequences (defined as having the same amino acid sequence and length after whitespace removal). The script takes a Fasta file as input and prints the unique sequences to standard output. For example: `python scripts/deduplicate_fasta.py input.fasta > output_unique.fasta`.

## Protein Sequence Data Format
### Method #1
The Python v3 code in this repository is based on use of the GPT-2 model and the Huggingface transformers API. The code requires an input data file of protein sequences. If the sequence data is originally in the Fasta format, then it would be converted to the format below before running the Python code, where each sequence is flanked by <|endoftext|> tokens as described by documentation on the GPT-2 model at Huggingface. Each sequence and its tokens are in a single line in a file:

<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>\
<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>\
<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>

### Method #2
The above format considers each protein sequence as a single unit. To instead consider each amino acid of the sequence as the single unit, like a word, then the format includes spaces as follows:

<|endoftext|>M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>\
<|endoftext|>M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>\
<|endoftext|>M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>

With this method, the tokenizer, based on byte pair encoding, should match each single amino acid to a separate token, in contrast with Method #1 where each token usually matches to a subsequence. During testing, Method #2 is resulting in the generation of sequences that more closely resemble a prediction of sequence data. Moreover, during the prediction step (see prediction.py) under Method #2, the input sequence should be separated by spaces, like so:\
$sequence = 'M F V F L V L L P L V S S Q C V N L T T R T'

To convert sequence data from Method #1 (condensed) to Method #2 (space-delimited) for use with the tokenizer and model, a Python script `scripts/prepare_dataset.py` is provided. This script reads sequences (one per line, potentially with or without `<|endoftext|>` tokens initially) from an input file (or standard input) and outputs each sequence in Method #2 format, ready for `dataset.fas`. Each processed sequence will have spaces inserted between amino acids and will be wrapped with `<|endoftext|>` tokens on a new line.
Example usage: `python scripts/prepare_dataset.py input_sequences.txt > dataset.fas`

Internally, this script uses a utility function `format_sequence_with_spaces` located in `protein_utils.py`. This function handles the core logic of removing unwanted characters (newlines, `<|endoftext|>` tokens) and inserting spaces between letters. The `prediction.py` script also utilizes this function to correctly format its input sequence.

## Running the Model
The Python code files may be run in Google Colab. The code is currently for use with Google Drive, but the code may be changed for running on a local server or a different method of remote access. In the current code, the input file is named dataset.fas.

For changing the expected length of prediction of a protein sequence, the max_length parameter value may be modified in prediction.py. There are other impactful parameters in the code, such as in tokenizer.py. These parameter values may be changed during testing. There is information on the parameters at Huggingface.

Lastly, the prediction step in prediction.py requires a partial sequence to serve as the prompt to begin the prediction. In the code, the partial sequence is assigned to the variable "sequence". This input sequence prompts the generative model for generating the tokens and protein sequence.

The code should be inspected and run in the following order, including any input file requirements and modifying of parameters: tokenizer.py, train.py, prediction.py. There are further details on the method in a manuscript, but it is in review and not yet published.

## License
These code files are for examples on creating a generative model for protein sequence data. They may be expanded upon as described in the LICENSE file (Apache License v2.0). This license is identical to the Huggingface transformers API. It is an open-source and permissive software license.

## Credits
The concept is based on the ProtGPT2 research study and published article:\
https://www.nature.com/articles/s41467-022-32007-7 (ProtGPT2 publication)\
https://huggingface.co/nferruz/ProtGPT2 (ProtGPT2 files for generative model)\
https://openai.com/blog/better-language-models (GPT-2)\
https://huggingface.co (Huggingface API)
