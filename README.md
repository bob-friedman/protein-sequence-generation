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

## Important Note on Script Versions

The older scripts for model training and prediction, namely `train.py` and `prediction.py`, have been archived into the `archived_scripts/` directory.

The current, recommended scripts to use are the PyTorch-based versions:
-   `train_pytorch.py` for training the model.
-   `predict_pytorch.py` for generating sequences with a trained model.

Please refer to the instructions below for using these updated scripts.

## Running the Model (PyTorch Transformer)

This project uses a custom PyTorch-based Transformer model for generating protein sequences. The workflow involves three main scripts: `tokenizer.py` (for preparing the tokenizer), `train_pytorch.py` (for training the model), and `predict_pytorch.py` (for generating sequences).

### 1. Prepare the Dataset (`dataset.fas`)

The model expects input data in a file typically named `dataset.fas`. This file should contain one protein sequence per line, with amino acids separated by spaces, and each sequence wrapped with `<|endoftext|>` tokens. This is referred to as "Method #2" in the "Protein Sequence Data Format" section.

Example line in `dataset.fas`:
`<|endoftext|> M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>`

You can use the `scripts/prepare_dataset.py` script to convert sequences from a condensed format (Method #1) or plain sequences into the required `dataset.fas` format.
Example: `python scripts/prepare_dataset.py your_raw_sequences.txt > dataset.fas`

### 2. Train the Tokenizer

A Byte-Pair Encoding (BPE) tokenizer is trained on your `dataset.fas`. The `tokenizer.py` script handles this.
```bash
python tokenizer.py
```
This script will read your dataset (ensure the path within `tokenizer.py` points to your `dataset.fas`, e.g., `/content/drive/MyDrive/GPT2/dataset.fas` or modify the script for a local path) and save the trained tokenizer as `protein_tokenizer.json` (e.g., in `/content/drive/MyDrive/GPT2/protein_tokenizer.json`). This path will be needed for training and prediction.
The tokenizer uses `<|endoftext|>` as a special token for beginning/end of sequence and `[PAD]` as the padding token.

### 3. Train the PyTorch Transformer Model

Use the `train_pytorch.py` script to train the custom Transformer model.
You'll need to provide paths to your tokenizer and dataset, and specify model/training hyperparameters.

Example usage:
```bash
python train_pytorch.py \
    --tokenizer_path /path/to/your/protein_tokenizer.json \
    --dataset_path /path/to/your/dataset.fas \
    --model_output_dir ./trained_protein_transformer \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --block_size 128 \
    --d_model 256 \
    --nhead 4 \
    --d_hid 512 \
    --nlayers 3 \
    --dropout 0.1 \
    --save_every 1
```
Adjust paths and hyperparameters as needed. Model checkpoints will be saved in the `--model_output_dir`.

### 4. Generate Sequences

Use `predict_pytorch.py` to generate sequences using a trained model checkpoint.
Provide the tokenizer, model checkpoint, model hyperparameters (matching the trained model), and a prompt sequence.

Example usage:
```bash
python predict_pytorch.py \
    --tokenizer_path /path/to/your/protein_tokenizer.json \
    --model_checkpoint_path ./trained_protein_transformer/model_epoch_10.pt \
    --prompt_sequence "MFVFL" \
    --max_length 100 \
    --block_size 128 \
    --d_model 256 \
    --nhead 4 \
    --d_hid 512 \
    --nlayers 3 \
    --dropout 0.1
```
This will print the generated sequence to the console.


## Testing and Feedback

The project has recently been updated to a new PyTorch-based Transformer pipeline. We encourage users to:
*   Test the new `tokenizer.py`, `train_pytorch.py`, and `predict_pytorch.py` scripts with their own data and environments.
*   Report any issues, bugs, or usability concerns they encounter.
*   Share feedback on the new model's performance and the overall workflow.

Your feedback is valuable for improving the project! Please raise an issue in the repository for any bugs or suggestions.

## Utility and Example Notebooks

This section highlights notebooks that provide utility functions or demonstrate specific examples related to protein engineering tasks.

### TCR Structure Prediction with ImmuneBuilder (`TCRBuilder2_mamba_setup.ipynb`)

-   **Purpose:** This [Jupyter Colab notebook](notebook/TCRBuilder2_mamba_setup.ipynb) demonstrates how to predict the 3D structure of a T-Cell Receptor (TCR) using the `ImmuneBuilder` library, specifically its `TCRBuilder2` module.
-   **Environment:** It is designed for Google Colab and includes comprehensive environment setup steps using `mamba` (via `condacolab`) and `pip` to install necessary dependencies such as `ImmuneBuilder`, `ANARCI`, `OpenMM`, `PDBFixer`, and `py3Dmol`.
-   **Workflow:**
    1.  Takes TCR alpha and beta chain sequences as input.
    2.  Uses `ANARCI` for sequence annotation.
    3.  Employs `TCRBuilder2` to predict the 3D structure.
    4.  Saves the resulting structure as a PDB file.
    5.  Visualizes the predicted structure directly within the notebook using `py3Dmol`.
    6.  Allows for downloading the generated PDB file.
-   **Usage:** Open the notebook in Google Colab and run the cells sequentially. You can modify the input sequences (`sequence_1`, `sequence_2`) to predict structures for different TCRs.
-   **Location:** [notebook/TCRBuilder2_mamba_setup.ipynb](notebook/TCRBuilder2_mamba_setup.ipynb)
-   **Original Source:** Abanades, B.; Wong, W.K.; Boyles, F.; Georges, G.; Bujotzek, A.; Deane, C.M. ImmuneBuilder: Deep-Learning models for predicting the structures of immune proteins. Commun. Biol. 2023, 6, 575.

## Acknowledgements

Portions of this project were developed with the assistance of Google AI (Jules).
The underlying concepts for protein sequence generation draw inspiration from the original ProtGPT2 work and the Hugging Face Transformers library.

## Citation
If you use the software in this repository, please cite the following research article:
[https://www.mdpi.com/2073-8994/14/11/2274](https://www.mdpi.com/2073-8994/14/11/2274)

## License
These code files are for examples on creating a generative model for protein sequence data. They may be expanded upon as described in the LICENSE file (Apache License v2.0). This license is identical to the Huggingface transformers API. It is an open-source and permissive software license.

## Credits
The concept is based on the ProtGPT2 research study and published article:\
https://www.nature.com/articles/s41467-022-32007-7 (ProtGPT2 publication)\
https://huggingface.co/nferruz/ProtGPT2 (ProtGPT2 files for generative model)\
https://openai.com/blog/better-language-models (GPT-2)\
https://huggingface.co (Huggingface API)
