# Generative Model for Protein Sequences

The Python v3 code is based on use of the GPT-2 model and the Huggingface transformers API. The code requires an input file of data, such as protein sequences. The sequence may be in the following format, where each sequence is flanked by <|endoftext|> tokens as described by the GPT-2 model:

<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>
<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>
<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>

The Python code files may be run in Google Colab. They are currectly formatted for use of Google Drive, but the code may be changed for a local installation or a method of remote access.

These code files are for examples only on creating a generative model for protein sequence data. They may be extended upon as described in the LICENSE file (Apache License v2.0). This license is identical to that of the Huggingface transformers API. It is an open-source and permissive software license.

The concept is based on the ProtGPT2 research study and published article: \
https://www.nature.com/articles/s41467-022-32007-7 (publication) \
https://huggingface.co/nferruz/ProtGPT2 (files of generative model of protein sequences)
