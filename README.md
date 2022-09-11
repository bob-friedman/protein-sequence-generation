# Generative Model for Protein Sequences

## Redundancy in Sequence Data
The common format of protein sequences is Fasta. These files are available from sources such as NCBI, EMBL, Ensembl, and UniProt. UniProt has different versions of sequence data where each version has a different level of sequence redundancy removed. UniProt labels these versions as UniRef datasets, such as UniRef50. They use software and algorithms to achieve this result, such as the MMseqs2 algorithm. This kind of filter of the data may lead to better results from the Generative Model. However, it is possible to write scripts in the Perl language to filter by simpler criteria.

An unsupported Perl script (fasta_md5sum.pl) is in the unsupported directory of this repository. It removes redundant and identical sequences in a Fasta formatted file. Identical sequences are defined as having the same amino acid sequence and the same length. This code uses the MD5 hash algorithm to find these redundant sequences. The resulting output must be validated against the original file.

## Protein Sequence Data Format
### Method #1
The Python v3 code is based on use of the GPT-2 model and the Huggingface transformers API. The code requires an input file of data, such as protein sequences. The sequence may be in the following format, where each sequence is flanked by <|endoftext|> tokens as described by the GPT-2 model, and each sequence and its tokens are along a single line in the file:

<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>\
<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>\
<|endoftext|>MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLP<|endoftext|>

### Method #2
The above format considers each protein sequence as a single unit. To consider each amino acid of the sequence as a single unit, like a word, then the format includes spaces as follows:

<|endoftext|>M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>\
<|endoftext|>M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>\
<|endoftext|>M F V F L V L L P L V S S Q C V N L T T R T Q L P P A Y T <|endoftext|>

With this method, the tokenizer (byte pair encoding) should match each single amino acid to a separate token, in contrast with Method #1 where each token frequently matches to a larger subsequence of amino acids. During testing, Method #2 is resulting in the generation of sequences that better resemble a prediction of sequence data. Moreover, during the prediction step (prediction.py) under Method #2, the input sequence should be separated by spaces, like so:\
$sequence = 'M F V F L V L L P L V S S Q C V N L T T R T'

Below is a script in Perl that reads standard input (STDIN) in the above sequence formats and then adds a space after each letter (converts the above sequence as shown in Method #1 to the sequence in Method #2). The resulting output should be verified that it appears correctly (tested in Windows).
```
while(<>){
	$_=~s/\n//g;
	$_=~s/<\|endoftext\|>//g;
	$_=~s/(?<=[a-z])(?=[a-z])/ /ig;
	$_=~s/\r//g;
	print "<|endoftext|>",$_,"<|endoftext|>\n";
}
```

## Running the Model
The Python code files may be run in Google Colab. They are currectly formatted for use of Google Drive, but the code may be changed for a local installation or a method of remote access. In the current code, the input file is named dataset.fas. Also, for predicting protein sequences by a generative model, the max_length parameter value may be changed in prediction.py. There is another important parameter in tokenizer.py: min_frequency. This parameter value may also be changed during testing of the code.

The prediction step in prediction.py requires sequence data. In the code its value is assigned to the variable "sequence". This is the input value for prompting the generative model to create sequence data. This value may be changed in the code for prediction.py.

## License
These code files are for examples only on creating a generative model for protein sequence data. They may be extended upon as described in the LICENSE file (Apache License v2.0). This license is identical to that of the Huggingface transformers API. It is an open-source and permissive software license.

## Credits
The concept is based on the ProtGPT2 research study and published article: \
https://www.nature.com/articles/s41467-022-32007-7 (publication) \
https://huggingface.co/nferruz/ProtGPT2 (files of generative model of protein sequences)
