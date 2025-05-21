!pip install -q transformers

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from google.colab import drive
from protein_utils import format_sequence_with_spaces

drive.mount('/content/drive')  # Mount google drive

sequence = 'MFVFLVLLPLVSSQCVNLRTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSIQDLFLPFFSNV'
processed_sequence = format_sequence_with_spaces(sequence)
final_sequence_for_encoding = f"<|endoftext|>{processed_sequence}<|endoftext|>"

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/GPT2/model')
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/GPT2/model')

input = tokenizer.encode(final_sequence_for_encoding, return_tensors='pt')
output = model.generate(input, max_length=10, do_sample=True)
print(output)
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)  # print predicted protein sequence based on variable 'sequence'
