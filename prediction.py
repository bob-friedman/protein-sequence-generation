!pip install -q transformers

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from google.colab import drive

drive.mount('/content/drive')  # Mount google drive

sequence = 'MFVFLVLLPLVSSQCVNLRTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSIQDLFLPFFSNV'

# initialize tokenizer and model from pretrained GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/GPT2/model')
model = GPT2LMHeadModel.from_pretrained('/content/drive/MyDrive/GPT2/model')

input = tokenizer.encode(sequence, return_tensors='pt')
output = model.generate(input, max_length=10, do_sample=True)
print(output)
text = tokenizer.decode(output[0], skip_special_tokens=True)
print(text)  # print predicted protein sequence based on variable 'sequence'
