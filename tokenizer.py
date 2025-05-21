!pip install -q tokenizers

from tokenizers import ByteLevelBPETokenizer, AddedToken
# from tokenizers import Tokenizer # For loading, if needed for verification later

from google.colab import drive  # mount Google drive
drive.mount('/content/drive')

# Define special tokens
# Using AddedToken to ensure they are treated as special tokens during tokenization if needed,
# though for BPE, their presence in `special_tokens` list during train is key.
eos_bos_token = AddedToken("<|endoftext|>", single_word=True, special=True)
pad_token = AddedToken("[PAD]", single_word=True, special=True)

tokenizer = ByteLevelBPETokenizer()  # Initialize tokenizer

# Train the tokenizer
tokenizer.train(
    files='/content/drive/MyDrive/GPT2/dataset.fas',
    min_frequency=5,
    special_tokens=[str(eos_bos_token), str(pad_token)] # Convert AddedToken to str for this list
)

# Set the pad token on the tokenizer instance (important for padding utilities)
tokenizer.pad_token = str(pad_token) # Use the string representation

# Save the tokenizer to a single JSON file
tokenizer.save('/content/drive/MyDrive/GPT2/protein_tokenizer.json', pretty=True)

print("Tokenizer trained and saved to /content/drive/MyDrive/GPT2/protein_tokenizer.json")
# To demonstrate loading and getting token IDs (optional, for verification)
# loaded_tokenizer = Tokenizer.from_file('/content/drive/MyDrive/GPT2/protein_tokenizer.json')
# print(f"PAD token ID: {loaded_tokenizer.token_to_id(str(pad_token))}")
# print(f"EOS/BOS token ID: {loaded_tokenizer.token_to_id(str(eos_bos_token))}")
