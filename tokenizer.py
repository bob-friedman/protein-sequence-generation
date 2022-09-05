!pip install -q tokenizers

from tokenizers import ByteLevelBPETokenizer

from google.colab import drive  # mount Google drive
drive.mount('/content/drive')

tokenizer = ByteLevelBPETokenizer()  # Initialize tokenizer
tokenizer.train(files = 
   '/content/drive/MyDrive/GPT2/dataset.fas', min_frequency=5, special_tokens=["<|endoftext|>"])

tokenizer.model.save("/content/drive/MyDrive/GPT2")  # Save model to disk
