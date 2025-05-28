!pip install -q transformers

from transformers import Trainer, GPT2LMHeadModel, AutoConfig, TrainingArguments, \
     DataCollatorForLanguageModeling, GPT2Tokenizer, TextDataset
from google.colab import drive

drive.mount('/content/drive')  # Mount Google drive

tokenizer = GPT2Tokenizer.from_pretrained('/content/drive/MyDrive/GPT2', 
     additional_special_tokens=["<|endoftext|>"])

tokenized_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='/content/drive/MyDrive/GPT2/dataset.fas',
    block_size=384
)

config = AutoConfig.from_pretrained("gpt2")
model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(tokenizer))  # account for any added tokens

args = TrainingArguments(  # Setup training hyperparameters
    output_dir='/content/drive/MyDrive/GPT2/model',
    per_device_train_batch_size=8, # decrease batch size if low RAM
    num_train_epochs=3, save_steps=3000)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer( model=model, tokenizer=tokenizer, args=args, 
    data_collator=data_collator, train_dataset=tokenized_dataset)

trainer.train()
trainer.save_model('/content/drive/MyDrive/GPT2/model')
