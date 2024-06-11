import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Function to load JSONL data from a folder
def load_jsonl_from_folder(folder_path, encoding='utf-8'):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for line in f:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_path}: {e}")
    return data
    
# Load your data
train_data = load_jsonl_from_folder('extracted_data/')

print("Data sample:", train_data[0])

# Create Dataset objects
train_dataset = Dataset.from_list(train_data)

# Combine into a DatasetDict if needed
datasets = DatasetDict({
    'train': train_dataset,
})

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['doc'], padding="max_length", truncation=True)

# Map the tokenize function to the dataset
tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Add labels for language modeling (shift the input ids by one position)
def add_labels(examples):
    examples['labels'] = examples['input_ids'].copy()
    return examples

# Apply the label function
tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Set the format for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# Train the model
trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")