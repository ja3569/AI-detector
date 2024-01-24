import os
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Read texts from folders
def load_texts(folder):
    texts = []
    for filename in os.listdir(folder):
        try:
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
        except UnicodeDecodeError:
            with open(os.path.join(folder, filename), 'r', encoding='latin1') as file:
                texts.append(file.read())
    return texts


# Prompt for folder names
ai_folder = input("Enter the name of the folder for AI-generated texts (within 'gptData'): ")
human_folder = input("Enter the name of the folder for human-written texts (within 'humanData'): ")

# Load data
ai_texts = load_texts(f'./gptData/{ai_folder}')
human_texts = load_texts(f'./humanData/{human_folder}')
print(f"Loaded {len(ai_texts)} AI-generated texts and {len(human_texts)} human-written texts.")

all_texts = ai_texts + human_texts
labels = [1] * len(ai_texts) + [0] * len(human_texts)  # 1 for AI, 0 for human

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, labels, test_size=0.2)
print("Data split into training and test sets.")

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
print("Texts tokenized.")

# Convert to torch tensors
train_seq = torch.tensor(train_encodings['input_ids'])
train_mask = torch.tensor(train_encodings['attention_mask'])
train_y = torch.tensor(train_labels)

test_seq = torch.tensor(test_encodings['input_ids'])
test_mask = torch.tensor(test_encodings['attention_mask'])
test_y = torch.tensor(test_labels)

# DataLoader
batch_size = 32
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_seq, test_mask, test_y)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
print("Data loaders prepared.")

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
print("BERT model loaded.")

# Training
print("Starting model training...")
for epoch in range(3):
    model.train()
    for batch in tqdm(train_dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
print("Model training completed!")

# Save model
model.save_pretrained('trained_bert_model')
tokenizer.save_pretrained('trained_bert_model')
print("Model saved.")
