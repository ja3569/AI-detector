import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('trained_bert_model')
tokenizer = BertTokenizer.from_pretrained('trained_bert_model')

def predict(text, model, tokenizer):
    """ Prediction """
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    outputs = model(**inputs)
    likelihood = torch.nn.functional.softmax(outputs.logits, dim=1)[0][1].item()  # Probability of being AI-generated
    ai_detection = "Y" if likelihood > 0.5 else "N"
    return likelihood, ai_detection

def process_folder(folder_name, model, tokenizer):
    results = []
    for filename in os.listdir(folder_name):
        file_path = os.path.join(folder_name, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            likelihood, ai_detection = predict(text, model, tokenizer)
            results.append([filename, text, f"{likelihood:.2%}", ai_detection])

    # Create and save DataFrame
    df = pd.DataFrame(results, columns=["FID", "Text", "AI Likelihood", "AI Detection"])
    report_file = os.path.join(folder_name, "report.csv")
    df.to_csv(report_file, index=False)
    print(f"Report saved to {report_file}")

# Prompt user for folder name
folder_name = input("Enter the name of the folder to analyze: ")
process_folder(folder_name, model, tokenizer)
