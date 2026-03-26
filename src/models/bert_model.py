# src/models/bert_model.py
import pandas as pd
import time
import tracemalloc
import torch
import joblib
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# Create a custom dataset class required for PyTorch and Hugging Face Trainer
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Evaluation function for the Hugging Face Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run_bert_model():
    print("=== Running BERT Fine-Tuning Model ===")

    # 1. Load the exact same datasets
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Encode string labels to integers for BERT
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['intent_name'])
    val_labels = le.transform(val_df['intent_name'])
    test_labels = le.transform(test_df['intent_name'])

    # 2. Tokenization using pre-trained BERT tokenizer
    print("Tokenizing text data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_df['text'].tolist(), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=64)

    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)
    test_dataset = IntentDataset(test_encodings, test_labels)

    # 3. Model Initialization
    num_labels = len(le.classes_)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # Training arguments specifically chosen for a reasonable student laptop/environment
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 4. Train the Model (with time tracking and memory tracking)
    print("Starting BERT training... (This may take several minutes depending on your hardware)")

    tracemalloc.start()
    start_time = time.time()

    trainer.train()

    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    training_time = end_time - start_time
    peak_mem_mb = peak_mem / 1024 / 1024

    # 5. Evaluation
    print("Evaluating BERT on test data...")
    test_results = trainer.evaluate(test_dataset)

    print("\n--- BERT Results ---")
    print(f"Training Time:      {training_time / 60:.2f} minutes")
    print(f"Peak Memory Usage:  {peak_mem_mb:.2f} MB")
    print(f"Accuracy:           {test_results['eval_accuracy'] * 100:.2f}%")
    print(f"Precision:          {test_results['eval_precision'] * 100:.2f}%")
    print(f"Recall:             {test_results['eval_recall'] * 100:.2f}%")
    print(f"F1-Score:           {test_results['eval_f1'] * 100:.2f}%")

    print("\n--- Performance Tradeoff Summary ---")
    print(f"{'Model':<20} {'Accuracy':<12} {'Training Time':<20} {'Memory Usage':<15}")
    print("-" * 67)
    bert_acc = f"{test_results['eval_accuracy'] * 100:.2f}%"
    bert_time = f"{training_time / 60:.2f} minutes"
    bert_mem = f"{peak_mem_mb:.2f} MB"
    print(f"{'Fine-Tuned BERT':<20} {bert_acc:<12} {bert_time:<20} {bert_mem:<15}")

    # Save the model and label encoder for our unique tests
    model_save_path = "src/models/saved_models/bert_finetuned"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    joblib.dump(le, "src/models/saved_models/label_encoder.pkl")

    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    run_bert_model()
