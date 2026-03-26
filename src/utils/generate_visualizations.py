import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import warnings

warnings.filterwarnings('ignore')

def load_data_and_models():
    print("Loading test data and models...")
    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Load TF-IDF Baseline
    tfidf_vectorizer = joblib.load("src/models/saved_models/tfidf_vectorizer.pkl")
    tfidf_classifier = joblib.load("src/models/saved_models/tfidf_classifier.pkl")
    
    # Load BERT
    model_path = "src/models/saved_models/bert_finetuned"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertForSequenceClassification.from_pretrained(model_path)
    label_encoder = joblib.load("src/models/saved_models/label_encoder.pkl")
    
    return test_df, tfidf_vectorizer, tfidf_classifier, tokenizer, bert_model, label_encoder

def get_predictions(test_df, tfidf_vectorizer, tfidf_classifier, tokenizer, bert_model, label_encoder):
    print("Generating predictions for the test dataset. This may take a minute...")
    
    # TF-IDF Predictions
    X_test_vec = tfidf_vectorizer.transform(test_df['text'].tolist())
    tfidf_preds = tfidf_classifier.predict(X_test_vec)
    
    # BERT Predictions
    bert_preds = []
    bert_model.eval()
    
    # Process in small batches for efficiency
    texts = test_df['text'].tolist()
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=64)
            outputs = bert_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            bert_preds.extend(preds.cpu().numpy())
            
    # Decode BERT predictions back to string labels
    bert_preds_decoded = label_encoder.inverse_transform(bert_preds)
    
    return test_df['intent_name'].tolist(), tfidf_preds, bert_preds_decoded, label_encoder.classes_

def plot_accuracy_comparison(y_true, tfidf_preds, bert_preds):
    print("Plotting Accuracy Comparison Bar Chart...")
    tfidf_acc = accuracy_score(y_true, tfidf_preds) * 100
    bert_acc = accuracy_score(y_true, bert_preds) * 100
    
    models = ['TF-IDF + Logistic Regression', 'Fine-Tuned BERT']
    accuracies = [tfidf_acc, bert_acc]
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=models, y=accuracies, palette=['#6baed6', '#3182bd'])
    
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, pad=15)
    
    # Add percentage labels on top of bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12, fontweight='bold')
        
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, preds, labels, model_name, filename):
    print(f"Plotting Confusion Matrix for {model_name}...")
    cm = confusion_matrix(y_true, preds, labels=labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title(f'Confusion Matrix: {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Intent', fontsize=12)
    plt.xlabel('Predicted Intent', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"docs/{filename}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    test_df, tfidf_vec, tfidf_clf, tokenizer, bert_model, label_enc = load_data_and_models()
    y_true, tfidf_preds, bert_preds, labels = get_predictions(
        test_df, tfidf_vec, tfidf_clf, tokenizer, bert_model, label_enc
    )
    
    plot_accuracy_comparison(y_true, tfidf_preds, bert_preds)
    plot_confusion_matrix(y_true, bert_preds, labels, "Fine-Tuned BERT", "bert_confusion_matrix")
    plot_confusion_matrix(y_true, tfidf_preds, labels, "TF-IDF Baseline", "tfidf_confusion_matrix")
    
    print("\n✅ Success! Visualizations saved in the 'docs/' folder:")
    print("   - docs/accuracy_comparison.png")
    print("   - docs/bert_confusion_matrix.png")
    print("   - docs/tfidf_confusion_matrix.png")
