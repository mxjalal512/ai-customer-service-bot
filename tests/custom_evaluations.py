import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

# Suppress warnings for cleaner output during presentation
warnings.filterwarnings('ignore')

def load_saved_models():
    print("Loading models for custom evaluation...\n")
    
    # 1. Load TF-IDF Baseline
    tfidf_vectorizer = joblib.load("src/models/saved_models/tfidf_vectorizer.pkl")
    tfidf_classifier = joblib.load("src/models/saved_models/tfidf_classifier.pkl")
    
    # 2. Load BERT and Label Encoder
    model_path = "src/models/saved_models/bert_finetuned"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertForSequenceClassification.from_pretrained(model_path)
    label_encoder = joblib.load("src/models/saved_models/label_encoder.pkl")
    
    return tfidf_vectorizer, tfidf_classifier, tokenizer, bert_model, label_encoder

def predict_tfidf(text, vectorizer, classifier):
    vec = vectorizer.transform([text])
    prediction = classifier.predict(vec)
    return prediction[0]

def predict_bert(text, tokenizer, model, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    
    # Ensure no gradient calculation is needed for inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Decode the integer back to the string label
    predicted_intent = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_intent

def run_unique_tests():
    vectorizer, classifier, tokenizer, bert_model, label_encoder = load_saved_models()
    
    # Defining the unique custom tests
    # IDEA 1: Paraphrase Robustness
    # IDEA 2: Ambiguity Handling
    # IDEA 3: Out-of-Scope Detection
    test_cases = [
        {
            "category": "1. Paraphrase Robustness Testing",
            "description": "Demonstrates BERT understands meaning, not just keywords.",
            "original": "I want to pay my bill.",
            "paraphrased": ["Let me settle my bill.", "I need to make a payment."],
            "expected_intent": "pay_bill"
        },
        {
             "category": "2. Ambiguity Handling Test",
            "description": "Shows how BERT handles vague context better.",
            "queries": ["Something is wrong with my account.", "My account is not working."],
            "expected_intent": "Context Dependent (e.g., account_blocked or update_password)"
        },
        },
        {
            "category": "3. Out-of-Scope Detection",
            "description": "Tests system robustness against irrelevant queries.",
            "queries": ["Tell me a joke.", "What is the weather like today?"],
            "expected_intent": "oos"
        }
    ]
    
    print("="*60)
    print("CUSTOM EVALUATION TESTS: TF-IDF vs BERT")
    print("="*60)
    
    # 1. Paraphrase Test
    print(f"\n{test_cases[0]['category'].upper()}")
    print(f"Goal: {test_cases[0]['description']}\n")
    
    all_paraphrase_queries = [test_cases[0]['original']] + test_cases[0]['paraphrased']
    print(f"{'Input Query':<35} | {'TF-IDF Prediction':<20} | {'BERT Prediction':<20}")
    print("-" * 80)
    for query in all_paraphrase_queries:
        tfidf_pred = predict_tfidf(query, vectorizer, classifier)
        bert_pred = predict_bert(query, tokenizer, bert_model, label_encoder)
        print(f"{query:<35} | {tfidf_pred:<20} | {bert_pred:<20}")

    # 2. Ambiguity Test
    print(f"\n\n{test_cases[1]['category'].upper()}")
    print(f"Goal: {test_cases[1]['description']}\n")
    print(f"{'Input Query':<35} | {'TF-IDF Prediction':<20} | {'BERT Prediction':<20}")
    print("-" * 80)
    for query in test_cases[1]['queries']:
        tfidf_pred = predict_tfidf(query, vectorizer, classifier)
        bert_pred = predict_bert(query, tokenizer, bert_model, label_encoder)
        print(f"{query:<35} | {tfidf_pred:<20} | {bert_pred:<20}")

    # 3. Out-of-Scope Test
    print(f"\n\n{test_cases[2]['category'].upper()}")
    print(f"Goal: {test_cases[2]['description']}\n")
    print(f"{'Input Query':<35} | {'TF-IDF Prediction':<20} | {'BERT Prediction':<20}")
    print("-" * 80)
    for query in test_cases[2]['queries']:
        tfidf_pred = predict_tfidf(query, vectorizer, classifier)
        bert_pred = predict_bert(query, tokenizer, bert_model, label_encoder)
        print(f"{query:<35} | {tfidf_pred:<20} | {bert_pred:<20}")
        
    print("\n" + "="*60)
    print("Test execution complete. Copy these results into your final report!")

if __name__ == "__main__":
    run_unique_tests()
