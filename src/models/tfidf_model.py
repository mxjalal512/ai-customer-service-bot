import pandas as pd
import time
import tracemalloc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import os

def run_tfidf_baseline():
    print("=== Running TF-IDF + Logistic Regression Baseline ===")
    
    # 1. Load the processed datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df['text'].tolist()
    y_train = train_df['intent_name'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['intent_name'].tolist()

    # 2. Text Preprocessing & Feature Extraction
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, lowercase=True, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Model Training (with time tracking and memory tracking for tradeoff analysis)
    print("Training Logistic Regression model...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)

    tracemalloc.start()
    start_time = time.time()

    classifier.fit(X_train_vec, y_train)

    end_time = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    training_time = end_time - start_time
    peak_mem_mb = peak_mem / 1024 / 1024

    # 4. Evaluation
    print("Evaluating model on test data...")
    y_pred = classifier.predict(X_test_vec)
    
    # Calculate evaluation criteria
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print("\n--- TF-IDF Results ---")
    print(f"Training Time:      {training_time:.2f} seconds")
    print(f"Peak Memory Usage:  {peak_mem_mb:.2f} MB")
    print(f"Accuracy:           {accuracy * 100:.2f}%")
    print(f"Precision:          {precision * 100:.2f}%")
    print(f"Recall:             {recall * 100:.2f}%")
    print(f"F1-Score:           {f1 * 100:.2f}%")

    print("\n--- Performance Tradeoff Summary ---")
    print(f"{'Model':<30} {'Accuracy':<12} {'Training Time':<20} {'Memory Usage':<15}")
    print("-" * 77)
    print(f"{'TF-IDF + Logistic Regression':<30} {f'{accuracy * 100:.2f}%':<12} {f'{training_time:.2f} seconds':<20} {f'{peak_mem_mb:.2f} MB':<15}")

    # Save the model and vectorizer for our unique tests later
    os.makedirs("src/models/saved_models", exist_ok=True)
    joblib.dump(vectorizer, "src/models/saved_models/tfidf_vectorizer.pkl")
    joblib.dump(classifier, "src/models/saved_models/tfidf_classifier.pkl")
    print("\nModel saved to src/models/saved_models/")

if __name__ == "__main__":
    run_tfidf_baseline()