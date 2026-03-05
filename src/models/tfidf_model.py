# src/models/tfidf_model.py
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import joblib
import os

def run_tfidf_baseline():
    print("=== Running TF-IDF + Logistic Regression Baseline ===")
    
    # 1. Load the processed datasets
    # Using the exact same dataset for fair comparison
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df['text'].tolist()
    y_train = train_df['intent_name'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['intent_name'].tolist()

    # 2. Text Preprocessing & Feature Extraction
    # Convert text to numerical vectors using TF-IDF
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000, lowercase=True, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Model Training (with time tracking for tradeoff analysis)
    print("Training Logistic Regression model...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    start_time = time.time()
    classifier.fit(X_train_vec, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time

    # 4. Evaluation
    print("Evaluating model on test data...")
    y_pred = classifier.predict(X_test_vec)
    
    # Calculate evaluation criteria
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    print("\n--- TF-IDF Results ---")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    
    # Save the model and vectorizer for our unique tests later
    os.makedirs("src/models/saved_models", exist_ok=True)
    joblib.dump(vectorizer, "src/models/saved_models/tfidf_vectorizer.pkl")
    joblib.dump(classifier, "src/models/saved_models/tfidf_classifier.pkl")
    print("Model saved to src/models/saved_models/")

if __name__ == "__main__":
    run_tfidf_baseline()