# AI-Powered Customer Service Intent Classification

## Project Overview

Customer service platforms increasingly utilize artificial intelligence to develop conversational chatbots that automatically assist users by answering queries and resolving issues. The objective of this project is to design an AI-powered chatbot capable of automatically classifying customer queries into predefined intents, providing instant responses, and escalating complex cases. By accurately understanding user intent, digital businesses can significantly reduce operational costs and improve overall customer satisfaction.

---

## Algorithmic Comparison

To demonstrate the evolution and efficiency of natural language understanding in customer service, this project implements and compares two distinct approaches within the same experimental framework using the same dataset:

| Approach                      | Model                        | Description                                                               |
| ----------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| **Baseline (Traditional ML)** | TF-IDF + Logistic Regression | Relies on word frequency; struggles with contextual meaning               |
| **Proposed (Deep Learning)**  | Fine-tuned BERT              | Uses self-attention to capture contextual and bidirectional relationships |

> **Note:** RoBERTa was excluded from this comparison because it requires significantly more computational power and complex training, offering only marginal improvements for medium-sized datasets like the one used in this project.

---

## Dataset

The project utilizes the **CLINC150** dataset, which originally contains approximately 22,500 labeled user queries across 150 intent categories. For this implementation, the dataset was filtered down to **25 relevant customer service intent categories**, plus an "Out-of-Scope" category. Using a single structured dataset ensures a fair and controlled comparison between the baseline and proposed models.

---

## Evaluation Metrics

Beyond standard metrics (Accuracy, Precision, Recall, F1-score), this system is evaluated against four custom robustness tests to highlight the contextual power of BERT:

- **Paraphrase Robustness Testing** — Evaluates how models handle different phrasings of the same intent, demonstrating that BERT understands semantic meaning rather than just keyword matching.
- **Ambiguity Handling Test** — Assesses context resolution for vague user queries.
- **Out-of-Scope Detection** — Tests the system's robustness by identifying and labeling irrelevant queries.
- **Performance Tradeoff** — A comprehensive analysis comparing Model Accuracy, Training Time, and Memory Usage.

---

## Project Structure

```
ai-customer-service-bot/
├── data/                         # Raw and processed dataset files (ignored in version control)
├── docs/                         # Generated evaluation visualizations (Confusion Matrices, Accuracy Charts)
├── src/
│   ├── models/
│   │   ├── tfidf_model.py        # TF-IDF baseline model
│   │   └── bert_model.py         # BERT fine-tuning architecture
│   └── utils/
│       ├── data_preparation.py   # Dataset download and preprocessing
│       └── generate_visualizations.py  # Evaluation charts and plots
└── tests/
    └── custom_evaluations.py     # Paraphrase, ambiguity, and out-of-scope evaluations
```

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/mxjalal512/ai-customer-service-bot.git
   cd ai-customer-service-bot
   ```

2. Ensure **Python 3.8+** is installed.

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # macOS/Linux
   source venv/bin/activate
   # Windows
   .\venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Execute the following scripts **in order** from the root directory to reproduce the experiment:

### 1. Prepare the Data

Downloads the CLINC150 dataset and filters it to the required intent categories.

```bash
python src/utils/data_preparation.py
```

### 2. Train the TF-IDF Baseline

Trains the traditional machine learning model and logs execution time and metrics.

```bash
python src/models/tfidf_model.py
```

### 3. Train the BERT Model

Fine-tunes the BERT model on the dataset. _(This step may take several minutes depending on hardware.)_

```bash
python src/models/bert_model.py
```

### 4. Generate Visualizations

Evaluates both models and outputs the Accuracy Bar Chart and Confusion Matrices to the `docs/` folder.

```bash
python src/utils/generate_visualizations.py
```

### 5. Run Custom Evaluations

Executes the robustness tests (Paraphrase, Ambiguity, Out-of-Scope) and prints a side-by-side comparison to the terminal.

```bash
python tests/custom_evaluations.py
```

---

## Results

| Metric           | TF-IDF + Logistic Regression       | Fine-tuned BERT                    |
| ---------------- | ---------------------------------- | ---------------------------------- |
| Accuracy         | See `docs/accuracy_comparison.png` | See `docs/accuracy_comparison.png` |
| Confusion Matrix | `docs/tfidf_confusion_matrix.png`  | `docs/bert_confusion_matrix.png`   |

---

## Author

**Jalal**
