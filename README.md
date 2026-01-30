# News Article Classification

Machine learning pipeline built to classify news articles into 7 categories.
This project implements a Hierarchical Voting Ensemble, combining linear and non-linear classifiers with robust feature engineering (source embeddings, text cleaning, strict deduplication) to achieve high-performance text classification.

---

## 📂 Project Structure

```
DSML_News_Classification/
├── dataset/
│   ├── development.csv       # Training data (80,000 articles)
│   └── evaluation.csv        # Test data (20,000 articles)
├── src/                      # Source code
│   ├── preprocessing.py      # Data cleaning, deduplication, text processing
│   ├── models.py             # Model pipelines (SVC, LR, HGB, NB)
│   ├── ensemble.py           # Hierarchical voting ensemble & CV
│   ├── seed.py               # Reproducibility 
│   ├── tuning.py             # Hyperparameter optimization
│   ├── ablation.py           # Feature ablation studies
├── results/
│   ├── ensemble_cv/          # Cross-validation reports
│   ├── ablation/             # Feature ablation results
│   ├── figures/              # Visualizations
│   └── tuning/               # Hyperparameter tuning logs
├── main.py                   # **Main submission script**
├── eda.ipynb                 # Exploratory data analysis
├── best_params.json          # Optimized hyperparameters
├── pyproject.toml            # Poetry dependencies
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Generate Submission (Evaluators)

**Prerequisites**: Python 3.11+ and ~2GB RAM.
The fastest way to run the pipeline and generate predictions is via Poetry:

### Option A: Using Poetry (Recommended)

1. **Install Poetry** (if not installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Run the Pipeline**:
   ```bash
   poetry run python main.py
   ```

### Option B: Using Standard pip

1. **Create Virtual Environment** (requires Python 3.11+):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   # .venv\Scripts\activate       # Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline**:
   ```bash
   python main.py
   ```

### Option C: Using Existing Environment

If you already have a configured environment:
```bash
source .venv/bin/activate
python main.py
```

---

## 📊 Dataset

Place the dataset files in the `dataset/` directory:
- `development.csv` - Training data (80,000 rows)
- `evaluation.csv` - Test data (20,000 rows)

**Columns**:
- `source`: News source (e.g., "CNN", "BBC")
- `title`: Article headline
- `article`: Full article text (may contain HTML)
- `timestamp`: Publication date (many missing values: ~36%)
- `page_rank`: PageRank score (1-5)
- `label`: Category (development.csv only)
- `Id`: Unique identifier

---

## 🧠 Model Architecture

### Ensemble Strategy
Hierarchical Voting Classifier (Hard Voting):

| Model | Role | Weights | Description |
|-------|------|---------|-------------|
| **LinearSVC** | Strong Learner | 1 | Fast, linear decision boundaries, heavy metadata lifting |
| **LogisticRegression** | Strong Learner | 1 | Robust baseline, probabilistic outputs |
| **Weak Consensus** | Tie-Breaker | 1 | Sub-ensemble of 3 models (HGB + MNB + CNB) |

**Structure**: `LinearSVC` vs `LogisticRegression` vs `[HGB + MNB + CNB]`
*The "Weak Consensus" acts as a single voter representing the majority view of the non-linear/probabilistic models.*

### Feature Engineering

**Text Features**:
- TF-IDF vectorization (30K vocab, 1-3 grams, sublinear TF)
- **Source Token Injection**: Source names are tokenized (e.g., `src_reuters`) and injected into the text to capture local context.

**Metadata Features**:
- **Source OHE**: Top 300 sources (one-hot encoded).
- **PageRank**: One-hot encoded (Rank 1-5).
- **Time Features**: **REMOVED**. Extensive ablation studies showed time features (cyclical or raw) introduced noise and degraded performance on this specific dataset.

### Preprocessing Pipeline
1. **Cleaning**: Fix encoding (ftfy), standardized artifacts.
2. **Strict Deduplication**:
   - **Conflict Drop**: Any content group with >1 unique label ID is dropped entirely (removes ambiguous training signals).
   - **Chronological Keep**: For non-conflicting duplicates, only the *earliest* timestamp is kept.
   - **Leakage Prevention**: Deduplication is performed on `[title, article]` only, ignoring `source` to prevent label leakage from syndicated content.
3. **Text Extraction**: Concatenate `source_token + title + article`.

### Cross-Validation
**StratifiedKFold** (5 folds):
- Standard stratified splitting to maintain class distribution.

---

## 📈 Results

### Validation Performance
- **Mean F1 Score**: **0.7322** ± 0.0017
- **Cross-Validation**: 5-fold Stratified Shuffle

| Fold | F1 Score |
|------|----------|
| 1 | 0.7321 |
| 2 | 0.7319 |
| 3 | 0.7334 |
| 4 | 0.7293 |
| 5 | 0.7344 |

### Ablation Summary
*From `results/ablation/ablation_results.csv`*

*Analysis made for LinearSVC*

| Step | F1 Score (Stratified) | Impact |
|------|----------------------|--------|
| Baseline (Raw) | 0.6696 | - |
| + Cleaner & Dedup | 0.6781 | +0.85% |
| + Features & Source | 0.7236 | +4.55% |
| + PageRank | **0.7280** | +0.44% |
| + Time Features | 0.7192 | **-0.88% (Harmful)** |

*Note: Time features consistently degraded performance, justifying their removal.*

---

## 🔬 Additional Analysis

### Exploratory Data Analysis
See `eda.ipynb` for:
- Full dataset cleaning journey
- Missing value analysis 
- Category distribution 
- Source-category correlations
- Temporal trends

### Hyperparameter Tuning
Run individual model tuning:
```bash
python -m src.tuning --model mnb   # Naive Bayes
python -m src.tuning --model cnb   # Complement NB
python -m src.tuning --model hgb   # Gradient Boosting
python -m src.tuning --model lr   # Logistic Regression
python -m src.tuning --model svc   # LinearSVC
```

Best parameters saved to `best_params.json`.

### Cross-Validation Report
```bash
python -m src.ensemble
```
Generates detailed fold-by-fold analysis in `results/ensemble_cv/fold_scores.json`.

---

## 🛠 Troubleshooting

### Missing Dataset
```
Error: Dataset files not found in ./dataset/
```
**Solution**: Place `development.csv` and `evaluation.csv` in `./dataset/` directory.

---
