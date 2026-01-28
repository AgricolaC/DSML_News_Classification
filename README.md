# News Article Classification

Machine learning pipeline built to classify news articles into 7 categories: **International**, **Sports**, **Business**, **Sci/Tech**, **City**, **Entertainment**, and **Politics**.

This project implements a Hierarchical Voting Ensemble, combining linear and non-linear classifiers with feature engineering (time extraction, source embeddings,text cleaning) to achieve high-performance text classification.

---

## 📂 Project Structure

```
DSML_News_Classification/
├── dataset/
│   ├── development.csv      # Training data (80,000 articles)
│   └── evaluation.csv        # Test data (20,000 articles)
├── src/                      # Source code
│   ├── preprocessing.py      # Data cleaning, time features, text processing
│   ├── models.py             # Model pipelines (SVC, LR, HGB, NB)
│   ├── ensemble.py           # Hierarchical voting ensemble & CV
│   ├── cv_utils.py           # Time-series cross-validation
│   ├── seed.py               # Reproducibility 
│   ├── tuning.py             # Hyperparameter optimization
│   ├── ablation.py           # Feature ablation studies
│   └── explainability.py     # Model interpretability
├── results/
│   ├── ensemble_cv/          # Cross-validation reports
│   ├── ablation/             # Feature ablation results
│   ├── explainability/       # Model interpretability outputs
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
Hierarchical Voting Classifier (5 models):

| Model | Weight | Time Features | Description |
|-------|--------|---------------|-------------|
| **LinearSVC** | 1 | Cyclical (sin/cos) | Fast, linear decision boundaries |
| **LogisticRegression** | 1 | None | Content-focused, robust baseline |
| **HistGradientBoosting** | 1/3 | Raw (NaN-aware) | Non-linear interactions |
| **MultinomialNB** | 1/3 | None | Probabilistic text classifier |
| **ComplementNB** | 1/3 | None | Handles imbalanced classes |

**Voting**: `LinearSVC(1) + LogisticRegression(1) + [HGB+MNB+CNB](1)`

### Feature Engineering

**Text Features**:
- TF-IDF vectorization (50K vocab, 1-3 grams)
- Preserves HTML/URLs (contain category-indicative patterns)
- Source tokenization (e.g., "TechCrunch" → "tech crunch")

**Metadata Features**:
- **Source OHE**: Top 300 sources (one-hot encoded)
- **PageRank**: 5-bin categorical encoding
- **Time Features** (model-specific):
  - Cyclical: `hour_sin/cos`, `day_sin/cos`, `month_sin/cos`, `week_sin/cos`, `quarter_sin/cos`
  - Linear: `year_offset` (median-centered)
  - Indicator: `is_missing_date`

### Preprocessing Pipeline
1. **Cleaning**: Fix encoding issues, detect hidden missing values
2. **Deduplication**: Remove conflicting duplicates (training only)
3. **Text Extraction**: Concatenate `source + title + article`, apply source tagging
4. **Time Extraction**: Extract temporal features (where available)
5. **Sorting**: Chronological ordering (training only, for CV)

### Cross-Validation
**AnchoredTimeSeriesSplit** (5 folds):
- Respects temporal order (prevents future info leakage)
- Expanding window: each fold trains on all previous data

---

## 📈 Results

### Validation Performance
- **Mean F1 Score**: **0.7577** ± 0.0285
- **Cross-Validation**: 5-fold time-series split

| Fold | F1 Score | Training Period |
|------|----------|-----------------|
| 1 | 0.7023 | Earliest 20% → Next 20% |
| 2 | 0.7641 | Earliest 40% → Next 20% |
| 3 | 0.7660 | Earliest 60% → Next 20% |
| 4 | 0.7834 | Earliest 80% → Last 20% |
| 5 | 0.7730 | All → Validation |

- **Public Leaderboard F1**: **0.735**

---

## 🔬 Additional Analysis

### Exploratory Data Analysis
See `eda.ipynb` for:
- Full dataset cleaning journey
- Missing value analysis (~36% timestamps missing) 
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

Bes parameters saved to `best_params.json`.

### Cross-Validation Report
```bash
python -m src.ensemble
```
Generates detailed fold-by-fold analysis in `results/ensemble_cv/cv_report.md`.

---

## 🛠 Troubleshooting

### Missing Dataset
```
Error: Dataset files not found in ./dataset/
```
**Solution**: Place `development.csv` and `evaluation.csv` in `./dataset/` directory.

---

