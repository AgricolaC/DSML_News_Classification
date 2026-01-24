# News Classification Pipeline

This project implements a machine learning solution for classifying news articles into 7 categories (International, Sports, Business, etc.). The core analysis and modeling are contained within the `eda.ipynb` notebook.

## 📂 Project Structure

```text
News-Classification-Project/
├── dataset/
│   ├── development.csv
│   └── evaluation.csv
├── eda.ipynb         # Main analysis and modeling notebook
├── README.md         # Project documentation
├── pyproject.toml    # Dependencies (optional)
└── .gitignore
```

## 🚀 Setup & Installation

### 1. Prerequisites

*   Python 3.10+
*   Jupyter Notebook or JupyterLab

### 2. Clone the Repository

```bash
git clone https://github.com/AgricolaC/DSML_News_Classification
cd DSML_News_Classification
```

### 3. Install Dependencies

You can install the required packages using `pip` or `poetry`. Poetry was used by the authors due to its superior dependency management and virtual environment capabilities. It is also highly recommended to the users & reviewers of this repository. 

**a. Using pip:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**b. Using Poetry:**
```bash
poetry install
```

### 4. Dataset

The dataset is too large to be hosted on GitHub. You must add it manually:

1. Create a folder named `dataset` in the root directory.
2. A download link to the datasets can be found in "guidelines/DSML Project Assignment.pdf". Download it from the link
3. Place the downloaded `development.csv` and `evaluation.csv` files inside it. They represent our training and test samples, respectively.

## 🏃‍♂️ How to Run

1.  Activate your environment (if using Poetry: `poetry shell`).
2.  Launch the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
3.  Open `eda.ipynb` in the browser interface.
4.  Run all cells to execute the Exploratory Data Analysis, Preprocessing, Model Training & Final Submission steps.

## 🧠 Model Architecture

*   **Text Preprocessing**: Minimal cleaning strategy (preserving high-signal artifacts like HTML tags and specific URL patterns).
*   **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) with n-grams (1-3).
*   **Classifier**: **LinearSVC** (Support Vector Machine with Linear Kernel), optimized for high-dimensional sparse text data.

## 📊 Results

The model achieves strong performance on the validation set, sometimes leveraging "leakage" signals (source artifacts) as predictive features.

*   **Macro-F1 (Validation):** ~0.76
