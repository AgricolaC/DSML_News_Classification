import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import sys
import os



from .preprocessing import DatasetCleaner, DatasetDeduplicator, FeatureExtractor, TimeExtractor, PageRankOneHot, SourceTransformer
from . import models  # Use centralized model definitions

sns.set_theme(style="whitegrid")

class Interpreter:
    def __init__(self, model):
        self.model = model
        self.feature_names = None

    def get_feature_names(self, X_sample):
        """Reconstructs feature names from the pipeline (TF-IDF + Dense columns)."""
        svc_pipe = self.model.estimators_[0] 
        preprocessor = svc_pipe.named_steps['prep']
        
        # Use sklearn's native method to guarantee alignment with coefficients
        self.feature_names = preprocessor.get_feature_names_out()
        
        # Optional: Clean up prefixes if desired, but keeping them is safer for clarity
        # self.feature_names = [f.split('__')[-1] for f in self.feature_names]
        
        print(f"[Interpreter] Extracted {len(self.feature_names)} features.")
        return self.feature_names

    def plot_top_coefficients(self, class_label, top_n=20):
        if self.feature_names is None:
            raise ValueError("Run get_feature_names() first.")

        # Access SVC (Estimator 0)
        svc_clf = self.model.estimators_[0].named_steps['clf']
        
        # Handle CalibratedClassifierCV wrapper
        if hasattr(svc_clf, 'calibrated_classifiers_'):
            # Take the first calibrated fold's estimator
            base_model = svc_clf.calibrated_classifiers_[0].estimator
        else:
            base_model = svc_clf

        if base_model.coef_.shape[0] == 1:
            coefs = base_model.coef_[0]
        else:
            coefs = base_model.coef_[class_label]

        df_coef = pd.DataFrame({'feature': self.feature_names, 'coef': coefs})
        
        top_positive = df_coef.nlargest(top_n, 'coef')
        top_negative = df_coef.nsmallest(10, 'coef') 
        plot_df = pd.concat([top_positive, top_negative])
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if c < 0 else 'blue' for c in plot_df['coef']]
        sns.barplot(data=plot_df, y='feature', x='coef', palette=colors, orient='h')
        plt.title(f"Top Features for Class {class_label} (LinearSVC)")
        plt.xlabel("Coefficient Strength")
        plt.tight_layout()
        
        # Save plot instead of showing
        os.makedirs('results/explainability', exist_ok=True)
        plt.savefig(f'results/explainability/class_{class_label}_coefficients.png', dpi=150, bbox_inches='tight')
        plt.close()

def run_analysis():
    print("="*60)
    print("EXPLAINABILITY PROTOCOL (Split Development Data)")
    print("="*60)
    
    # 1. Load Data
    df = pd.read_csv('dataset/development.csv')
    
    # 2. Common Preprocessing (Stateless)
    print("\n[Prep] Cleaning, Deduplicating, and Extracting Features...")
    df = DatasetCleaner(verbose=True).fit_transform(df)
    df = DatasetDeduplicator(mode='advanced').fit_transform(df)
    df = FeatureExtractor().fit_transform(df)
    # df = AdvancedTextCleaner().fit_transform(df)  # Removed: unused in production
    df = PageRankOneHot().fit_transform(df)
    df = TimeExtractor().fit_transform(df)
    
    # 3. Split (Hold-out for Explanation)
    print("\n[Split] Creating 80/20 Train/Validation Split for audit...")
    X = df 
    y = df['label']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Stateful Preprocessing (Source Transformer)
    print("[Prep] Fitting SourceTransformer on Train only...")
    source_trans = SourceTransformer(top_k=300)
    X_train = source_trans.fit_transform(X_train)
    X_val = source_trans.transform(X_val)
    
    # 5. Train
    print(f"\n[Train] Fitting Ensemble on {len(X_train)} rows...")
    models.load_best_params()  # Load hyperparameters
    estimators = models.get_pipelines()
    voting = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1) 
    voting.fit(X_train, y_train)
    
    return voting, X_val, y_val

def audit_errors(model, X_val, y_val):
    print("\n" + "="*60)
    print("ANALYSIS C: FAILURE AUDIT")
    print("="*60)
    
    preds_ensemble = model.predict(X_val)
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, preds_ensemble)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Validation Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Save confusion matrix
    os.makedirs('results/explainability', exist_ok=True)
    plt.savefig('results/explainability/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Consensus Failures
    pred_svc = model.estimators_[0].predict(X_val)
    pred_lr = model.estimators_[1].predict(X_val)
    pred_hgb = model.estimators_[2].predict(X_val)
    pred_mnb = model.estimators_[3].predict(X_val)
    
    consensus_mask = (pred_svc == pred_lr) & (pred_lr == pred_hgb) & (pred_hgb == pred_mnb)
    failure_mask = consensus_mask & (pred_svc != y_val)
    
    failures = X_val[failure_mask].copy()
    failures['True_Label'] = y_val[failure_mask]
    failures['Pred_Label'] = pred_svc[failure_mask]
    
    print(f"\nFound {len(failures)} Consensus Failures out of {len(X_val)} validation samples.")
    print("Top 10 Examples:")
    
    topic_map = {0:'International', 1:'Business', 2:'Tech', 3:'Entertainment', 4:'Sports', 5:'General', 6:'Health'}
    
    for idx, row in failures.head(10).iterrows():
        t_lbl = topic_map.get(row['True_Label'], str(row['True_Label']))
        p_lbl = topic_map.get(row['Pred_Label'], str(row['Pred_Label']))
        
        print(f"\n[ID: {idx}] True: {t_lbl} | Pred: {p_lbl}")
        print(f"Title: {row['title']}")
        print(f"Text Snippet: {row['final_text'][:200]}...") 

def check_metadata_importance(model, X_val, y_val):
    print("\n" + "="*60)
    print("ANALYSIS B: HGB PERMUTATION IMPORTANCE")
    print("="*60)
    
    hgb_pipe = model.estimators_[2]
    
    print("Running Permutation Importance on HGB...")
    X_sample = X_val.sample(min(2000, len(X_val)), random_state=42)
    y_sample = y_val.loc[X_sample.index]
    
    result = permutation_importance(hgb_pipe, X_sample, y_sample, n_repeats=5, random_state=42, n_jobs=-1)
    
    importances = pd.Series(result.importances_mean, index=X_val.columns)
    
    src_imp = importances[importances.index.str.startswith('src_')].sum()
    rank_imp = importances[importances.index.str.startswith('rank_')].sum()
    time_imp = importances[importances.index.str.startswith(('hour_', 'day_', 'week_', 'month_', 'is_missing_'))].sum()
    txt_imp = importances['final_text']
    
    print(f"Text Importance:     {txt_imp:.4f}")
    print(f"Source Importance:   {src_imp:.4f}")
    print(f"PageRank Importance: {rank_imp:.4f}")
    print(f"Time Importance:     {time_imp:.4f}")
    
    meta_df = pd.DataFrame({
        'Feature Group': ['Text', 'Source', 'PageRank', 'Time'],
        'Importance': [txt_imp, src_imp, rank_imp, time_imp]
    })
    plt.figure(figsize=(8,5))
    sns.barplot(data=meta_df, x='Feature Group', y='Importance')
    plt.title("Feature Group Importance (HGB)")
    
    # Save feature importance plot
    os.makedirs('results/explainability', exist_ok=True)
    plt.savefig('results/explainability/feature_group_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save importance scores to JSON
    import json
    with open('results/explainability/feature_importance.json', 'w') as f:
        json.dump({
            'text': float(txt_imp),
            'source': float(src_imp),
            'pagerank': float(rank_imp),
            'time': float(time_imp)
        }, f, indent=2)

if __name__ == "__main__":
    voting, X_val, y_val = run_analysis()
    
    interpreter = Interpreter(voting)
    interpreter.get_feature_names(X_val)
    
    print("\n" + "="*60)
    print("ANALYSIS A: VOCABULARY ANALYSIS")
    print("="*60)
    for i in range(7):
        print(f"Plotting coefficients for Class {i}...")
        interpreter.plot_top_coefficients(class_label=i)     
    check_metadata_importance(voting, X_val, y_val)
    audit_errors(voting, X_val, y_val)
    
    print("\n[Done] Explainability Report Generated.")
    print("Results saved to results/explainability/")
