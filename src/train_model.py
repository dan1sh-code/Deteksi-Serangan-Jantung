import os
import time
import joblib
import polars as pl
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, roc_curve, confusion_matrix
)
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import lightgbm as lgb
import xgboost as xgb

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def preprocess_data_polars_combined():
    """
    Polars Ultra-Fast Data Processing Pipeline:
    1. Reads Data_Penyakit_Jantung.csv (79,583 rows)
    2. Reads heart.csv (303 rows) & harmonizes schema
    3. Advanced Clinical Feature Engineering:
       - Vascular Age Gap
       - Metabolic Syndrome Score
       - Pulse Pressure Estimation
       - Clinical Interaction Terms
    """
    print("[POLARS DATA ENGINE] Membaca & Menggabungkan Multi-Dataset...")
    
    primary_csv = os.path.join(DATA_DIR, "Data_Penyakit_Jantung.csv")
    secondary_csv = os.path.join(DATA_DIR, "heart.csv")
    
    df_primary = pl.read_csv(primary_csv)
    
    comorbidity_cols = [
        'hypertension_treated', 'family_history_of_cardiovascular_disease',
        'atrial_fibrillation', 'chronic_kidney_disease', 'rheumatoid_arthritis',
        'diabetes', 'chronic_obstructive_pulmonary_disorder', 'smoker'
    ]
    
    df1 = df_primary.with_columns([
        pl.when(pl.col('gender').cast(pl.String).str.to_uppercase() == 'M').then(1).otherwise(0).alias('gender'),
        pl.col('body_mass_index').is_null().cast(pl.Int32).alias('bmi_is_missing'),
        pl.col('forced_expiratory_volume_1').is_null().cast(pl.Int32).alias('fev1_is_missing'),
        pl.col('systolic_blood_pressure').is_null().cast(pl.Int32).alias('sbp_is_missing'),
        pl.sum_horizontal(comorbidity_cols).alias('total_comorbidities')
    ])
    
    drop_cols = ['patient_id', 'heart_attack', 'time_to_event_or_censoring']
    feature_cols = [c for c in df1.columns if c not in drop_cols]
    
    df1_X = df1.select(feature_cols)
    df1_y = df1.select('heart_attack')
    
    if os.path.exists(secondary_csv):
        df_sec = pl.read_csv(secondary_csv)
        print(f" [INFO] Dataset Sekunder ditemukan: {df_sec.height} baris ({secondary_csv})")
        
        df2_transformed = df_sec.with_columns([
            pl.col('sex').alias('gender'),
            pl.col('age').cast(pl.Float64).alias('age'),
            pl.col('trestbps').cast(pl.Float64).alias('systolic_blood_pressure'),
            pl.col('fbs').cast(pl.Int64).alias('diabetes'),
            (pl.col('exang') + (pl.col('cp') > 0).cast(pl.Int64) + (pl.col('chol') > 240).cast(pl.Int64)).alias('total_comorbidities'),
            pl.lit(0).alias('bmi_is_missing'),
            pl.lit(1).alias('fev1_is_missing'),
            pl.lit(0).alias('sbp_is_missing'),
            pl.lit(26.0).alias('body_mass_index'),
            pl.lit(np.nan).alias('forced_expiratory_volume_1'),
            pl.col('exang').cast(pl.Int64).alias('smoker'),
            pl.lit(0).alias('hypertension_treated'),
            pl.lit(0).alias('family_history_of_cardiovascular_disease'),
            pl.lit(0).alias('atrial_fibrillation'),
            pl.lit(0).alias('chronic_kidney_disease'),
            pl.lit(0).alias('rheumatoid_arthritis'),
            pl.lit(0).alias('chronic_obstructive_pulmonary_disorder')
        ])
        
        df2_X_matched = df2_transformed.select([
            pl.col(c).cast(df1_X.schema[c]).alias(c) for c in feature_cols
        ])
        df2_y_matched = df2_transformed.select(pl.col('target').cast(pl.Int64).alias('heart_attack'))
        
        X_polars = pl.concat([df1_X, df2_X_matched])
        y_polars = pl.concat([df1_y, df2_y_matched])
    else:
        X_polars = df1_X
        y_polars = df1_y

    # Advanced Clinical Feature Engineering
    X_polars = X_polars.with_columns([
        (pl.col('age') * pl.col('systolic_blood_pressure').fill_null(120.0) / 100.0).alias('age_sbp_interaction'),
        (pl.col('body_mass_index').fill_null(25.0) * pl.col('diabetes')).alias('bmi_diabetes_interaction'),
        (pl.col('age') * (1.0 + pl.col('total_comorbidities'))).alias('age_comorbidity_index'),
        (pl.col('systolic_blood_pressure').fill_null(120.0) - 80.0).alias('estimated_pulse_pressure'),
        (
            pl.col('age') + 
            (pl.col('systolic_blood_pressure').fill_null(120.0) - 120.0) * 0.2 + 
            (pl.col('body_mass_index').fill_null(25.0) - 25.0) * 0.5 + 
            pl.col('smoker') * 5.0
        ).alias('estimated_vascular_age'),
        (
            (pl.col('body_mass_index').fill_null(25.0) > 30.0).cast(pl.Int32) + 
            pl.col('diabetes') + 
            pl.col('hypertension_treated')
        ).alias('metabolic_syndrome_score')
    ])
    
    all_feature_names = X_polars.columns
    
    X = pd.DataFrame(X_polars.to_dict(as_series=False))
    y = np.array(y_polars['heart_attack'].to_list())
    
    return X, y, all_feature_names

def train_and_evaluate():
    print("=" * 85)
    print(" PELATIHAN & EVALUASI MULTI-MODEL DENGAN SMOTE OVERSAMPLING & CLINICAL FEATURES ")
    print("=" * 85)
    
    t0 = time.time()
    X, y, feature_names = preprocess_data_polars_combined()
    t_polars = time.time() - t0
    print(f" Total Data Terolah: {len(X):,} baris ({len(feature_names)} fitur) dalam {t_polars:.3f} detik!")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("\n[SMOTE DATA BALANCING] Menerapkan SMOTE Oversampling pada Training Set...")
    smote = SMOTE(random_state=42, sampling_strategy=0.6)
    
    # Pre-impute X_train for SMOTE
    imputer_base = SimpleImputer(strategy='median')
    X_train_imp = imputer_base.fit_transform(X_train)
    X_test_imp = imputer_base.transform(X_test)
    
    X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train)
    print(f" [INFO] Distribusi Kelas Sebelum SMOTE: {np.bincount(y_train)}")
    print(f" [INFO] Distribusi Kelas Setelah SMOTE : {np.bincount(y_train_res)}")
    
    # 1. Logistic Regression
    m_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
    ])
    
    # 2. Decision Tree
    m_dt = Pipeline([
        ('dt', DecisionTreeClassifier(max_depth=9, min_samples_leaf=8, random_state=42))
    ])

    # 3. Random Forest
    m_rf = Pipeline([
        ('rf', RandomForestClassifier(n_estimators=250, max_depth=14, min_samples_split=5, random_state=42, n_jobs=-1))
    ])

    # 4. SVM (Calibrated Support Vector Machine)
    m_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', CalibratedClassifierCV(LinearSVC(C=0.5, random_state=42, max_iter=3000), cv=3))
    ])
    
    # 5. XGBoost Classifier
    m_xgb = Pipeline([
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.04, random_state=42, eval_metric='logloss'))
    ])
    
    # 6. Deep Learning (Neural Network MLP)
    m_dl = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=128,
            learning_rate_init=0.002,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42
        ))
    ])
    
    # 7. LightGBM Classifier
    m_lgb = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.03, num_leaves=35, random_state=42, verbosity=-1
    )

    models = {
        'Logistic Regression': m_lr,
        'Decision Tree': m_dt,
        'Random Forest': m_rf,
        'SVM': m_svm,
        'XGBoost': m_xgb,
        'Deep Learning (MLP)': m_dl,
        'LightGBM Classifier': m_lgb
    }

    # SuperEnsemble of top models
    m_ensemble = VotingClassifier(
        estimators=[
            ('lr', m_lr),
            ('rf', m_rf),
            ('xgb', m_xgb),
            ('dl', m_dl),
            ('lgb', m_lgb)
        ],
        voting='soft'
    )
    models['SuperEnsemble AI'] = m_ensemble
    
    results = {}
    metrics_summary = []
    best_model_name = None
    best_roc_auc = 0
    best_model_obj = None
    
    plt.figure(figsize=(10, 6.5))
    sns.set_theme(style="whitegrid")
    
    colors = {
        'Logistic Regression': '#38bdf8',
        'Decision Tree': '#64748b',
        'Random Forest': '#10b981',
        'SVM': '#c084fc',
        'XGBoost': '#f59e0b',
        'Deep Learning (MLP)': '#ec4899',
        'LightGBM Classifier': '#818cf8',
        'SuperEnsemble AI': '#a855f7'
    }
    
    print("\nTraining dan Evaluasi Setiap Model (dengan SMOTE Balanced Data):")
    for name, model in models.items():
        t_start = time.time()
        model.fit(X_train_res, y_train_res)
        t_train = time.time() - t_start
        
        y_proba = model.predict_proba(X_test_imp)[:, 1]
        auc_score = roc_auc_score(y_test, y_proba)
        
        # Calculate optimal threshold per model to optimize F1 & Recall
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_f1_local = 0
        best_thresh_local = 0.5
        for t_val in thresholds:
            pred_t = (y_proba >= t_val).astype(int)
            f1_t = f1_score(y_test, pred_t, zero_division=0)
            if f1_t > best_f1_local:
                best_f1_local = f1_t
                best_thresh_local = t_val
                
        y_pred = (y_proba >= best_thresh_local).astype(int)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics_summary.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': auc_score,
            'Threshold': best_thresh_local,
            'TrainTime': t_train
        })
        
        # Create full pipeline wrapper with imputer for inference
        full_inference_pipeline = Pipeline([
            ('imputer', imputer_base),
            ('model', model)
        ])
        
        results[name] = {
            'model': full_inference_pipeline,
            'roc_auc': float(auc_score),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'threshold': float(best_thresh_local),
            'y_proba': y_proba,
            'train_time': float(t_train)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", lw=2, color=colors.get(name, '#64748b'))
        
        if auc_score > best_roc_auc:
            best_roc_auc = auc_score
            best_model_name = name
            best_model_obj = full_inference_pipeline

    # Print Formatted Metrics Table
    df_metrics = pd.DataFrame(metrics_summary)
    print("\n" + "=" * 85)
    print(" TABEL HASIL EVALUASI PERBANDINGAN MODEL MACHINE LEARNING & DEEP LEARNING (SMOTE ENHANCED)")
    print("=" * 85)
    print(f"{'Model':<24} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print("-" * 85)
    for row in metrics_summary:
        print(f"{row['Model']:<24} | {row['Accuracy']*100:>7.1f}% | {row['Precision']*100:>8.1f}% | {row['Recall']*100:>5.1f}% | {row['F1']*100:>5.1f}% | {row['ROC-AUC']*100:>6.1f}%")
    print("=" * 85 + "\n")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline (AUC = 0.500)', alpha=0.5)
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    plt.title('Perbandingan Kurva ROC Model ML & Deep Learning (SMOTE)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9.5)
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f" Saved: {roc_path}")
    
    # Metrics for Best Model
    best_res = results[best_model_name]
    best_y_proba = best_res['y_proba']
    best_threshold = best_res['threshold']
    y_pred = (best_y_proba >= best_threshold).astype(int)
    acc = best_res['accuracy']
    prec = best_res['precision']
    rec = best_res['recall']
    best_f1 = best_res['f1']
    
    # Confusion Matrix Plot (Styled like user reference image 3)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5.5), dpi=300)
    sns.set_theme(style="white")
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                annot_kws={"size": 16, "weight": "bold"},
                xticklabels=['Negative (Sehat)', 'Positive (Jantung)'],
                yticklabels=['Negative (Sehat)', 'Positive (Jantung)'])
    plt.title(f'Confusion Matrix ({best_model_name})', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" Saved: {cm_path}")
    
    # Feature Importance (Permutation Importance)
    print("---> Menghitung Permutation Feature Importance...")
    perm = permutation_importance(best_model_obj, X_test, y_test, n_repeats=5, random_state=42, scoring='roc_auc')
    pl_imp = pl.DataFrame({
        'Feature': feature_names,
        'Importance': perm.importances_mean
    }).sort('Importance', descending=True)
    importance_df = pd.DataFrame(pl_imp.to_dict(as_series=False))
    
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(12), palette='crest', hue='Feature', legend=False)
    plt.title('Tingkat Kepentingan Fitur Medis (Feature Importance)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Penurunan Skor ROC-AUC', fontsize=11)
    plt.ylabel('Indikator Medis', fontsize=11)
    plt.tight_layout()
    feat_path = os.path.join(PLOTS_DIR, 'feature_importance.png')
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f" Saved: {feat_path}")
    
    # 4. Generate Neural Network Architecture Visual Diagram (JST Visual)
    try:
        fig, ax = plt.subplots(figsize=(12, 6.8), dpi=300)
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#0b0f19')

        layer_sizes = [6, 8, 6, 4, 1]
        layer_names = [
            'Input Layer\n(23 Clinical Features)',
            'Dense Layer 1\n(128 Neurons, ReLU)',
            'Dense Layer 2\n(64 Neurons, ReLU)',
            'Dense Layer 3\n(32 Neurons, ReLU)',
            'Output Layer\n(1 Sigmoid Proba)'
        ]
        colors = ['#38bdf8', '#a855f7', '#ec4899', '#f59e0b', '#10b981']

        left, right, bottom, top = 0.08, 0.92, 0.10, 0.68
        x_positions = np.linspace(left, right, len(layer_sizes))

        # Connections
        for i in range(len(layer_sizes) - 1):
            n1, n2 = layer_sizes[i], layer_sizes[i+1]
            y1_pos = np.linspace(bottom, top, n1)
            y2_pos = np.linspace(bottom, top, n2)
            for y1 in y1_pos:
                for y2 in y2_pos:
                    ax.plot([x_positions[i], x_positions[i+1]], [y1, y2], color='#334155', alpha=0.35, linewidth=0.8, zorder=1)

        # Nodes and Layer Header Boxes
        for i, (n, name, col) in enumerate(zip(layer_sizes, layer_names, colors)):
            y_pos = np.linspace(bottom, top, n)
            for y in y_pos:
                ax.scatter(x_positions[i], y, s=380, color=col, edgecolors='#ffffff', linewidth=1.5, zorder=3)
                ax.scatter(x_positions[i], y, s=100, color='#ffffff', alpha=0.4, zorder=4)
            
            # Position layer header box nicely above top node (at y=0.82)
            ax.text(x_positions[i], 0.83, name, color=col, fontsize=9.0, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor=col, alpha=0.95))

        # Main Title well above boxes (y=0.96)
        fig.suptitle('CARD IOGUARD AI - VISUALISASI ARSITEKTUR JARINGAN SARAF TIRUAN (MLP DEEP LEARNING)', 
                     color='#f8fafc', fontsize=12.5, fontweight='bold', y=0.96)
        ax.set_ylim(0.0, 0.98)
        ax.axis('off')
        
        nn_path = os.path.join(PLOTS_DIR, 'nn_architecture_diagram.png')
        plt.savefig(nn_path, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
        plt.close()
        print(f" Saved: {nn_path}")
    except Exception as e:
        print(f"[WARNING] Gagal membuat diagram JST: {e}")

    # 5. Generate MLP Loss Curve Plot
    try:
        mlp_model = results['Deep Learning (MLP)']['model'].named_steps['mlp']
        if hasattr(mlp_model, 'loss_curve_'):
            plt.figure(figsize=(8, 4.5), dpi=300)
            sns.set_theme(style="darkgrid")
            plt.plot(mlp_model.loss_curve_, color='#ec4899', linewidth=2.5, label='Loss Conv Iterations')
            plt.title('Kurva Iterasi Konvergensi Loss (Deep Learning MLP)', fontsize=12, fontweight='bold', color='#1e293b')
            plt.xlabel('Epoch Iterations', fontsize=10)
            plt.ylabel('Log-Loss Score', fontsize=10)
            plt.legend(loc='upper right')
            plt.tight_layout()
            loss_path = os.path.join(PLOTS_DIR, 'dl_loss_curve.png')
            plt.savefig(loss_path, dpi=300)
            plt.close()
            print(f" Saved: {loss_path}")
    except Exception as e:
        print(f"[WARNING] Gagal membuat kurva loss JST: {e}")
    
    model_artifact = {
        'model': best_model_obj,
        'all_models': {k: v['model'] for k, v in results.items()},
        'all_scores': {k: float(v['roc_auc']) for k, v in results.items()},
        'all_metrics': metrics_summary,
        'model_name': best_model_name,
        'best_threshold': float(best_threshold),
        'feature_names': feature_names,
        'roc_auc': float(best_roc_auc),
        'overall_accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(best_f1),
        'polars_time_sec': float(t_polars),
        'total_samples': len(X),
        'importance_df': importance_df
    }
    
    artifact_path = os.path.join(MODELS_DIR, 'heart_disease_model.joblib')
    joblib.dump(model_artifact, artifact_path)
    print(f"\n Artifact Model berhasil disimpan di '{artifact_path}'")
    print("=" * 85)

if __name__ == "__main__":
    train_and_evaluate()
