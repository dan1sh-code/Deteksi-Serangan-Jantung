import os
import polars as pl
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

def run_eda_polars():
    print("=" * 75)
    print(" GENERATING HIGH-END CLINICAL EDA DASHBOARDS & VISUALIZATIONS ")
    print("=" * 75)
    
    primary_csv = os.path.join(DATA_DIR, "Data_Penyakit_Jantung.csv")
    secondary_csv = os.path.join(DATA_DIR, "heart.csv")
    
    df_primary = pl.read_csv(primary_csv)
    
    if os.path.exists(secondary_csv):
        df_sec = pl.read_csv(secondary_csv)
    else:
        df_sec = None

    # Set aesthetic styles
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    # -------------------------------------------------------------------
    # CHART 1: Correlation Matrix Heatmap (Exact match to User Image 1)
    # -------------------------------------------------------------------
    if df_sec is not None:
        df_corr_data = df_sec.to_pandas()
    else:
        num_cols = [c for c, dt in zip(df_primary.columns, df_primary.dtypes) if dt in [pl.Int64, pl.Float64, pl.Int32] and c != 'patient_id']
        df_corr_data = df_primary.select(num_cols).to_pandas()

    corr = df_corr_data.corr()

    plt.figure(figsize=(9, 8.5), dpi=300)
    sns.set_theme(style="white")
    ax = sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="Blues",
        cbar_kws={'shrink': 0.8},
        linewidths=0.5, linecolor='white',
        annot_kws={"size": 9.5}
    )
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    corr_path = os.path.join(PLOTS_DIR, 'eda_correlation_matrix.png')
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f" [1/4] Saved Correlation Heatmap: {corr_path}")

    # -------------------------------------------------------------------
    # CHART 2: Analysis on Heart Disease (Exact match to User Image 2)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), dpi=300)
    fig.suptitle("Analysis on Heart Disease", fontsize=18, color="#D9534F", fontweight="bold", y=0.98)
    fig.patch.set_facecolor('#FFFFFF')

    df_p = df_sec.to_pandas() if df_sec is not None else df_primary.to_pandas()

    # 1. Age Distribution
    ax1 = axes[0, 0]
    sns.histplot(df_p['age'], bins=10, color='#5B9BD5', edgecolor='white', ax=ax1)
    ax1.set_title("Age distribution", fontsize=12, pad=10)
    ax1.set_xlabel("Age (bin)", fontsize=9)
    ax1.set_ylabel("Count of Age", fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 2. Resting Heart Rate Range / Blood Pressure
    ax2 = axes[0, 1]
    bp_col = 'trestbps' if 'trestbps' in df_p.columns else ('systolic_blood_pressure' if 'systolic_blood_pressure' in df_p.columns else df_p.columns[2])
    sns.histplot(df_p[bp_col].dropna(), bins=10, color='#ED7D31', edgecolor='white', ax=ax2)
    ax2.set_title("Resting Heart Rate Range", fontsize=12, pad=10)
    ax2.set_xlabel(f"{bp_col} (bin)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. Exercise vs No Exercise by Sex
    ax3 = axes[1, 0]
    sex_col = 'sex' if 'sex' in df_p.columns else ('gender' if 'gender' in df_p.columns else df_p.columns[1])
    ex_col = 'exang' if 'exang' in df_p.columns else ('smoker' if 'smoker' in df_p.columns else df_p.columns[3])
    df_ex = df_p.groupby([sex_col, ex_col]).size().unstack().fillna(0)
    df_ex.plot(kind='bar', ax=ax3, color=['#F2C99C', '#4682B4'], width=0.7)
    ax3.set_title("Exercise vs No Exercise", fontsize=12, pad=10)
    ax3.set_xticklabels(['Female', 'Male'], rotation=0)
    ax3.set_xlabel("")
    ax3.set_ylabel("Count", fontsize=9)
    ax3.legend(["YES", "NO"], fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 4. Heart Patients vs No Heart Disease Patients (Pie)
    ax4 = axes[1, 1]
    target_col = 'target' if 'target' in df_p.columns else ('heart_attack' if 'heart_attack' in df_p.columns else df_p.columns[-1])
    counts = df_p[target_col].value_counts()
    colors_pie = ['#4682B4', '#ED7D31']
    labels_pie = ['No Heart Disease', 'Heart Disease']
    ax4.pie(counts, labels=[f"{l}\n{v/len(df_p)*100:.2f}%" for l, v in zip(labels_pie, counts)], colors=colors_pie, startangle=140)
    ax4.set_title("Heart Patients vs No Heart Disease Patients", fontsize=12, pad=10)

    # 5. Chest Pain Types (Pie Chart / Bar)
    ax5 = axes[2, 0]
    cp_col = 'cp' if 'cp' in df_p.columns else ('diabetes' if 'diabetes' in df_p.columns else df_p.columns[0])
    cp_counts = df_p[cp_col].value_counts().sort_index()
    colors_cp = ['#4682B4', '#ED7D31', '#D9534F', '#5CB85C']
    labels_cp = ['Asymptomatic', 'Atypical Angina', 'Non-anginal pain', 'Typical Angina'][:len(cp_counts)]
    ax5.pie(cp_counts, labels=labels_cp, colors=colors_cp[:len(cp_counts)], autopct='%1.2f%%', startangle=90)
    ax5.set_title("Chest Pain Types", fontsize=12, pad=10)

    # 6. Blood Pressure Rate
    ax6 = axes[2, 1]
    sns.histplot(df_p[bp_col].dropna(), bins=8, color='#4682B4', edgecolor='white', ax=ax6)
    ax6.set_title("Blood Pressure Rate", fontsize=12, pad=10)
    ax6.set_xlabel("BP Range", fontsize=9)
    ax6.set_ylabel("Count", fontsize=9)
    ax6.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    analysis_path = os.path.join(PLOTS_DIR, 'eda_target_distribution.png')
    plt.savefig(analysis_path, dpi=300)
    plt.close()
    print(f" [2/4] Saved Multi-Panel Analysis: {analysis_path}")

    # -------------------------------------------------------------------
    # CHART 3: Heart Disease Data Visualization (Exact match to User Image 4)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    fig.suptitle("Heart Disease Data Visualization", fontsize=16, color="#1F2937", fontweight="bold", y=0.98)
    fig.patch.set_facecolor('#F9FAFB')

    coral_blue = ['#E06D53', '#6BA2C7']

    # 1. Age Distribution by Target
    ax1 = axes[0, 0]
    sns.histplot(data=df_p, x='age', hue=target_col, multiple='dodge', palette=coral_blue, ax=ax1, shrink=0.8)
    ax1.set_title("Age Distribution", fontsize=11, color="#4B5563")
    ax1.set_xlabel("")

    # 2. Sex Distribution
    ax2 = axes[0, 1]
    sns.countplot(data=df_p, x=sex_col, hue=target_col, palette=coral_blue, ax=ax2)
    ax2.set_title("Sex Distribution", fontsize=11, color="#4B5563")
    ax2.set_xlabel("")

    # 3. Exercise Induced Angina
    ax3 = axes[0, 2]
    sns.countplot(data=df_p, x=ex_col, hue=target_col, palette=coral_blue, ax=ax3)
    ax3.set_title("Exercise Induced Angina", fontsize=11, color="#4B5563")
    ax3.set_xlabel("")

    # 4. Chest Pain Type
    ax4 = axes[1, 0]
    sns.countplot(data=df_p, x=cp_col, hue=target_col, palette=coral_blue, ax=ax4)
    ax4.set_title("Chest Pain Type", fontsize=11, color="#4B5563")
    ax4.set_xlabel("")

    # 5. Resting Blood Pressure Boxplot by Target
    ax5 = axes[1, 1]
    sns.boxplot(data=df_p, x=target_col, y=bp_col, palette=coral_blue, ax=ax5, width=0.4)
    ax5.set_title("Resting Blood Pressure by Target", fontsize=11, color="#4B5563")
    ax5.set_xlabel("")

    # 6. Fasting Blood Sugar
    ax6 = axes[1, 2]
    fbs_col = 'fbs' if 'fbs' in df_p.columns else ('diabetes' if 'diabetes' in df_p.columns else df_p.columns[0])
    sns.countplot(data=df_p, x=fbs_col, hue=target_col, palette=coral_blue, ax=ax6)
    ax6.set_title("Fasting Blood Sugar", fontsize=11, color="#4B5563")
    ax6.set_xlabel("")

    for ax in axes.flat:
        ax.set_facecolor('#F9FAFB')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    viz_path = os.path.join(PLOTS_DIR, 'eda_clinical_visualization.png')
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f" [3/4] Saved Clinical Subplots Grid: {viz_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_eda_polars()
