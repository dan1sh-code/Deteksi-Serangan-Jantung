import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.predict import load_trained_model, predict_risk

if __name__ == "__main__":
    artifact = load_trained_model()
    print("=" * 60)
    print(" PREDIKSI RISKSI SERANGAN JANTUNG ")
    print(f" Model Default Terpilih : {artifact['model_name']}")
    print(f" ROC-AUC Score          : {artifact['roc_auc']:.4f}")
    print(f" Total Samples Trained  : {artifact.get('total_samples', 79886):,} baris data")
    print("=" * 60)
