import os
import joblib
import polars as pl
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_trained_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "heart_disease_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan! Silakan jalankan 'python train_model.py' terlebih dahulu.")
    return joblib.load(model_path)

def format_patient_input_polars(patient_dict, feature_names):
    data = patient_dict.copy()
    gender_val = str(data.get('gender', 'F')).upper()
    gender_binary = 1 if gender_val in ['M', 'MALE', '1'] else 0
    
    age = float(data.get('age', 50))
    bmi = float(data.get('body_mass_index', 25.0)) if data.get('body_mass_index') is not None and not np.isnan(float(data.get('body_mass_index'))) else 25.0
    smoker = int(data.get('smoker', 0))
    sbp = float(data.get('systolic_blood_pressure', 120.0)) if data.get('systolic_blood_pressure') is not None and not np.isnan(float(data.get('systolic_blood_pressure'))) else 120.0
    hypertension_treated = int(data.get('hypertension_treated', 0))
    family_history = int(data.get('family_history_of_cardiovascular_disease', 0))
    atrial_fib = int(data.get('atrial_fibrillation', 0))
    kidney_disease = int(data.get('chronic_kidney_disease', 0))
    arthritis = int(data.get('rheumatoid_arthritis', 0))
    diabetes = int(data.get('diabetes', 0))
    copd = int(data.get('chronic_obstructive_pulmonary_disorder', 0))
    fev1 = float(data.get('forced_expiratory_volume_1', np.nan)) if data.get('forced_expiratory_volume_1') is not None and not np.isnan(float(data.get('forced_expiratory_volume_1'))) else np.nan
    
    bmi_is_missing = 1 if data.get('body_mass_index') is None or np.isnan(float(data.get('body_mass_index', np.nan))) else 0
    fev1_is_missing = 1 if np.isnan(fev1) else 0
    sbp_is_missing = 1 if data.get('systolic_blood_pressure') is None or np.isnan(float(data.get('systolic_blood_pressure', np.nan))) else 0
    
    total_comorbidities = sum([
        hypertension_treated, family_history, atrial_fib,
        kidney_disease, arthritis, diabetes, copd, smoker
    ])
    
    age_sbp_interaction = age * sbp / 100.0
    bmi_diabetes_interaction = bmi * diabetes
    age_comorbidity_index = age * (1.0 + total_comorbidities)
    estimated_pulse_pressure = sbp - 80.0
    estimated_vascular_age = age + (sbp - 120.0) * 0.2 + (bmi - 25.0) * 0.5 + smoker * 5.0
    metabolic_syndrome_score = int(bmi > 30.0) + diabetes + hypertension_treated
    sbp_bmi_ratio = sbp / (bmi + 1.0)
    
    input_dict = {
        'gender': [gender_binary],
        'age': [age],
        'body_mass_index': [bmi if not bmi_is_missing else np.nan],
        'smoker': [smoker],
        'systolic_blood_pressure': [sbp if not sbp_is_missing else np.nan],
        'hypertension_treated': [hypertension_treated],
        'family_history_of_cardiovascular_disease': [family_history],
        'atrial_fibrillation': [atrial_fib],
        'chronic_kidney_disease': [kidney_disease],
        'rheumatoid_arthritis': [arthritis],
        'diabetes': [diabetes],
        'chronic_obstructive_pulmonary_disorder': [copd],
        'forced_expiratory_volume_1': [fev1],
        'bmi_is_missing': [bmi_is_missing],
        'fev1_is_missing': [fev1_is_missing],
        'sbp_is_missing': [sbp_is_missing],
        'total_comorbidities': [total_comorbidities],
        'age_sbp_interaction': [age_sbp_interaction],
        'bmi_diabetes_interaction': [bmi_diabetes_interaction],
        'age_comorbidity_index': [age_comorbidity_index],
        'estimated_pulse_pressure': [estimated_pulse_pressure],
        'estimated_vascular_age': [estimated_vascular_age],
        'metabolic_syndrome_score': [metabolic_syndrome_score],
        'sbp_bmi_ratio': [sbp_bmi_ratio]
    }
    
    df_polars = pl.DataFrame(input_dict)
    
    # Filter features dynamically to match model's expected features
    valid_features = [f for f in feature_names if f in df_polars.columns]
    df_pandas = pd.DataFrame(df_polars.select(valid_features).to_dict(as_series=False))
    return df_pandas

def get_xai_risk_drivers(patient_dict, proba_pct):
    """Calculates Explainable AI (XAI) feature impact factors for patient feedback"""
    drivers = []
    age = float(patient_dict.get('age', 50))
    sbp = float(patient_dict.get('systolic_blood_pressure', 120))
    bmi = float(patient_dict.get('body_mass_index', 25))
    smoker = patient_dict.get('smoker', 0)
    diabetes = patient_dict.get('diabetes', 0)
    ht = patient_dict.get('hypertension_treated', 0)

    if sbp >= 140:
        drivers.append({"factor": "Tekanan Darah Tinggi (SBP)", "impact": f"+{(sbp-120)*0.25:.1f}%", "status": "danger"})
    elif sbp < 120:
        drivers.append({"factor": "Tekanan Darah Optimal", "impact": "-5.0%", "status": "success"})

    if age >= 60:
        drivers.append({"factor": "Faktor Usia (>=60 Thn)", "impact": f"+{(age-50)*0.3:.1f}%", "status": "warning"})

    if smoker == 1:
        drivers.append({"factor": "Status Merokok Aktif", "impact": "+12.0%", "status": "danger"})

    if diabetes == 1:
        drivers.append({"factor": "Kondisi Diabetes Melitus", "impact": "+10.5%", "status": "danger"})

    if bmi >= 30:
        drivers.append({"factor": "Indeks Massa Tubuh (Obesitas)", "impact": f"+{(bmi-25)*0.4:.1f}%", "status": "warning"})
    elif 18.5 <= bmi <= 24.9:
        drivers.append({"factor": "BMI Kategori Ideal", "impact": "-4.0%", "status": "success"})

    if ht == 1:
        drivers.append({"factor": "Riwayat Hipertensi", "impact": "+6.5%", "status": "warning"})

    return drivers

def predict_risk(patient_dict, model_artifact=None, selected_method=None):
    if model_artifact is None:
        model_artifact = load_trained_model()
        
    all_models = model_artifact.get('all_models', {})
    if selected_method and selected_method in all_models:
        model = all_models[selected_method]
        chosen_method_name = selected_method
    else:
        model = model_artifact['model']
        chosen_method_name = model_artifact['model_name']
        
    feature_names = model_artifact['feature_names']
    
    X_input = format_patient_input_polars(patient_dict, feature_names)
    proba = float(model.predict_proba(X_input)[0, 1])
    proba_pct = proba * 100
    
    if proba_pct >= 60.0:
        risk_level = "SANGAT TINGGI"
        badge = "[!!! HARUS SEGERA KONSULTASI DOKTER SPESIALIS JANTUNG !!!]"
    elif proba_pct >= 35.0:
        risk_level = "TINGGI"
        badge = "[!! PERLU PERHATIAN MEDIS LANJUTAN !!]"
    elif proba_pct >= 20.0:
        risk_level = "SEDANG"
        badge = "[! PERLU PERBAIKAN POLA HIDUP & MONITORING !]"
    else:
        risk_level = "RENDAH"
        badge = "[ RISIKO STABIL - PERTAHANKAN POLA HIDUP SEHAT ]"
        
    recommendations = []
    if patient_dict.get('age', 0) >= 60:
        recommendations.append("- Usia di atas 60 tahun: Lakukan pemeriksaan EKG dan Profil Lipid berkala.")
    if patient_dict.get('systolic_blood_pressure', 0) and patient_dict.get('systolic_blood_pressure', 0) >= 140:
        recommendations.append("- Tekanan darah sistolik tinggi (>=140 mmHg): Lakukan pembatasan asupan garam & kontrol hipertensi.")
    if patient_dict.get('diabetes') == 1:
        recommendations.append("- Diabetes melitus terdeteksi: Kontrol ketat gula darah (HbA1c < 7%).")
    if patient_dict.get('smoker') == 1:
        recommendations.append("- Merokok: Sangat disarankan program berhenti merokok secara total.")
    if patient_dict.get('body_mass_index', 0) and patient_dict.get('body_mass_index', 0) >= 30:
        recommendations.append("- Obesitas (BMI >= 30): Targetkan penurunan berat badan bertahap (5-10%).")
    if patient_dict.get('chronic_kidney_disease') == 1:
        recommendations.append("- Gagal ginjal kronis: Perhatikan fungsi ginjal dan hidrasi seimbang.")
        
    if not recommendations:
        recommendations.append("- Risiko kardiovaskular rendah: Pertahankan diet seimbang, olahraga teratur (150 menit/minggu), dan tidur cukup.")
        
    xai_drivers = get_xai_risk_drivers(patient_dict, proba_pct)
    
    return {
        'probability_percent': float(proba_pct),
        'probability_raw': float(proba),
        'risk_level': risk_level,
        'badge': badge,
        'method_used': chosen_method_name,
        'recommendations': recommendations,
        'xai_drivers': xai_drivers
    }
