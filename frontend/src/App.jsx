import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Heart, Activity, Brain, BarChart3, LineChart, FileText, Upload, 
  Sparkles, ShieldCheck, AlertTriangle, CheckCircle2, ArrowRight, 
  RefreshCw, Cpu, Layers, Sliders, Info
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid 
} from 'recharts';

const API_BASE = 'http://127.0.0.1:5000';

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [metrics, setMetrics] = useState([]);
  const [bestModel, setBestModel] = useState('SuperEnsemble AI');

  // Form State
  const [form, setForm] = useState({
    selected_method: 'Auto Best Model (SMOTE Ensemble)',
    gender: 'M',
    age: 58,
    systolic_blood_pressure: 145,
    body_mass_index: 29.4,
    smoker: 1,
    hypertension_treated: 1,
    diabetes: 0,
    family_history_of_cardiovascular_disease: 1,
    atrial_fibrillation: 0,
    chronic_kidney_disease: 0
  });

  // Prediction Output State
  const [result, setResult] = useState(null);

  // What-If Simulator State
  const [simSbp, setSimSbp] = useState(120);
  const [simBmi, setSimBmi] = useState(24.0);
  const [simResult, setSimResult] = useState(null);

  // Batch CSV State
  const [batchFile, setBatchFile] = useState(null);
  const [batchData, setBatchData] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Load Metrics on Mount
  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/metrics`);
      if (res.data.status === 'success') {
        setMetrics(res.data.data);
        setBestModel(res.data.best_model || 'SuperEnsemble AI');
      }
    } catch (e) {
      console.warn("Could not fetch metrics from backend", e);
    }
  };

  const loadPreset = (type) => {
    if (type === 'high') {
      setForm({
        selected_method: 'Auto Best Model (SMOTE Ensemble)',
        gender: 'M', age: 67, systolic_blood_pressure: 168, body_mass_index: 33.5,
        smoker: 1, hypertension_treated: 1, diabetes: 1,
        family_history_of_cardiovascular_disease: 1, atrial_fibrillation: 1, chronic_kidney_disease: 0
      });
    } else if (type === 'medium') {
      setForm({
        selected_method: 'Auto Best Model (SMOTE Ensemble)',
        gender: 'M', age: 54, systolic_blood_pressure: 142, body_mass_index: 28.2,
        smoker: 1, hypertension_treated: 1, diabetes: 0,
        family_history_of_cardiovascular_disease: 1, atrial_fibrillation: 0, chronic_kidney_disease: 0
      });
    } else {
      setForm({
        selected_method: 'Auto Best Model (SMOTE Ensemble)',
        gender: 'F', age: 34, systolic_blood_pressure: 118, body_mass_index: 21.8,
        smoker: 0, hypertension_treated: 0, diabetes: 0,
        family_history_of_cardiovascular_disease: 0, atrial_fibrillation: 0, chronic_kidney_disease: 0
      });
    }
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/api/predict`, form);
      if (res.data.status === 'success') {
        setResult(res.data.data);
        // Sync simulator initial sliders
        setSimSbp(form.systolic_blood_pressure);
        setSimBmi(form.body_mass_index);
        calculateWhatIf(form.systolic_blood_pressure, form.body_mass_index, res.data.data.probability_percent);
      }
    } catch (err) {
      alert("Error calculating prediction: " + (err.response?.data?.message || err.message));
    } finally {
      setLoading(false);
    }
  };

  const calculateWhatIf = async (sbpVal, bmiVal, baseProb = null) => {
    const payload = { ...form, systolic_blood_pressure: sbpVal, body_mass_index: bmiVal };
    try {
      const res = await axios.post(`${API_BASE}/api/predict`, payload);
      if (res.data.status === 'success') {
        const newPct = res.data.probability_percent;
        const currentBase = baseProb !== null ? baseProb : (result ? result.probability_percent : newPct);
        const diff = newPct - currentBase;
        setSimResult({ prob: newPct, diff: diff });
      }
    } catch (e) {}
  };

  const handleSimSbpChange = (val) => {
    setSimSbp(val);
    calculateWhatIf(val, simBmi);
  };

  const handleSimBmiChange = (val) => {
    setSimBmi(val);
    calculateWhatIf(simSbp, val);
  };

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      const res = await axios.post(`${API_BASE}/api/train`);
      if (res.data.status === 'success') {
        alert(`✅ Model Retraining Selesai! Model terbaik terpilih: ${res.data.data.best_model} (ROC-AUC: ${(res.data.data.metrics.roc_auc * 100).toFixed(1)}%)`);
        fetchMetrics();
      }
    } catch (err) {
      alert("Gagal melakukan retraining model: " + err.message);
    } finally {
      setRetraining(false);
    }
  };

  const handleDownloadPdf = () => {
    if (!result) {
      alert("Silakan hitung prediksi risiko pasien terlebih dahulu.");
      return;
    }
    const payload = {
      patient_age: form.age,
      patient_gender: form.gender === 'M' ? 'Pria' : 'Wanita',
      systolic_blood_pressure: form.systolic_blood_pressure,
      body_mass_index: form.body_mass_index,
      risk_level: result.risk_level,
      probability_percent: result.probability_percent,
      method_used: result.method_used,
      recommendations: result.recommendations
    };
    
    axios.post(`${API_BASE}/api/generate-pdf`, payload, { responseType: 'blob' })
      .then(response => {
        const blob = new Blob([response.data], { type: 'application/pdf' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', `Laporan_Pasien_${form.age}th.pdf`);
        document.body.appendChild(link);
        link.click();
        link.remove();
      })
      .catch(err => alert("Gagal mengunduh PDF: " + err.message));
  };

  const handleBatchSubmit = async (e) => {
    e.preventDefault();
    if (!batchFile) {
      alert("Pilih file CSV terlebih dahulu.");
      return;
    }
    setBatchLoading(true);
    const formData = new FormData();
    formData.append('file', batchFile);
    try {
      const res = await axios.post(`${API_BASE}/api/batch-predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      if (res.data.status === 'success') {
        setBatchData(res.data.data);
      }
    } catch (err) {
      alert("Gagal memproses file CSV batch.");
    } finally {
      setBatchLoading(false);
    }
  };

  // Format Recharts data
  const chartData = metrics.map(m => ({
    name: m.Model.replace(' Classifier', '').replace(' Deep Learning (MLP)', ' Deep Learning'),
    Accuracy: parseFloat(m.Accuracy),
    RocAuc: parseFloat(m['ROC-AUC'])
  }));

  return (
    <div className="min-h-screen pb-16">
      {/* Top Header */}
      <header className="sticky top-0 z-50 bg-[#0f172a]/80 backdrop-blur-xl border-b border-white/10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-rose-500 to-pink-500 shadow-lg shadow-rose-500/30">
              <Heart className="w-6 h-6 text-white pulse-icon" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-white bg-clip-text text-transparent">
                  CARDIO<span className="text-rose-500">GUARD</span> AI
                </h1>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                  CLINICAL V2.5 REACT
                </span>
              </div>
              <p className="text-xs text-slate-400">Polars High-Performance Engine & Multi-Model Deep Learning</p>
            </div>
          </div>

          {/* Quick Stats & Server Badge */}
          <div className="flex items-center gap-3">
            <button 
              onClick={handleRetrain}
              disabled={retraining}
              className="btn-secondary flex items-center gap-2 text-xs py-2 px-3 bg-slate-800 hover:bg-slate-700 text-cyan-400 border-cyan-500/30"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${retraining ? 'animate-spin' : ''}`} />
              {retraining ? 'Retraining All Models...' : 'Latih Ulang Semua Model'}
            </button>
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
              Flask API Active :5000
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 md:px-6 pt-6">
        {/* Navigation Tabs */}
        <div className="flex flex-wrap gap-2 p-1.5 rounded-2xl bg-slate-900/80 border border-white/10 mb-8">
          {[
            { id: 'predict', label: '🧮 Single & Batch Prediction', icon: Cpu },
            { id: 'eda', label: '📊 Polars EDA Multi-Panel', icon: BarChart3 },
            { id: 'dl', label: '🧠 Deep Learning JST & Metrics', icon: Brain },
            { id: 'plots', label: '📈 Evaluation & ROC Curves', icon: LineChart }
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 min-w-[200px] flex items-center justify-center gap-2 py-3 px-4 rounded-xl text-sm font-bold transition-all ${
                  isActive 
                    ? 'bg-gradient-to-r from-sky-500 to-blue-600 text-white shadow-lg shadow-blue-500/25 border border-sky-400/30' 
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* TAB 1: PREDICTION & WHAT-IF SIMULATOR */}
        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Input Form */}
            <div className="lg:col-span-7 space-y-6">
              <div className="glass-card">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2 font-bold text-lg text-sky-400">
                    <Sliders className="w-5 h-5" />
                    Input Indikator Klinis Pasien
                  </div>
                  {/* Presets */}
                  <div className="flex gap-1.5">
                    <button onClick={() => loadPreset('low')} className="text-[11px] px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20">Sehat</button>
                    <button onClick={() => loadPreset('medium')} className="text-[11px] px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20">Sedang</button>
                    <button onClick={() => loadPreset('high')} className="text-[11px] px-2.5 py-1 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20 hover:bg-rose-500/20">Tinggi</button>
                  </div>
                </div>

                <form onSubmit={handlePredict} className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label>Algoritma Machine Learning</label>
                      <select 
                        value={form.selected_method} 
                        onChange={e => setForm({...form, selected_method: e.target.value})}
                      >
                        <option value="Auto Best Model (SMOTE Ensemble)">⚡ Auto Best Model (SMOTE Ensemble)</option>
                        <option value="SuperEnsemble AI">🏆 SuperEnsemble AI</option>
                        <option value="XGBoost">⚡ XGBoost Classifier</option>
                        <option value="LightGBM Classifier">🚀 LightGBM Classifier</option>
                        <option value="Random Forest">🌲 Random Forest</option>
                        <option value="Deep Learning (MLP)">🧠 Deep Learning (MLP Neural Net)</option>
                        <option value="SVM">🛡️ Calibrated Support Vector Machine</option>
                        <option value="Logistic Regression">📈 Logistic Regression</option>
                        <option value="Decision Tree">🌿 Decision Tree</option>
                      </select>
                    </div>

                    <div>
                      <label>Jenis Kelamin Pasien</label>
                      <select value={form.gender} onChange={e => setForm({...form, gender: e.target.value})}>
                        <option value="M">Laki-laki (Pria)</option>
                        <option value="F">Perempuan (Wanita)</option>
                      </select>
                    </div>

                    <div>
                      <label>Usia Pasien (Tahun)</label>
                      <input 
                        type="number" 
                        value={form.age} 
                        min="18" max="100" 
                        onChange={e => setForm({...form, age: parseFloat(e.target.value) || 0})}
                      />
                    </div>

                    <div>
                      <label>Tekanan Darah Sistolik / SBP (mmHg)</label>
                      <input 
                        type="number" 
                        value={form.systolic_blood_pressure} 
                        min="70" max="240" 
                        onChange={e => setForm({...form, systolic_blood_pressure: parseFloat(e.target.value) || 0})}
                      />
                    </div>

                    <div className="sm:col-span-2">
                      <label>Indeks Massa Tubuh / BMI (kg/m²)</label>
                      <input 
                        type="number" step="0.1" 
                        value={form.body_mass_index} 
                        min="10" max="60" 
                        onChange={e => setForm({...form, body_mass_index: parseFloat(e.target.value) || 0})}
                      />
                    </div>
                  </div>

                  {/* Medical Checkboxes */}
                  <div className="pt-2">
                    <label className="mb-2">Kondisi Kesehatan & Factor Risiko</label>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                      {[
                        { key: 'smoker', label: 'Perokok Aktif' },
                        { key: 'hypertension_treated', label: 'Hipertensi Berobat' },
                        { key: 'diabetes', label: 'Diabetes Melitus' },
                        { key: 'family_history_of_cardiovascular_disease', label: 'Riwayat Jantung' },
                        { key: 'atrial_fibrillation', label: 'Fibrilasi Atrium' },
                        { key: 'chronic_kidney_disease', label: 'Gagal Ginjal Kronis' }
                      ].map(item => (
                        <label key={item.key} className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-900/60 border border-white/5 cursor-pointer hover:border-sky-500/40 transition-colors">
                          <input 
                            type="checkbox" 
                            checked={form[item.key] === 1}
                            onChange={e => setForm({...form, [item.key]: e.target.checked ? 1 : 0})}
                            className="rounded accent-sky-500"
                          />
                          <span className="text-xs text-slate-200">{item.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <button 
                    type="submit" 
                    disabled={loading}
                    className="w-full btn-primary py-3.5 mt-4 text-base"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        Memproses Inferensi Model AI...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5" />
                        Hitung Prediksi Risiko Kardiovaskular Pasien
                      </>
                    )}
                  </button>
                </form>
              </div>

              {/* What-If Scenario Simulator Card */}
              <div className="glass-card border-sky-500/30">
                <div className="flex items-center gap-2 font-bold text-sky-400 mb-2">
                  <Activity className="w-5 h-5" />
                  Simulator Penurunan Risiko Real-Time ("What-If Scenario")
                </div>
                <p className="text-xs text-slate-400 mb-4">
                  Geser slider indikator medis untuk melihat perubahan risiko kardiovaskular pasien secara instan.
                </p>

                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300">Target Tekanan Darah (SBP)</span>
                      <span className="font-bold text-sky-400">{simSbp} mmHg</span>
                    </div>
                    <input 
                      type="range" min="90" max="200" value={simSbp}
                      onChange={e => handleSimSbpChange(parseFloat(e.target.value))}
                      className="w-full accent-sky-400 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-300">Target Indeks Massa Tubuh (BMI)</span>
                      <span className="font-bold text-purple-400">{simBmi}</span>
                    </div>
                    <input 
                      type="range" min="15" max="45" step="0.5" value={simBmi}
                      onChange={e => handleSimBmiChange(parseFloat(e.target.value))}
                      className="w-full accent-purple-400 cursor-pointer"
                    />
                  </div>

                  <div className="p-4 rounded-xl bg-sky-500/10 border border-sky-500/20 text-center">
                    <div className="text-xs text-slate-400">Hasil Simulasi Probabilitas Baru:</div>
                    <div className="text-2xl font-black text-emerald-400 my-1">
                      {simResult ? `${simResult.prob.toFixed(1)}%` : '--%'}
                    </div>
                    <div className="text-xs font-semibold text-sky-400">
                      {simResult ? (
                        simResult.diff < -0.05 ? `⚡ Potensi Penurunan Risiko: ${simResult.diff.toFixed(1)}%` :
                        simResult.diff > 0.05 ? `⚠️ Kenaikan Risiko: +${simResult.diff.toFixed(1)}%` :
                        `⚡ Probabilitas Stabil`
                      ) : 'Silakan hitung prediksi awal.'}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Results & Output */}
            <div className="lg:col-span-5 space-y-6">
              {/* Output Card */}
              <div className="glass-card relative overflow-hidden">
                <div className="flex items-center gap-2 font-bold text-lg text-slate-100 mb-4">
                  <ShieldCheck className="w-5 h-5 text-emerald-400" />
                  Hasil Analisis AI & Rekomendasi Medis
                </div>

                {result ? (
                  <div className="space-y-5">
                    {/* Badge & Proba */}
                    <div className="p-5 rounded-2xl bg-slate-900/80 border border-white/10 text-center space-y-3">
                      <div className="text-xs text-slate-400">
                        Algoritma Dipakai: <span className="font-bold text-sky-400">{result.method_used}</span>
                      </div>
                      
                      <div className="text-5xl font-black tracking-tight text-white">
                        {result.probability_percent.toFixed(1)}<span className="text-2xl text-slate-400">%</span>
                      </div>

                      {/* Risk Bar */}
                      <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden">
                        <div 
                          className={`h-full transition-all duration-700 ${
                            result.risk_level.includes('TINGGI') ? 'bg-rose-500' :
                            result.risk_level === 'SEDANG' ? 'bg-amber-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${result.probability_percent}%` }}
                        ></div>
                      </div>

                      <div className={`inline-block px-4 py-1.5 rounded-full text-xs font-extrabold tracking-wider ${
                        result.risk_level.includes('TINGGI') ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40' :
                        result.risk_level === 'SEDANG' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' :
                        'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                      }`}>
                        {result.risk_level}
                      </div>

                      <button 
                        onClick={handleDownloadPdf}
                        className="w-full py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-sky-400 font-bold text-xs border border-sky-500/30 flex items-center justify-center gap-2 transition-all mt-2"
                      >
                        <FileText className="w-4 h-4" />
                        Download Laporan Medis PDF Pasien
                      </button>
                    </div>

                    {/* XAI Explainable Drivers */}
                    <div>
                      <div className="text-xs font-bold text-pink-400 mb-2 flex items-center gap-1.5">
                        <Sparkles className="w-4 h-4" />
                        Explainable AI (XAI) - Kontributor Utama:
                      </div>
                      <div className="space-y-1.5">
                        {result.xai_drivers && result.xai_drivers.map((d, idx) => (
                          <div 
                            key={idx}
                            className={`p-2.5 rounded-xl flex items-center justify-between text-xs border ${
                              d.status === 'danger' ? 'bg-rose-500/10 border-rose-500/20 text-rose-300' :
                              d.status === 'warning' ? 'bg-amber-500/10 border-amber-500/20 text-amber-300' :
                              'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                            }`}
                          >
                            <span>{d.factor}</span>
                            <span className="font-bold">{d.impact}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Recommendations */}
                    <div>
                      <div className="text-xs font-bold text-sky-400 mb-2">Saran Intervensi Medis Terintegrasi:</div>
                      <div className="space-y-2">
                        {result.recommendations.map((rec, i) => (
                          <div key={i} className="p-3 rounded-xl bg-slate-900/70 border-l-4 border-sky-400 text-xs text-slate-300 leading-relaxed">
                            {rec.replace('- ', '')}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="py-12 text-center text-slate-500 space-y-3">
                    <Info className="w-10 h-10 mx-auto text-slate-600 animate-bounce" />
                    <p className="text-sm">Isi data pasien dan klik 'Hitung Prediksi' untuk menampilkan hasil analisis AI.</p>
                  </div>
                )}
              </div>

              {/* Batch Processing CSV Box */}
              <div className="glass-card">
                <div className="flex items-center gap-2 font-bold text-slate-200 mb-2">
                  <Upload className="w-5 h-5 text-purple-400" />
                  Batch Process CSV (Polars High-Speed Engine)
                </div>
                <p className="text-xs text-slate-400 mb-4">Unggah file Data_Penyakit_Jantung.csv untuk menghitung ribuan data sekaligus.</p>

                <form onSubmit={handleBatchSubmit} className="space-y-3">
                  <input 
                    type="file" accept=".csv"
                    onChange={e => setBatchFile(e.target.files[0])}
                    className="text-xs text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-purple-500/20 file:text-purple-300 hover:file:bg-purple-500/30"
                  />
                  <button 
                    type="submit" 
                    disabled={batchLoading}
                    className="w-full py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs transition-all flex items-center justify-center gap-2"
                  >
                    {batchLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : '⚡ Process CSV via Polars Engine'}
                  </button>
                </form>

                {batchData && (
                  <div className="mt-4 pt-4 border-t border-white/10 space-y-3">
                    <div className="grid grid-cols-4 gap-2 text-center text-xs">
                      <div className="p-2 rounded-lg bg-slate-800"><div className="text-slate-400 text-[10px]">Total</div><div className="font-bold">{batchData.total}</div></div>
                      <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400"><div className="text-[10px]">Tinggi</div><div className="font-bold">{batchData.high_risk}</div></div>
                      <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400"><div className="text-[10px]">Sedang</div><div className="font-bold">{batchData.med_risk}</div></div>
                      <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400"><div className="text-[10px]">Rendah</div><div className="font-bold">{batchData.low_risk}</div></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: POLARS EDA MULTI-PANEL */}
        {activeTab === 'eda' && (
          <div className="space-y-6">
            <div className="glass-card">
              <h2 className="text-lg font-bold text-cyan-400 mb-2">📊 Exploratory Data Analysis (EDA) - Polars High-Speed Analytics</h2>
              <p className="text-xs text-slate-400">Visualisasi statistik distribusi dan korelasi 23 fitur klinis.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass-card space-y-3">
                <h3 className="text-sm font-bold text-slate-200">1. Distribusi Kelas Target & Risiko Jantung</h3>
                <img src={`${API_BASE}/plots/eda_target_distribution.png`} alt="Target Distribution" className="w-full h-auto rounded-xl bg-white/5 border border-white/10" />
              </div>
              <div className="glass-card space-y-3">
                <h3 className="text-sm font-bold text-slate-200">2. Hubungan Tekanan Darah (SBP) & Usia</h3>
                <img src={`${API_BASE}/plots/eda_clinical_visualization.png`} alt="Clinical Boxplots" className="w-full h-auto rounded-xl bg-white/5 border border-white/10" />
              </div>
            </div>

            <div className="glass-card space-y-3">
              <h3 className="text-sm font-bold text-slate-200">3. Matrix Korelasi 23 Fitur Medis</h3>
              <img src={`${API_BASE}/plots/eda_correlation_matrix.png`} alt="Correlation Matrix" className="w-full h-auto max-h-[600px] object-contain mx-auto rounded-xl bg-white/5 border border-white/10" />
            </div>
          </div>
        )}

        {/* TAB 3: DEEP LEARNING JST & BENCHMARK METRICS */}
        {activeTab === 'dl' && (
          <div className="space-y-6">
            {/* Interactive Benchmark Chart */}
            <div className="glass-card space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-bold text-pink-400">🧠 Benchmark Model Machine Learning & Deep Learning</h2>
                  <p className="text-xs text-slate-400">Perbandingan akurasi & skor ROC-AUC antar algoritma klasifikasi</p>
                </div>
                <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-bold">
                  Terbaik: {bestModel}
                </span>
              </div>

              <div className="h-64 w-full pt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                    <YAxis domain={[70, 100]} stroke="#94a3b8" fontSize={11} />
                    <Tooltip contentStyle={{ background: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#fff' }} />
                    <Bar dataKey="ROC-AUC" name="ROC-AUC %" fill="#38bdf8" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="Accuracy" name="Akurasi %" fill="#a855f7" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Table */}
              <div className="overflow-x-auto pt-4">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-900/80 text-sky-400 uppercase text-[10px] tracking-wider border-b border-white/10">
                    <tr>
                      <th className="p-3">Model</th>
                      <th className="p-3 text-center">Accuracy</th>
                      <th className="p-3 text-center">Precision</th>
                      <th className="p-3 text-center">Recall</th>
                      <th className="p-3 text-center">F1-Score</th>
                      <th className="p-3 text-center">ROC-AUC</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {metrics.map((row, idx) => (
                      <tr key={idx} className="hover:bg-white/5 transition-colors">
                        <td className="p-3 font-semibold text-slate-200">{row.Model}</td>
                        <td className="p-3 text-center">{row.Accuracy}</td>
                        <td className="p-3 text-center">{row.Precision}</td>
                        <td className="p-3 text-center">{row.Recall}</td>
                        <td className="p-3 text-center">{row.F1}</td>
                        <td className="p-3 text-center font-bold text-sky-400">{row['ROC-AUC']}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Neural Network High-Res Diagram */}
            <div className="glass-card space-y-4">
              <h3 className="text-base font-bold text-pink-400">⚡ Topologi High-Resolution Jaringan Saraf Tiruan (JST / MLP)</h3>
              <img src={`${API_BASE}/plots/nn_architecture_diagram.png`} alt="JST Architecture Diagram" className="w-full h-auto rounded-xl bg-[#0b0f19] border border-white/10" />
            </div>

            <div className="glass-card space-y-4">
              <h3 className="text-base font-bold text-sky-400">📉 Kurva Iterasi Loss Training Deep Learning</h3>
              <img src={`${API_BASE}/plots/dl_loss_curve.png`} alt="Deep Learning Loss Curve" className="w-full h-auto rounded-xl bg-white/5 border border-white/10" />
            </div>
          </div>
        )}

        {/* TAB 4: EVALUATION PLOTS */}
        {activeTab === 'plots' && (
          <div className="space-y-6">
            <div className="glass-card">
              <h2 className="text-lg font-bold text-sky-400 mb-2">📈 Evaluasi Performa Model & Analisis Kurva ROC</h2>
              <p className="text-xs text-slate-400">Kurva ROC-AUC, Matriks Kebingungan (Confusion Matrix), dan Permutation Feature Importance.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass-card space-y-3">
                <h3 className="text-sm font-bold text-slate-200">1. Kurva ROC-AUC Semua Model</h3>
                <img src={`${API_BASE}/plots/roc_curve.png`} alt="ROC Curve" className="w-full h-auto rounded-xl bg-white/5 border border-white/10" />
              </div>
              <div className="glass-card space-y-3">
                <h3 className="text-sm font-bold text-slate-200">2. Confusion Matrix Model Terbaik</h3>
                <img src={`${API_BASE}/plots/confusion_matrix.png`} alt="Confusion Matrix" className="w-full h-auto rounded-xl bg-white/5 border border-white/10" />
              </div>
            </div>

            <div className="glass-card space-y-3">
              <h3 className="text-sm font-bold text-slate-200">3. Tingkat Kepentingan Fitur Medis (Feature Importance)</h3>
              <img src={`${API_BASE}/plots/feature_importance.png`} alt="Feature Importance" className="w-full h-auto rounded-xl bg-white/5 border border-white/10" />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
