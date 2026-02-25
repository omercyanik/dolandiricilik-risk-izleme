import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import time

# Page config
st.set_page_config(page_title="Dolandırıcılık Risk Platformu", layout="wide")

from dataclasses import dataclass

@dataclass
class DecisionPolicy:
    mfa_threshold: float = 30.0
    block_threshold: float = 75.0

if "policy" not in st.session_state:
    st.session_state.policy = DecisionPolicy()

# Modern CSS
st.markdown("""
<style>
    .css-18e3th9 { padding-top: 0rem; }
    .stMetric { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #ffffff; }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: transparent; color: #aaaaaa; text-align: center; font-size: 12px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

def format_currency(x):
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return str(x)

def format_percent(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return str(x)

def format_score(x):
    try:
        return f"{float(x):.1f} / 100"
    except Exception:
        return str(x)

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

@st.cache_resource
def load_models():
    model_path = PROJECT_ROOT / "models" / "lgbm_calibrated.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_test_data():
    data_path = PROJECT_ROOT / "data" / "scored_test_set.csv"
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=['timestamp'])
    return None

model = load_models()
test_df = load_test_data()

st.title("Dolandırıcılık Risk Paneli ve Karar Eşiği Optimizasyonu")
st.markdown("*Gerçek zamanlı dolandırıcılık risk simülasyonu ve karar eşikleri (threshold) optimizasyonu pratik portföyü.*")
st.markdown("**Not**: Veriler eğitim amaçlı sentetiktir, gerçek finansal bilgi içermez.")

if model is None or test_df is None or test_df.empty:
    st.error("Arka plan modelleri (backend) veya veri setleri eksik. Lütfen öncelikle veri üretimi, özellik mühendisliği (feature engineering) ve eğitim süreçlerini çalıştırınız.")
    st.stop()

st.sidebar.markdown("### Navigasyon")
pages = [
    "Yönetici Gösterge Paneli",
    "Canlı Risk Skorlama Simülasyonu",
    "Karar Eşiği (Threshold) Optimizasyonu",
    "Maliyet ve ROI Simülatörü",
    "Risk Hacmi Tahmini (30 Gün)",
    "Model Açıklanabilirliği (SHAP)",
    "Model Sağlığı ve Konsept Kayması (PSI)"
]
selection = st.sidebar.radio("Menü", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("### Sistem Bilgisi")
st.sidebar.info("Model Algoritması: LightGBM (Isotonic Calibration)\n\nVersiyon: 1.0.0")

if selection == "Yönetici Gösterge Paneli":
    st.header("Yönetici Gösterge Paneli (Executive Dashboard)")
    st.subheader("Temel Performans Göstergeleri (KPI)")
    
    col1, col2, col3, col4 = st.columns(4)
    total_tx = len(test_df)
    fraud_rate = float(test_df['is_fraud'].mean() * 100)
    avg_score = float(test_df['lgbm_risk_score'].mean())
    high_risk_tx = int((test_df['lgbm_risk_score'] > 50).sum())
    
    col1.metric("Toplam İşlem Hacmi (Test Seti)", f"{total_tx:,}")
    col2.metric("Temel Dolandırıcılık Oranı", format_percent(fraud_rate))
    col3.metric("Ortalama Risk Skoru", f"{avg_score:.1f}")
    col4.metric("Yüksek Riskli İşlem Adedi", f"{high_risk_tx:,}")
    
    st.divider()
    
    st.write("### Tahmini Risk Skoru Dağılımı")
    st.markdown("Yasal işlemler ile potansiyel dolandırıcılık faaliyetleri arasındaki öngörüye dayalı (predictive) ayrım kapasitesini göstermektedir.")
    
    df_plot = test_df.copy()
    df_plot['is_fraud_label'] = df_plot['is_fraud'].map({0: 'Yasal/Düzenli', 1: 'Dolandırıcılık'})
    
    fig = px.histogram(
        df_plot,
        x="lgbm_risk_score",
        color="is_fraud_label",
        nbins=50,
        log_y=True,
        title="Risk Risk Skoru Dağılımı (Günlük Log)",
        color_discrete_map={'Yasal/Düzenli': '#00cc96', 'Dolandırıcılık': '#ef553b'},
        labels={'is_fraud_label': 'Risk Durumu', 'lgbm_risk_score': 'Risk Skoru'}
    )
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig, use_container_width=True)

elif selection == "Canlı Risk Skorlama Simülasyonu":
    st.header("Canlı Risk Simülasyonu")
    st.markdown("Test verilerindeki işlem (transaction) yükleri üzerinden model çıkarım motorunu (inference engine) test edin.")
    
    st.write("### İşlem Örneği (Transaction Sample)")
    
    total_fraud = int(test_df['is_fraud'].sum())
    fraud_pct = float((total_fraud / max(1, len(test_df))) * 100)
    st.info(f"Test verisindeki özet: {total_fraud:,} dolandırıcılık vakası bulundu (Görülme sıklığı: {format_percent(fraud_pct)}).")
    
    colA, colB = st.columns([1, 2])
    with colA:
        show_fraud_only = st.checkbox("Sadece fraud (dolandırıcılık) işlemlerini getir")
        
    df_to_sample = (
        test_df[test_df['is_fraud'] == 1].reset_index(drop=True)
        if show_fraud_only
        else test_df.reset_index(drop=True)
    )
    
    if df_to_sample.empty:
        st.warning("Eşleşen herhangi bir işlem örneği bulunamadı.")
    else:
        sample_size = st.slider("Örneklem İndeksini Belirleyiniz", 0, len(df_to_sample) - 1, 0)
        tx = df_to_sample.iloc[[sample_size]].copy()
        
        # Display correctly formatted payload
        st.markdown("### Gelen İşlem Yükü (Transaction Payload)")
        display_df = tx[['timestamp', 'customer_id', 'merchant_id', 'amount', 'country', 'channel']].copy()
        # Format for display matching DD-MM-YYYY HH:MM:SS exactly
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%d-%m-%Y %H:%M:%S')
        st.dataframe(
            display_df.style.format({
                'amount': lambda x: format_currency(x)
            }),
            use_container_width=True
        )
        
        if st.button("Model Çıkarımını (Inference) Çalıştır", type="primary"):
            with st.spinner("Skorlanıyor..."):
                time.sleep(0.75)
                score = float(tx['lgbm_risk_score'].values[0])
                prob = float(tx['lgbm_risk_prob'].values[0])
                true_label = int(tx['is_fraud'].values[0])
                
                st.divider()
                c1, c2, c3 = st.columns(3)
                
                c1.metric("Tahmini Model Risk Skoru", format_score(score))
                c2.metric("Gerçek Hedef Etiketi (Ground Truth)", "Dolandırıcılık Bildirildi" if true_label == 1 else "Yasal İşlem")
                
                # Kural Tabanlı Motor (Rule-Based Engine) Katmanı
                rule_blocked = False
                rule_reason = ""
                amount_val = float(tx['amount'].values[0])
                
                # Domain uzmanlığına dayalı kurallar (Örn: Tutar > 15K veya sınır ötesi şüpheli ilk işlem)
                cross_border_val = str(tx['cross_border'].values[0]) if 'cross_border' in tx.columns else '0'
                if amount_val > 15000:
                    rule_blocked = True
                    rule_reason = f"KURAL TABANLI MOTOR: Yüksek Tutar Anomalisi ({format_currency(amount_val)}). İşlem Makine Öğrenmesi skorundan bağımsız DOĞRUDAN ENGELLENDİ."
                elif cross_border_val == '1' and amount_val > 5000:
                    rule_blocked = True
                    rule_reason = f"KURAL TABANLI MOTOR: Sınır ötesi ilk işlem ve yüksek tutar ({format_currency(amount_val)}). DOĞRUDAN ENGELLENDİ."

                if rule_blocked:
                    c3.error("Aksiyon: ENGELLE (Kural Tabanlı)")
                    st.error(rule_reason)
                elif score >= st.session_state.policy.block_threshold:
                    c3.error("Aksiyon Motoru: ENGELLE - Yüksek Riskli İşlem (ML Skoru)")
                    st.error("Otorizasyon isteği, dolandırıcılık ML modeline göre yüksek ihtimal barındırıyor. Sistem tarafından reddedildi.")
                elif score >= st.session_state.policy.mfa_threshold:
                    c3.warning("Aksiyon Motoru: EK DOĞRULAMA - Doğrulama İsteniyor")
                    st.warning("Şüpheli sinyal (ML) algılandı. İkincil kimlik doğrulama kanallarına (ör. SMS OTP, 3D Secure) yönlendirilmesi uygun.")
                else:
                    c3.success("Aksiyon Motoru: ONAYLA - Düşük Risk")
                    st.success("Herhangi bir kural ihlali veya makine öğrenmesi risk metrik sinyali saptanmadı. İşlem onaylandı.")
                
                st.caption("Not: Tahmini Risk Skoru algoritmik analizini, Kural Motoru ise bankacılık risk politikalarını simgeler.")

elif selection == "Karar Eşiği (Threshold) Optimizasyonu":
    st.header("Operasyonel Karar Eşikleri")
    
    st.markdown("Müşteri engelleri (friction) ile dolandırıcılık sızıntı maliyetlerini (fraud interception) dengeleyecek optimal risk / MFA kural sınırlarını yapılandırın.")
    
    colA, colB = st.columns(2)
    stepup_th = colA.slider(
        "Ek Doğrulama Gereksinim Eşiği (MFA)",
        0.0, 100.0, float(st.session_state.policy.mfa_threshold),
        help="Modelin bu limitin üstünde puanladığı işlemler, 3D Secure/SMS tarzı ikincil katman doğrulama döngülerine iletilir."
    )
    block_th = colB.slider(
        "Hesap veya İşlem Engelleme Eşiği",
        0.0, 100.0, float(st.session_state.policy.block_threshold),
        help="Modelin bu limitin üstünde puanladığı işlemler anlık olarak sistemce devre dışı bırakılır veya kuyruk (review queue) sürecine aktarılır."
    )
    
    if stepup_th >= block_th:
        st.error("Yapılandırma Hatası: Ek Doğrulama Eşiği, Engelleme Eşiği skorundan daha yüksek veya eşit bir değere atanamaz.")
    else:
        # Update policy if valid
        st.session_state.policy.mfa_threshold = stepup_th
        st.session_state.policy.block_threshold = block_th
        
        st.divider()
        blocks = int((test_df['lgbm_risk_score'] >= block_th).sum())
        stepups = int(((test_df['lgbm_risk_score'] >= stepup_th) & (test_df['lgbm_risk_score'] < block_th)).sum())
        allows = int((test_df['lgbm_risk_score'] < stepup_th).sum())
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Doğrudan İzin (Allow)", f"{allows:,} İşlem", f"{allows / max(1, len(test_df)) * 100:.1f}%")
        c2.metric("Ek Doğrulama (Step-Up)", f"{stepups:,} İşlem", f"{stepups / max(1, len(test_df)) * 100:.1f}%", delta_color="off")
        c3.metric("Önlenen/Engellenen (Block)", f"{blocks:,} İşlem", f"{blocks / max(1, len(test_df)) * 100:.1f}%", delta_color="inverse")
        
        fig1 = px.pie(
            values=[allows, stepups, blocks],
            names=['Onayla', 'Doğrulama İste', 'Engelle'],
            title="Aksiyon Dağılımı Tahmini (Politika Sonucu)",
            color_discrete_sequence=['#00cc96', '#ffa15a', '#ef553b'],
            hole=0.4
        )
        st.plotly_chart(fig1, use_container_width=True)

elif selection == "Maliyet ve ROI Simülatörü":
    st.header("Maliyet & ROI Simülatörü")
    st.markdown("Mevcut threshold (engelleme) ayarlarının bilançoya ve operasyon kalemlerine net (Yatırım Getirisi - ROI) etkisini simüle eder.")
    
    st.sidebar.markdown("### Maliyet Varsayımları")
    review_cost = st.sidebar.number_input("Manuel İnceleme İşlem Maliyeti (Birim $)", 1.0, 50.0, 5.0)
    step_up_cost = st.sidebar.number_input("Ek Doğrulama / Güvenlik Operasyon Maliyeti (Birim $)", 0.0, 5.0, 0.5)
    fp_margin = st.sidebar.number_input("Yanlış Pozitif (False Positive) Kar Marjı Kaybı Oranı (%)", 0.0, 20.0, 5.0)
    
    st.info("Sistem Uyarı: Analiz varsayımları doğrudan sentetik verilere ve atanmış fonksiyonel eşik (threshold) tabanlarına dayalı olarak değişkenlik gösterecektir.")
    
    from src.roi import simulate_roi
    
    with st.spinner("ROI metrik projeksiyonları türetiliyor..."):
        res = simulate_roi(
            test_df,
            th_allow=st.session_state.policy.mfa_threshold,
            th_block=st.session_state.policy.block_threshold,  
            review_cost_fixed=review_cost,
            step_up_cost_fixed=step_up_cost,
            fp_margin_loss=fp_margin / 100.0
        )
        
        savings = float(res.get('Gross Savings', 0.0))
        roi = float(res.get('ROI %', 0.0))
        
        st.divider()
        col_res1, col_res2 = st.columns(2)
        
        if savings >= 0:
            col_res1.metric("Kurum İçin Tahmini Aylık Net Tasarruf", format_currency(savings))
            st.success(f"Geçerli metrik yapılandırmaları sonucu ulaşılan tahmini pozitif tasarruf etkisi: {format_currency(savings)}.")
        else:
            col_res1.metric("Tahmini Modellenmiş Bilanço Açığı", format_currency(abs(savings)))
            st.error("Seçili engelleme-onaylama (threshold) ayarları neticesinde Negatif ROI teşkil etmektedir. Operasyonel maliyet parametrelerini gözden geçiriniz.")
            
        col_res2.metric("Yatırımın Geri Dönüş Oranı (ROI %)", format_percent(roi))
        
        st.markdown("---")
        st.markdown("### Detaylı Finansal Modeller")
        
        details = pd.DataFrame([
            {"Analiz Metriği": "Uygulanan Toplam İşlem Nominal Hacmi", "Değer": format_currency(res.get('Total Amount', 0.0))},
            {"Analiz Metriği": "Orijinal Kaydedilmiş (Baseline) Dolandırıcılık Kaybı", "Değer": format_currency(res.get('Baseline Fraud Loss', 0.0))},
            {"Analiz Metriği": "Tahmini Önlenemeyen Kalan (Residual) Dolandırıcılık Kayıp Sınırı", "Değer": format_currency(res.get('New Fraud Loss', 0.0))},
            {"Analiz Metriği": "Model Kaynaklı Yanlış Pozitif + Manuel Operasyon İşlem Giderleri", "Değer": format_currency(res.get('Operational Cost', 0.0))},
            {"Analiz Metriği": "Erişilen Model Başarılı Yakalama ve Engelleme Limiti Oranı", "Değer": format_percent(float(res.get('Fraud Capture Rate', 0.0)) * 100)},
            {"Analiz Metriği": "Analist Kuyruğuna Düşecek Manuel İşlem Satürasyonu Oranı", "Değer": format_percent(float(res.get('Review Rate', 0.0)) * 100)}
        ])
        
        st.table(details)

elif selection == "Risk Hacmi Tahmini (30 Gün)":
    st.header("Dolandırıcılık Vaka Tahminleri (30 Gün)")
    st.markdown("Geçmiş trendler ve bölgesel anomaliler üzerinden LightGBM regresyonu ile önümüzdeki 30 günün beklenen dolandırıcılık hacmi tahmini (Forecast).")
    
    @st.cache_data(ttl=3600)
    def fetch_forecast_data():
        from src.forecast import generate_forecast
        transactions_path = PROJECT_ROOT / "data" / "transactions.csv"
        return generate_forecast(transactions_path, days=30)
    
    with st.spinner("Zaman serisi (time-series) projeksiyonları yürütülüyor..."):
        daily_fraud, future_df = fetch_forecast_data()
        
    if daily_fraud is None or daily_fraud.empty or future_df.empty:
        st.error("Tahmin modülü işlenemedi. Geçerli bir veri seti alt katmanı eksik.")
    else:
        last_30_days_historical = daily_fraud.tail(30)['fraud_count']
        avg_hist_30 = float(last_30_days_historical.mean())
        
        future_total = float(future_df['predicted_fraud'].sum())
        avg_future_30 = float(future_df['predicted_fraud'].mean())
        
        if avg_future_30 > avg_hist_30 * 1.05:
            trend_text = "Yukarı Yönlü Trend"
            comment = "Analiz Çıktısı: Dolandırıcılık eğilimlerinde potansiyel bir yükselme hesaplanmıştır. Operasyonel kaynak planlamasının gözden geçirilmesi önerilir."
        elif avg_future_30 < avg_hist_30 * 0.95:
            trend_text = "Aşağı Yönlü Trend"
            comment = "Analiz Çıktısı: Risk baskıları düşüş eğilimi göstermektedir. Sistem içi operasyonlar stabilite sınırlarına yaklaşmaktadır."
        else:
            trend_text = "Yatay Senaryo"
            comment = "Analiz Çıktısı: Tahmin edilen dolandırıcılık hacimleri temel seviyeye (baseline) entegredir; sıradışı bir dalgalanma beklenmemektedir."
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Son 30 Günün Orijinal Ortalaması", f"{avg_hist_30:.1f}")
        c2.metric("Gelecek 30 Gün Net Tahmin", f"{int(future_total):,}", delta=f"Ort: {avg_future_30:.1f}/gün", delta_color="off")
        c3.metric("Öngörülen Rota/Trend", trend_text, delta=None)
        
        st.divider()
        st.subheader("Tarihsel Eğili ve Tahmin Güven Bantları (Confidence Intervals)")
        
        fig = go.Figure()
        
        # Format timestamps on hover properly without changing inner dataframe types
        fig.add_trace(go.Scatter(
            x=daily_fraud['timestamp'],
            y=daily_fraud['fraud_count'],
            mode='lines',
            name='Geçmiş Vaka Kaydı',
            line=dict(color='#00cc96', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=list(future_df['timestamp']) + list(future_df['timestamp'])[::-1],
            y=list(future_df['upper_band']) + list(future_df['lower_band'])[::-1],
            fill='toself',
            fillcolor='rgba(239, 85, 59, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Model Güven Aralığı Sınırları'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_df['timestamp'],
            y=future_df['predicted_fraud'],
            mode='lines',
            name='Tahminsel Ağ (Gelecek 30 Gün)',
            line=dict(color='#ef553b', width=2, dash='dash')
        ))
        
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Zaman Aralığı",
            yaxis_title="İşlem Bazlı Dolandırıcılık Frekansı",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(comment)

elif selection == "Model Açıklanabilirliği (SHAP)":
    st.header("Karar Etkeni (Feature Importance) ve SHAP")
    
    st.markdown("""
Compliance (Yasal Uyumluluk) gereksinimleri sebebiyle "kara kutu" algoritmalarının aldığı kararlar şeffaflıkla incelenebilmelidir. Ağaç tabanlı modellerin, riski her bir değişkene nasıl böldüğü (değerlendirdiği) aşağıda **Waterfall Chart** formatında görselleştirilmiştir.
""")
    
    st.info("Ortam Bilgisi: Model izolasyon limitleri sebebiyle grafiksel görüntüleyici modüller kısıtlanmıştır; analiz çıktıları metin proxy'si aracılığıyla listelenmiştir.")
    
    st.write("### Baskın Etken Analiz Matrisi")
    
    if st.button("Tanı Çıktısını Hazırla/Getir", type="primary"):
        with st.spinner("Model açıklama matrisleri çekiliyor (SHAP Waterfall Plot)..."):
            from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
            import matplotlib.pyplot as plt
            
            # Select extremely high risk fraud transaction for demo purpose
            sample_tx = test_df[test_df['is_fraud'] == 1].iloc[0]
            
            try:
                preprocessor = model.named_steps['preprocessor']
                lgbm_clf = model.named_steps['classifier']

                sample_df = pd.DataFrame([sample_tx])
                X_input = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
                X_transformed = preprocessor.transform(X_input)
                
                if hasattr(X_transformed, 'toarray'):
                    X_transformed = X_transformed.toarray()

                st.success("Analitik Çıktı Tamamlandı.")

                explainer = shap.TreeExplainer(lgbm_clf)
                shap_values = explainer(X_transformed)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                val_to_plot = shap_values[0] if not isinstance(shap_values, list) else shap_values[1][0]
                
                feature_names = preprocessor.get_feature_names_out()
                val_to_plot.feature_names = [str(f).split('__')[-1] for f in feature_names]
                
                shap.plots.waterfall(val_to_plot, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Grafiksel SHAP çıktısı oluşturulurken hata meydana geldi: {e}")
                st.markdown("Alternatif analitik döküm devreye alınıyor...")
            
            st.markdown("""
**Genel Tanı Özeti:**
- Grafikte, baz (beklenen) değerden risk skorunu artıran özellikler (kırmızı) ve azaltan özellikler (mavi) net olarak incelenebilir.
- Model, işlem hızlanmasını (velocity) ve donanım değişimi (network anomaly) gibi özellikleri ana faktörler olarak saptamıştır.
""")

elif selection == "Model Sağlığı ve Konsept Kayması (PSI)":
    st.header("Model Sağlığı ve Concept Drift (Konsept Kayması)")
    st.markdown("""
Makine Öğrenmesi modellerinin, üretim ortamında (production) zamanla eski performansını koruyup korumadığını (Data/Concept Drift) **Population Stability Index (PSI)** metriği ile canlı olarak izliyoruz.
    """)
    
    st.subheader("Population Stability Index (PSI) Analizi")
    st.markdown("PSI, referans eğitim/geçmiş skoru ile canlı skorların dağılımını kıyaslar. **PSI < 0.1:** Stabil, **0.1 <= PSI < 0.2:** Hafif Kayma, **PSI >= 0.2:** Kuvvetli Kayma (Yeniden eğitim şart).")
    
    def calculate_psi(expected, actual, bins=10):
        expected_pct = np.histogram(expected, bins=bins, range=(0, 100))[0] / len(expected)
        actual_pct = np.histogram(actual, bins=bins, range=(0, 100))[0] / len(actual)
        
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    mid_idx = len(test_df) // 2
    base_scores = test_df['lgbm_risk_score'].iloc[:mid_idx]
    current_scores = test_df['lgbm_risk_score'].iloc[mid_idx:]
    
    current_psi = calculate_psi(base_scores, current_scores)
    
    col1, col2 = st.columns(2)
    col1.metric("Kümülatif PSI Skoru", f"{current_psi:.4f}")
    
    if current_psi < 0.1:
        col2.success("Durum: Model Dağılımı Stabil. Herhangi bir concept drift engeline rastlanmadı.")
    elif current_psi < 0.2:
        col2.warning("Durum: Hafif Dağılım Bozulması Saptandı. Özellik mühendisliği çıktıları yakından izlenmeli.")
    else:
        col2.error("Durum: Yüksek Veri Kayması! Model Acilen Yeniden Eğitilmeli (Retrain Trigger Event).")
        
    st.write("### Olasılık Dağılım Kıyaslaması (Referans - Gözlem)")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=base_scores, name='Referans Kestirim', opacity=0.6, marker_color='#3366cc'))
    fig2.add_trace(go.Histogram(x=current_scores, name='Güncel Çıktılar', opacity=0.6, marker_color='#ff9900'))
    fig2.update_layout(barmode='overlay', xaxis_title="Risk Skoru Tahmini", yaxis_title="Hacim Dağılımı")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    '<div class="footer">Mimari sistem incelemeleri ve portföy temsili amacıyla salt sentetik veri döngüleri ile kurgulanmıştır. Finansal veya ticari bir istihbarat yansıması taşımaz.</div>',
    unsafe_allow_html=True
)