# Fraud Risk & Threshold Optimization Platform

**Bankacılık ve E-Ticaret için Gerçek Zamanlı (Real-Time) ML Destekli Dolandırıcılık Önleme Sistemi**

Bu repo, finansal işlemleri makine öğrenmesi (LightGBM) ile skorlayıp operasyonel kararlara (Onay/MFA/Engelle) dönüştüren uçtan uca bir risk simülasyon iskeletidir. Klasik ML projelerinden farklı olarak; **işletme karlılığı (ROI), threshold optimizasyonu, concept drift takibi ve model açıklanabilirliğine (SHAP)** odaklanır.

> **Not:** Veriler tamamen sentetiktir, herhangi bir kurumsal/gerçek veri içermez.

---

## 🚀 Özellikler (Features)

- **Gelişmiş Özellik Mühendisliği (Feature Engineering):** Velocity (hız/ivme), behavioral z-score, ve graph network tabanlı (ortak cihaz kullanımı) t-1 gecikmeli özellikler.
- **Sınıf Dengesizliği (Class Imbalance) Yönetimi:** Olasılık kalibrasyonuna gerek bırakmayan, ham risk skorları üzerinden `scale_pos_weight` kullanılarak kurgulanmış LightGBM altyapısı.
- **Canlı Hibrit Kural Motoru (Rule + ML Engine):** İşlemleri hem ML risk skoruna hem de iş kurallarına (örn: "Sınır ötesi > $5000") göre değerlendirir.
- **Mali Etki (ROI) Simülatörü:** False Positive (Yanlış Pozitif) müşteri kayıp maliyeti ile Fraud kayıp maliyetini heaplayıp karar vericiye optimal threshold stratejisini sunar.
- **XAI & Monitoring (SHAP & PSI):** Regülasyonlara uyumluluk (BDDK vs.) için kararları SHAP waterfall grafikleriyle detaylandırır. Population Stability Index (PSI) ile model sürüklenmesini (data drift) canlı izler.

---

## 🛠️ Kurulum & Çalıştırma

### Bağımlılıklar
- Python 3.9+ 
- (Mac kullanıcıları için LightGBM derleyicisi: `brew install libomp`)

### 1- Docker ile Hızlı Başlangıç (Önerilen)
Hiçbir lokal kütüphaneyle uğraşmadan direkt ayağa kaldırın:
```bash
docker-compose up --build
```

### 2- Manuel Kurulum (Local/Venv)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Pipeline'ı Çalıştırma
```bash
# 1. Sentetik transaction verisini üret
python -m src.data_generation

# 2. Velocity ve Graph feature'larını çıkar (Data Leakage korumalı)
python -m src.feature_engineering

# 3. Optuna hiperparametre optimizasyonu ile modeli eğit ve test setini skorla
python -m src.train
```

#### Streamlit Dashboard'u Başlatma
```bash
streamlit run app/streamlit_app.py
```

### 🧪 Unit Tests (Test Yazılımı)
Feature engineering (Özellik Mühendisliği) katmanındaki time-shift ve veri sızıntısı (leakage) kontrollerini test etmek için:
```bash
pytest tests/
```

> **📚 Önemli Dokümantasyon:** False Negative (Makine Öğrenmesi Gözden Kaçırmaları) savunmaları, Threshold/MFA stratejileri ve Mülakat Notları için 👉 [OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md) dosyasına göz atın.