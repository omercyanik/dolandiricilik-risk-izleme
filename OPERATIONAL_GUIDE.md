# Operational Guide: Thresholds & False Negatives

Bu makine öğrenmesi destekli karar platformu, sadece teorik doğrulukla ilgilenmek yerine operasyonel kârlılığı (ROI) hedefler. 

## 1. Neden ML Her Dolandırıcıyı Yakalayamaz? (False Negatives)

Mülakat ve teknik takım incelemelerinde karşınıza çıkacak en büyük konsept **False Negative** (Yalancı Negatif) kavramıdır. Sistemde `%4` risk skoru aldığı halde `is_fraud=1` (dolandırıcı) çıkan işlemler (transactions) görebilirsiniz. 

Bu bir "model zafiyeti" veya "kod hatası" **değildir**.
* **Mükemmel Taklit:** Dolandırıcı; asıl müşterinin işlem geçmişi (tutar, zaman, sıklık) profiline birebir uyacak şekilde ufak ödemeler çekerse, model matematiksel olarak "Bu doğal müşteri davranışıdır" diyecektir.
* **Trade-off İkilemi:** Sırf bu ufak sızıntıları yakalamak uğruna threshold'u (engelleme sınırını) `%75` yerine `%4`'e çekersek, sistemdeki temiz/dürüst müşterilerin **%90'ını bloğa (False Positive)** düşürmüş oluruz.

_Bankacılıkta müşteri memnuniyetsizliği (churn) zararı, küçük miktarlı fraudlardan bile daha büyüktür. Hedef **Sıfır Dolandırıcılık** değil, **Optimal Kârlılıktır.**_

---

## 2. Karar Eşiği (Threshold) Stratejileri

Streamlit arayüzündeki **Threshold Rule Optimization** ve **ROI & Cost Simulator** sekmelerinde sektörde uygulanan 3 farklı senaryoyu test edebilirsiniz:

### A) Balanced (Standart Operasyon)
- **MFA Threshold:** ~35.0
- **Block Threshold:** ~75.0
- **Use Case:** Normal bir iş günü. Gri alandaki şüphelilere sadece SMS OTP / 3D Secure (Step-Up) gidilir. Bariz fraudlar bloklanır. Dürüst kitlenin %95'i sürtünme (friction) yaşamaz.

### B) Aggressive (Saldırı / BIN Attack Durumu)
- **MFA Threshold:** ~20.0
- **Block Threshold:** ~60.0
- **Use Case:** Sisteme yöneltilen sistematik bir bot ya da BIN carding saldırısı altında, UX (kullanıcı deneyimi) anlık olarak feda edilip defans sınırları daraltılır. Operasyon (Review) maliyeti artar ama sızıntı kökten önlenir.

### C) Frictionless (VIP / Black Friday Kampanyaları)
- **MFA Threshold:** ~50.0
- **Block Threshold:** ~90.0
- **Use Case:** "Sepet Terk Etme (Cart Abandonment)" oranının bankaya veya işyerine vereceği zararın en riskli olduğu dönemler. Modele sadece "Çok eminsen müşteriyi engelle" yetkisi verildiği toleranslı stratejidir.

---
_Projenin özü: Makine öğrenmesini çalıştırmak teknik bir iştir, zor olan bu tahmin çıktılarını bankanın iş birimlerinin (business) kabul edeceği bilanço ve ROI dengelerine uydurabilmektir._
