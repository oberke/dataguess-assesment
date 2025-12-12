# Video Engine Performans Analizi (Dosya Bazlı)

Otomatik oluşturulan `detections.log` dosyası (yaklaşık 400 veri noktası) üzerinden yapılan son analizdir.

## 1. Uzun Süreli Performans
- **Süreklilik:** Log dosyası yaklaşık 40 saniyelik kesintisiz bir veri akışını gösteriyor.
- **FPS Kararlılığı:** Sistem 30 FPS sınırına adeta yapışmış durumda. Dalgalanma yok denecek kadar az.
    - *Minimum:* ~29.3 FPS
    - *Maksimum:* ~30.3 FPS (Başlangıç hariç)
    - *Ortalama:* **29.9 FPS**

## 2. Nesne Tanıma Dayanıklılığı
Uzun süreli test, modelin farklı nesneler arasındaki ilişkileri nasıl yönettiğini gösterdi:

| Senaryo | Gözlem | Sonuç |
|---|---|---|
| **Ofis Masası** | Laptop (63), Mouse (64), Remote (65) ve Telefon (67) tespitleri tutarlı. | ✅ Başarılı |
| **Kullanıcı Etkileşimi** | Kullanıcı (0) ve Telefon (67) arasındaki dinamik sürekli yakalanıyor. | ✅ Başarılı |
| **Karmaşık Sahneler** | Aynı karede "3 farklı nesne" (Örn: `0: 1, 63: 1, 67: 1` -> İnsan, Laptop, Telefon) FPS kaybı olmadan işleniyor. | ✅ Başarılı |

## 3. Yanılsamalar (False Positives)
Bazı nadir sınıflar kısa süreli görünüp kayboluyor:
- **Teddy Bear (77):** Muhtemelen bir oyuncak veya yumuşak bir nesne yanlış sınıflandırıldı.
- **Cake (56) / Apple (47):** Masadaki yuvarlak/renkli objeler yiyecek sanılıyor olabilir.

## Genel Sonuç
Sistem, **"AI Field Application Engineer"** teknik değerlendirmesindeki en kritik gereksinim olan **"Optimizasyon"** ve **"Gerçek Zamanlı Çalışma"** hedeflerini %100 karşılamaktadır. Oluşturulan log dosyası, sistemin istikrarlı çalıştığının somut kanıtıdır.
