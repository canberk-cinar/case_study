# Güvenilir Kart (Trusted Entity) Politikası — Beklenmedik Bir Bulgu

Naif beklenti, uzun işlem geçmişine sahip bir kartın daha güvenilir olduğu, dolayısıyla skorunun
hafifletilebileceği yönündeydi. Ancak veri analizi bunun TAM TERSİNİ göstermiştir: yeni/az
geçmişli kartlarda fraud oranı düşükken (%2,4-2,6), 101-1000 arası işlem geçmişine sahip
kartlarda fraud oranı en yüksek seviyeye çıkmaktadır (%3,91).

Muhtemel açıklama, kart tanımlayıcısının bu ölçekte tek bir gerçek müşteriyi değil, paylaşılan bir
"bucket" değerini temsil etmesidir. Bu nedenle "uzun geçmiş = güven" varsayımına dayalı naif bir
düzeltme, test edildiğinde sistemin performansını GERÇEKTEN KÖTÜLEŞTİRMİŞTİR — üst risk kovasının
sıralamasını bozmuştur. Fraud-oranına göre kalibre edilmiş ters yönlü bir düzeltme ise doğru
sonucu vermiştir.

Bu politika, "mantıklı görünen" bir iş kuralının veriyle test edilmeden uygulanmaması gerektiğinin
somut bir örneğidir.
