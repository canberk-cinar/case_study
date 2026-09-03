# Velocity (Hızlı Tekrar İşlem) Politikası

Aynı kartın çok kısa süre içinde tekrar işlem yapması, kart-test etme (card testing) veya otomatik
fraud saldırılarının klasik bir işaretidir. Sistem, bir kartın önceki işleminden bu yana geçen
süreyi izler.

60 saniyeden kısa sürede yapılan tekrar işlemler HIGH önem derecesiyle işaretlenir ve incelemeye
(REVIEW) yönlendirilir. Daha güçlü bir örüntü ise "adres değişimi + hızlı tekrar" birleşimidir:
kart için daha önce hiç kullanılmamış yeni bir faturalandırma adresiyle, 300 saniye içinde tekrar
işlem yapılması — bu, çalıntı kart numaralarının birden fazla "drop" adresine karşı test edildiği
fraud çetesi örüntüsüne benzer ve CRITICAL önem derecesiyle işlemi durdurma (BLOCK) eylemini
tetikler.

Kartın hiç önceki işlemi yoksa (ilk işlem), velocity kontrolü anlamsızdır ve uygulanmaz.
