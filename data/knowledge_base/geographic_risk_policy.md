# Coğrafi Risk Politikası

Faturalandırma bölge/ülke kodu (addr2), sistemde ölçülen en güçlü risk sinyalidir. Verinin
%99,2'si tek bir yerli bölge koduna aittir; bu koddan farklı (yabancı) işlemlerde fraud oranı
yerli işlemlere göre yaklaşık 4,26 kat daha yüksektir. Bölge kodu eksik olan işlemlerde ise fraud
oranı yabancı işlemlerden bile daha yüksektir (yaklaşık 4,91 kat) — bu yüzden politika, yabancı ve
eksik durumları AYRI iki risk katmanı olarak ele alır, tek bir "yabancı" kategorisinde birleştirmez.

İki yöntem uygulanır: sabit bir politika çarpanı (sadece kesin bilinen yabancı işlemler için, 2,0
kat — eksik veriye dokunulmaz, çünkü etiketsiz bir politika "bilinmiyor"u "riskli" diye
varsayamaz) ve fraud-oranı-kalibreli bir çarpan (yerli=1,0, yabancı≈4,26, eksik≈4,91).

Diğer coğrafi adaylar (fiziksel mesafe, e-posta alan adı uzantısı) test edilmiş ama güvenilir
bulunmadığı için kullanılmamıştır.
