# Mesai Saatleri Risk Politikası

Sistem, işlemleri mesai saatleri (hafta içi 09:00-18:00) ile mesai dışı saatler arasında farklı
değerlendirir. Veri analizi, mesai dışı saatlerde — özellikle 04:00-09:00 arası — işlem hacminin
çok düşük olduğunu ama fraud oranının bu saatlerde daha yüksek çıktığını göstermiştir.

Bu nedenle iki farklı düzeltme yaklaşımı uygulanır: hacim/güven tabanlı yaklaşım, az veriye
dayanan mesai dışı saatlerin istatistiksel olarak daha az güvenilir olduğunu varsayarak skoru
hafifletir; fraud-oranı-kalibreli yaklaşım ise, ölçülen gerçek fraud oranına göre mesai dışı
işlemlerin skorunu artırır. Her iki yöntem de ayrı ayrı hesaplanır ve karşılaştırılır — sistem tek
bir "doğru" yön varsaymaz, iki farklı prensibin sonuçlarını şeffaf şekilde sunar.

Mesai saatleri içindeki işlemler için herhangi bir düzeltme uygulanmaz; çarpan her zaman 1,0'dır.
