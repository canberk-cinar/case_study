# Yeni Kart, Yüksek Değerli İlk İşlem Politikası

Bir kartın gözlemlenen ilk işleminin (önceki işlem geçmişi sıfır) yüksek tutarlı olması, kartın
davranışını doğrulayacak hiçbir geçmiş veri olmadan büyük bir riske girildiği anlamına gelir. Bu
durum CRITICAL önem derecesiyle işaretlenir ve işlemi durdurma (BLOCK) eylemini tetikler.

Ayrıca, hafta sonu ve düşük-hacimli saat diliminde (04:00-09:00) gerçekleşen yüksek tutarlı
işlemler ayrı bir MEDIUM seviyeli işaretleme (FLAG) alır — bu, tek başına zayıf olan iki bağlamsal
sinyalin (hafta sonu + düşük hacim) birleşiminin, tek başına her birinden daha güçlü bir risk
göstergesi olduğu gözlemine dayanır.

Yüksek tutar eşiği, verinin kendi yüzdelik dilimlerinden (yaklaşık üst %5) türetilmiştir; gerçek
fraud etiketine bakılarak seçilmemiştir.
