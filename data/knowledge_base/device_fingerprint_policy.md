# Cihaz Parmak İzi Politikası

Cihaz bilgisi (DeviceInfo) eksik olan ya da yalnızca genel bir işletim sistemi etiketi (örneğin
"Windows", "iOS Device", "MacOS") taşıyan — belirli bir cihaz modeli bilgisi içermeyen — işlemler,
cihaz parmak izinin doğrulanamadığı işlemler olarak değerlendirilir.

Bu durum tek başına düşük bir risk taşır, ancak yüksek tutarlı bir işlemle birleştiğinde MEDIUM
önem derecesiyle işaretlenip incelemeye (FLAG) yönlendirilir. Ayrıca, aynı kart için hem yeni bir
cihazın hem de yeni bir faturalandırma adresinin AYNI ANDA görülmesi — klasik bir hesap ele
geçirme (account takeover) işaretidir — HIGH önem derecesiyle işaretlenir.

Bu politika yalnızca cihaz bilgisinin genel/eksik olmasına dayanır; belirli bir cihaz markası veya
modeli hedef alınmaz.
