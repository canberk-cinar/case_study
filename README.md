# Fraud / Anomaly Detection Platform

LOGO Yazılım case study. IEEE-CIS Kaggle veri seti üzerinde tamamen etiketsiz (unsupervised) bir
fraud/anomali tespit platformu. Case 1'den Case 10'a kadar katman katman inşa edildi: veri
hazırlığı → feature engineering → çok katmanlı anomali skorlama → skor birleştirme → iş-bağlamı
düzeltmeleri → configurable rule engine → RAG pipeline → multi-agent orkestrasyon → FastAPI
servisleştirme.

**Temel disiplin:** `isFraud` (gerçek fraud etiketi) hiçbir tasarım kararını yönlendirmedi, sadece
sonradan, betimleyici doğrulama için kullanıldı (Case 6'daki bilinçli ve açıkça işaretlenmiş
kalibrasyon istisnaları hariç).

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env içindeki LLM_API_KEY ve LLM_MODEL alanlarını doldurun (bkz. "LLM Konfigürasyonu")
```

**Bu repo `data/processed/merged_transactions.parquet` ve `case_study.db`'yi doğrudan içeriyor**;
normalde türetilmiş/yeniden üretilebilir dosyalar git'e eklenmez, ama indirme/kurulum zahmeti
olmadan doğrudan çalışabilsin diye bilinçli olarak dahil edildi. Yani API ve Case 2-10'un hiçbiri
için ham veri indirmenize gerek yok; sadece `python -m src.pipelines.build_scoring_artifact`
komutunu çalıştırmanız yeterli (aşağıda).

Ham Kaggle CSV'leri (`data/raw/*.csv`) yine de `.gitignore`'da: IEEE-CIS yarışma verisi yeniden
dağıtılabilir değil, ve `train_transaction.csv` (652MB) zaten GitHub'ın 100MB dosya limitini
aşıyor. Bu dosyalara SADECE Case 1 notebook'unu (`merged_transactions.parquet`'i üreten adım)
sıfırdan yeniden çalıştırmak isterseniz ihtiyacınız var, aksi halde atlayabilirsiniz:

1. [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
   yarışmasına katılın (ücretsiz, Kaggle hesabı gerektirir) ve `train_transaction.csv` ile
   `train_identity.csv`'yi indirip `data/raw/` altına koyun (`kaggle competitions download -c
   ieee-fraud-detection` komutu, kurulu bir Kaggle API anahtarıyla, aynı işi yapar).
2. Case 1 notebook'unu çalıştırın (`jupyter nbconvert --to notebook --execute --inplace
   notebooks/case_01_analysis.ipynb`): bu, ham CSV'leri birleştirip `data/processed/
   merged_transactions.parquet`'i üretir (repodaki hazır kopyanın üzerine yazar).

**Skorlama artifact'i de repodaki `case_study.db`'de zaten hazır** (590.540 satır); API'yi
başlatmadan önce ekstra bir adım gerekmez. Skorlama/feature mantığında bir değişiklik yapıp
yeniden hesaplamak isterseniz:

```bash
python -m src.pipelines.build_scoring_artifact
```

Bu, tüm veri seti için feature'ları ve anomali skorlarını hesaplayıp `case_study.db` içindeki
`scored_transactions` tablosuna yazar (~40 sn, mevcut tabloyu değiştirir). API bu tablodan okur;
olmadan (repodaki `.db` silinip yeniden oluşturulursa) endpoint'ler 404 verir (net bir hata
mesajıyla, sessiz çökme yok).

## LLM Konfigürasyonu (Case 8/9)

Case 8 (RAG) ve Case 9 (Agentic AI), brief'in local-LLM (Ollama) gereksinimiyle tasarlandı; bu
makinenin RAM kısıtı nedeniyle **OpenRouter'ın ücretsiz katmanı** varsayılan sağlayıcı olarak
kullanılıyor, hem chat hem embedding tarafı için. Mimari baştan sağlayıcıdan bağımsız (Strategy
pattern + `dependency_injector` Selector) yazıldığı için OpenRouter ↔ yerel Ollama arasında geçiş,
kod dokunmadan **tek bir `.env` değişkeni**:

```bash
LLM_PROVIDER=openrouter   # veya "ollama"
```

`LLM_PROVIDER`, `RAGContainer`'ın hem embedding hem chat Selector'ını aynı anda değiştiriyor;
`openrouter` seçiliyken aşağıdaki değerler kullanılır:

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<openrouter-api-anahtarınız>
LLM_MODEL=<seçtiğiniz ücretsiz chat modeli, örn. "nvidia/nemotron-3.5-lightning:free">
EMBEDDING_MODEL=<seçtiğiniz ücretsiz embedding modeli, örn. "nvidia/nemotron-3-embed-1b:free">
```

`LLM_PROVIDER=ollama` yapıldığında yukarıdakiler devre dışı kalır, Ollama'nın yerel modelleri
(`smollm2:360m` chat, `all-minilm` embedding) `localhost:11434` üzerinden kullanılır. Case 8
notebook'unun 8. bölümü bu geçişi fiilen doğruluyor.

**LLM/embedding anahtarı ayarlanmadan da** API çalışır ama farklı şekillerde düşer: `answer()` (LLM
tarafı) `httpx.HTTPError`'ı yakalayıp `answer: null` + açıklayıcı bir `note` ile zarif şekilde
döner (retrieval ve kural motoru sonuçları yine tam gelir); `ingest()` (embedding tarafı) ise
yakalayamaz: retrieval embedding olmadan mümkün olmadığı için anlaşılır bir `RuntimeError`
fırlatır (ham bir `httpx` hatası yerine).

## Çalıştırma

```bash
uvicorn main:app --reload
```

`http://localhost:8000/docs`: otomatik oluşturulan Swagger arayüzü. `GET /healthz` sağlık
kontrolü.

## Endpoint'ler

Beşi de bir `transaction_id` ile (`scored_transactions` tablosundaki, dolayısıyla veri setindeki
mevcut bir satır) çalışır; bu bir "demo/analiz" API'si, canlı/yeni bir işlemi skorlayan bir
prodüksiyon API'si değil. Skorlar/feature'lar önceden hesaplanmış (`build_scoring_artifact`), tek
satırlık lookup SQLite PK indeksiyle milisaniyeler sürüyor; ilk mimaride her istek tüm veri setini
(590k satır, 4 anomali katmanı) yeniden hesaplıyordu (`/score` ~27s, `/agent` ~80s); ölçüldü,
darboğaz teşhis edildi, düzeltildi.

| Metod | Path | Ne yapar | Tipik süre |
|---|---|---|---|
| GET | `/score/{transaction_id}` | Case 5'in ham anomali skoru (en hızlı, kural/LLM yok) | ~15ms |
| GET | `/rules/evaluate/{transaction_id}` | Case 7'nin yapılandırılmış kural değerlendirmesi (ateşlenen kurallar + verdict, mesajsız) | ~50ms |
| GET | `/explain/{transaction_id}` | Aynı değerlendirme, kural başına insan-okur açıklama ile | ~40ms |
| POST | `/rag/query` | Case 8'in RAG'ı: serbest soru (`{"question": "...", "transaction_id": null}`); `transaction_id` verilirse o işlemin kural verdict'ine göre açıklama üretir | ~12,5s (gerçek OpenRouter embedding + LLM ağ çağrıları) |
| GET | `/agent/{transaction_id}` | Case 9'un tam multi-agent orkestrasyonu (feature engineering → anomaly scoring → rule engine → koşullu RAG açıklaması) | ~9,5s (gerçek OpenRouter embedding + LLM ağ çağrıları) |

`/rag/query` ve `/agent`'ın süresi neredeyse tamamen OpenRouter'ın ücretsiz modellerinin ağ
gecikmesinden kaynaklanıyor (2 embedding isteği + 1 chat completion isteği); yerel bileşenler
(retrieval, prompt inşası, kural motoru) milisaniyeler içinde bitiyor. LLM/embedding anahtarı
olmadan aynı endpoint'ler ~0,4-0,5s'de döner (zarif düşüş, `answer: null`).

### Örnek istekler

```bash
curl http://localhost:8000/score/2988038

curl http://localhost:8000/explain/2988038

curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Yabancı ülkeden gece yapılan yüksek tutarlı işlem neden riskli?"}'

curl http://localhost:8000/agent/2988038
```

## Mimari notları

- **Design pattern'ler:** Strategy (embedding/LLM sağlayıcı takası, kural operatörleri), Composite
  (kural koşul ağaçları), Factory (YAML/JSON kural yükleyici), Chain of Responsibility (severity
  çözümleme), Adapter (Case 9 agent'ları Case 3-8'i sarmalıyor), Facade (`RAGPipeline`,
  `run_agentic_analysis`), Repository (`db_services/*.py`).
- **Dependency Injection:** `dependency_injector`: `RuleEngineContainer` (Case 7),
  `RAGContainer` (Case 8), ve bu ikisini FastAPI'nin `@inject`/`Provide[...]` mekanizmasıyla
  birleştiren üst-seviye `ApiContainer` (Case 10, `src/container.py`). Beş endpoint'in **tamamı**
  (agent dahil) container'dan besleniyor: `run_agentic_analysis(tx_id, rule_engine, rag_pipeline)`
  bu iki bağımlılığı parametre olarak alıyor (referans projenin `build_graph(db)` closure
  deseniyle aynı), kendi container'ını kurmuyor.
- **Skorlama artifact'i:** `src/pipelines/build_scoring_artifact.py` (typer CLI): tüm feature'ları
  ve anomali skorlarını bir kez hesaplayıp `scored_transactions` (SQLite, `TransactionID` PK)
  tablosuna yazar. `merged_transactions.parquet` (kolonsal, tam-tablo tarama, notebook'lar için
  doğru) ile `scored_transactions` (indeksli, tek-satır lookup, API için doğru) bilerek iki farklı
  depo: erişim deseni farklı olduğu için depo da farklı.
- **`schemas/` vs `domain/`:** `src/schemas/` (üst seviye) sadece Case 10'un Pydantic API
  sözleşmeleri: HTTP sınırını geçen tek katman. `rules/domain/models.py` (`Rule`, `Severity`,
  `Action`) ve `rag/domain/models.py` (`Chunk`, `RetrievedChunk`) Pydantic DEĞİL, düz
  dataclass/enum: Case 7/8'in API'den bağımsız, notebook'tan da çalışabilen iç veri modelleri.
  İsim çakışmasını önlemek için ayrı tutuldu.
- **Prompt'lar (`static/`):** `services/rag/static/system_prompt.json`: RAG'ın genel amaçlı
  sistem promptu (herhangi bir `RAGPipeline` çağıranı kullanır). `agents/policy_explanation/
  static/question_template.json`: SADECE bu agent'a özgü, bir rule verdict'ini doğal dile
  çeviren soru şablonu. Diğer 3 agent (feature_engineering, anomaly_scoring, rule_engine) LLM
  kullanmadığı için kendi `static/`'leri yok; sahte/kullanılmayan bir prompt eklenmedi.
- **Notebook'lar:** her case'in kendi `notebooks/case_0N_*.ipynb`'i, çalıştırılmış çıktılarla;
  altyapının nasıl inşa edildiğinin ve doğrulandığının tam kaydı.

## Proje yapısı

```
src/
├── config.py                    # tüm ayarlar (pydantic-settings, .env'den)
├── container.py                 # Case 10: üst-seviye DI container, 5 endpoint'in tamamı buradan besleniyor
├── database/
│   ├── db.py                    # SQLAlchemy engine/session/Base + run_migrations()
│   ├── models/                  # ORM modelleri
│   │   ├── artifact.py          # Artifact: Case 1/2 çıktı dosyalarının (parquet, rapor) kayıt defteri
│   │   ├── rag.py               # Document, DocumentChunk: RAG knowledge base
│   │   └── scoring.py           # ScoredTransaction: API'nin okuma modeli (bkz. pipelines/)
│   └── db_services/             # her model için Repository-pattern CRUD (artifact.py, rag.py, scoring.py)
├── pipelines/
│   └── build_scoring_artifact.py  # feature+skorları hesaplayıp scored_transactions'a yazan typer CLI
├── services/
│   ├── data_processing.py        # Case 1: ham CSV'leri birleştirip merged_transactions.parquet'i üretir
│   ├── analyzers/                # Case 1/2: veri kalitesi, dağılım, cardinality analizleri
│   ├── features/                 # Case 3: causal (sızıntısız) feature engineering
│   ├── anomaly/                  # Case 4/5/6: 4 katmanlı anomali skorlama + birleştirme + context adjustment
│   ├── evaluation/
│   │   └── roc.py                 # ROC-AUC: Case 4/5/6/7'nin ortak betimleyici doğrulama aracı
│   ├── rules/                    # Case 7: configurable if-then rule engine
│   │   ├── domain/models.py      # Rule, Severity, Action (dataclass/enum, Pydantic değil)
│   │   └── container.py          # RuleEngineContainer (DI)
│   └── rag/                      # Case 8: RAG pipeline
│       ├── domain/models.py      # Chunk, RetrievedChunk (dataclass, Pydantic değil)
│       ├── static/system_prompt.json  # genel amaçlı RAG sistem promptu
│       └── container.py          # RAGContainer (DI)
├── agents/                       # Case 9: LangGraph multi-agent orkestrasyon
│   ├── schemas/state.py          # AgentState: agent'lar arası paylaşılan state
│   ├── graph.py                  # StateGraph + koşullu kenarlar
│   ├── supervisor/agent.py       # giriş noktası (Facade)
│   ├── feature_engineering/agent.py  # deterministik, Case 3'ü sarmalıyor (Adapter)
│   ├── anomaly_scoring/agent.py      # deterministik, Case 5'i sarmalıyor (Adapter)
│   ├── rule_engine/agent.py          # deterministik, Case 7'yi sarmalıyor (Adapter)
│   └── policy_explanation/       # tek LLM-tabanlı agent
│       └── static/question_template.json  # bu agent'a özgü soru şablonu
└── routes/, schemas/              # Case 10: FastAPI endpoint'leri (5+5 dosya, endpoint başına bir çift;
                                    # schemas/ = Pydantic API sözleşmeleri)
main.py                            # FastAPI giriş noktası (lifespan, healthz, router'lar), repo kökünde
notebooks/                         # her case'in çalıştırılmış defteri (case_01 → case_09; Case 10'un
                                    # kendi notebook'u yok, ürünleşmiş hâli main.py/routes/schemas/container.py)
data/
├── raw/                          # train_transaction.csv, train_identity.csv (ham Kaggle verisi)
├── processed/                    # merged_transactions.parquet (Case 1'in ürettiği, tüm case'lerin okuduğu)
└── knowledge_base/               # RAG için policy dökümanları (İngilizce, kural motoruyla aynı dil)
```

## Case'ler

**Case 1: Veri Birleştirme ve Keşifsel Analiz.** `train_transaction.csv` ve `train_identity.csv`
birleştirilerek 590.540 satır / 435 kolonluk `merged_transactions.parquet` üretildi (identity
kapsaması %24,42, satır çoğalması yok). İki katmanlı otomatik kolon tipleme (istatistik + domain
hint), eksik veri deseni analizi (208/434 kolon %70+ eksik, 27 desen grubu, 6'sı join-kaynaklı
yapısal), ve genel bir veri kalitesi skoru (98,07/100) bu notebook'ta üretildi. Sonraki tüm case'ler
bu parquet dosyasını temel alıyor.

**Case 2: Derinlemesine Veri Analizi.** Case 1'in `analyzers/` modülleri yeniden kullanılarak
(numeric/categorical/datetime ayrımı: 307 kategorik, 124 sayısal, 1 datetime) kolon ilişkileri,
nadir kategorik kombinasyonlar ve entity davranış pattern'leri incelendi. `card1` bazlı entity'ler
için üç davranış bayrağı (bölge çeşitliliği, tutar değişkenliği, işlem hızı) tanımlandı; entity'lerin
~%2,8'i üçünü birden taşıyor.

**Case 3: Feature Engineering.** Betimleyici Case 1/2'den ayrı, üretici bir katman olarak
`services/features/` paketi açıldı: temporal (döngüsel sin/cos zaman sinyalleri, 8 kolon), entity
(`card1` bazlı nedensel `_so_far` tutar geçmişi, 6 kolon; kasıtlı sızıntılı bir referans kolonuyla
sızıntının büyüklüğü de ölçüldü), relational (`card1`×`addr1`/`DeviceInfo` ilişkisi, 5 kolon) ve
context (bağlama göre sapma) aileleri. Bağlamsal sapma sinyalinin gücü somut: `|z|>=1` gruplarında
fraud oranı `|z|<1` grubuna göre 2,5-3 kat daha yüksek.

**Case 4: Çok Katmanlı Anomali Tespiti.** Tek bir modele bağlı kalmadan dört bağımsız bakış açısı
kuruldu: column (MAD z-score), multivariate (Mahalanobis + Isolation Forest), entity (kendi
geçmişine göre sapma) ve temporal. Katmanların birbirine baskın gelmediği sayısal olarak doğrulandı:
590.540 işlemin sadece 4'ü beş skorun hepsinde top-%1'de, 13.314 işlem sadece tek bir katmanda
yakalanıyor: tek katmana güvenmenin gerçek bir bilgi kaybı olduğunun kanıtı.

**Case 5: Birleşik Risk Skoru.** Case 4'ün dört bağımsız katmanı, hangisinin ne kadar yeni bilgi
taşıdığına (korelasyon matrisi) dayanan bir ağırlıklandırmayla tek bir sürekli `final_raw_anomaly_score`'a
birleştirildi. Süreç boyunca `isFraud` hiçbir ağırlığı belirlemedi, sadece skorla fraud oranının
birlikte arttığını göstermek için betimleyici doğrulamada kullanıldı.

**Case 6: Business Context.** Case 5'in istatistiksel skoru, iş bağlamına göre yeniden
ağırlıklandırılarak false-positive oranını azaltan bir politika katmanı eklendi: business hours,
hafta sonu, trusted entity ve geographic risk: dördü de hem etiketsiz hem etiket-kalibreli
yöntemle karşılaştırıldı. Sonuç öğretici: dört kuraldan ikisi (hafta sonu, coğrafi risk) gerçekten
iyileşme sağladı, ikisi (business hours, trusted entity) tam tersi yönde çalıştı: "mantıklı
görünen" bir iş kuralının etkisini ölçmeden üretime almamak gerektiğinin somut kanıtı. En güçlü tek
sinyal geographic risk oldu (yabancı işlemlerde 4,26-4,91x fraud oranı).

**Case 7: Configurable Rule Engine.** `services/rules/` altında, kodu değiştirmeden YAML/JSON'dan
kural ekleyip çıkarabilen, açıklanabilir bir karar motoru kuruldu: Composite pattern ile iç içe
koşul ağaçları, öncelik (priority) ve iki bağımsız çakışma-çözümleme stratejisi (severity tabanlı
Chain of Responsibility ve priority tabanlı), ikisi karşılaştırıldı (%99,58 uyum). Motor, Case 5'in
anomali skoruyla birlikte tek bir final verdict (severity + action) üretiyor.

**Case 8: RAG Pipeline.** Case 7'nin kural kararlarını, İngilizce policy dökümanlarından oluşan bir
knowledge base'e dayanarak doğal dilde açıklayan bir retrieval-augmented generation pipeline'ı
kuruldu: chunking → embedding → SQLite'a persist → vector search → prompt injection → LLM üretimi.
Sağlayıcı seçimi (`embedding_provider`/`llm_provider`) baştan Strategy pattern + `dependency_injector`
Selector ile soyutlandı; varsayılan olarak OpenRouter'ın ücretsiz katmanı kullanılıyor, tek bir
`.env` değişkeniyle (`LLM_PROVIDER`) yerel Ollama'ya geçilebiliyor. Retrieval kalitesi dürüstçe
belgelendi: dil hizalaması (KB'nin İngilizceye çevrilmesi) ve gerçek semantik embedding'e geçiş
sonrası bile, birden fazla kural adını art arda sıralayan otomatik sorgularda en spesifik terim
seyrelebiliyor. Bu, mevcut kapsamın dışında bırakılan bir iyileştirme notu olarak işaretlendi.

**Case 9: Agentic AI Yapısı.** Case 3-8'in bağımsız bileşenleri, LangGraph ile kurulan bir
multi-agent orkestrasyon katmanına bağlandı: `feature_engineering` → `anomaly_scoring` →
`rule_engine` → (koşullu) → `policy_explanation`. Üç agent tamamen deterministik (Case 3/4/5/7'nin
fonksiyonlarını sarmalıyor, Adapter pattern); sadece `policy_explanation` gerçek bir LLM node'u.
Agent'lar birbirine doğrudan mesaj göndermiyor, paylaşılan `AgentState` üzerinden konuşuyor; test
sırasında bulunan gerçek bir tasarım hatası (`rule_engine`'in yanlışlıkla anomali skoruna bağımlı
kılınması) burada düzeltildi.

**Case 10: FastAPI Servisleştirme.** Case 1-9'un çıktıları 5 endpoint'lik bir API'ye
dönüştürüldü (`/score`, `/rules/evaluate`, `/explain`, `/rag/query`, `/agent`), `dependency_injector`
ile uçtan uca DI'lanmış (`ApiContainer`). İlk mimari her isteği veri setinin tamamı üzerinden
yeniden hesaplıyordu (`/score` ~27sn, `/agent` ~80sn); darboğaz ölçülüp teşhis edildi (yeniden
hesaplama, I/O değil) ve önceden hesaplanmış bir skorlama artifact'i (`scored_transactions`,
SQLite, PK indeksli) ile çözüldü, aynı endpoint'ler artık milisaniyeler içinde dönüyor. Bu case'in
kendi notebook'u yok; ürünleşmiş hâli `main.py` + `src/routes`/`src/schemas`/`src/container.py`.
