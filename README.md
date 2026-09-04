# Fraud / Anomaly Detection Platform

LOGO Yazılım case study — IEEE-CIS Kaggle veri seti üzerinde tamamen etiketsiz (unsupervised) bir
fraud/anomali tespit platformu. Case 1'den Case 10'a kadar katman katman inşa edildi: veri
hazırlığı → feature engineering → çok katmanlı anomali skorlama → skor birleştirme → iş-bağlamı
düzeltmeleri → configurable rule engine → RAG pipeline → multi-agent orkestrasyon → FastAPI
servisleştirme.

**Temel disiplin:** `isFraud` (gerçek fraud etiketi) hiçbir tasarım kararını yönlendirmedi — sadece
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

Ham veri (`data/raw/train_transaction.csv`, `data/raw/train_identity.csv`) yerinde olmalı — Case 1
notebook'u (`notebooks/case_01_analysis.ipynb`) bunları birleştirip `data/processed/
merged_transactions.parquet`'i üretiyor; API ve sonraki tüm case'ler bu dosyayı okuyor. Önce Case 1
notebook'unu (`jupyter nbconvert --to notebook --execute --inplace notebooks/case_01_analysis.ipynb`)
çalıştırmadan API'nin çoğu endpoint'i çalışmaz.

## LLM Konfigürasyonu (Case 8/9)

Case 8 (RAG) ve Case 9 (Agentic AI), brief'in local-LLM (Ollama) gereksinimiyle başladı, ama bu
makinenin RAM kısıtı nedeniyle case study ekibiyle görüşülüp **OpenRouter'ın ücretsiz bir modeli**
kullanılması onaylandı. Mimari baştan sağlayıcıdan bağımsız (Strategy pattern) yazıldığı için bu
sadece bir `.env` ayarı:

```bash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<openrouter-api-anahtarınız>
LLM_MODEL=<seçtiğiniz ücretsiz model, örn. "meta-llama/llama-3.2-3b-instruct:free">
```

Ollama yerel olarak kurulursa, aynı config'i değiştirmek yeterli (kod değişmez):

```bash
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=smollm2:360m
```

Embedding üretimi (Case 8) `LLM_*` ayarlarından bağımsız — OpenRouter'ın güvenilir bir embedding
endpoint'i olmadığı için varsayılan olarak tamamen yerel, ücretsiz bir yöntem (TF-IDF,
scikit-learn) kullanılıyor; `RAGContainer`'ın `embedding_provider` config'i `ollama`'ya çevrilerek
Ollama'nın embedding modeline de geçilebilir.

**LLM anahtarı ayarlanmadan da** API tamamen çalışır — `/rag/query` ve `/agent` endpoint'leri bu
durumda `answer: null` + açıklayıcı bir `note` ile zarif şekilde döner (retrieval ve kural motoru
sonuçları yine tam gelir), çökmez.

## Çalıştırma

```bash
uvicorn main:app --reload
```

`http://localhost:8000/docs` — otomatik oluşturulan Swagger arayüzü. `GET /healthz` sağlık
kontrolü.

## Endpoint'ler

Beşi de bir `transaction_id` ile (`data/processed/merged_transactions.parquet`'teki mevcut bir
satır) çalışır — bu bir "demo/analiz" API'si, canlı/yeni bir işlemi skorlayan bir prodüksiyon API'si
değil (istatistikler/embedding'ler her istekte veri setinden taze hesaplanıyor, kalıcı bir model
artifact'i hiç saklanmadı).

| Metod | Path | Ne yapar |
|---|---|---|
| GET | `/score/{transaction_id}` | Case 5'in ham anomali skoru — en hızlı, kural/LLM yok |
| GET | `/rules/evaluate/{transaction_id}` | Case 7'nin yapılandırılmış kural değerlendirmesi (ateşlenen kurallar + verdict, mesajsız) |
| GET | `/explain/{transaction_id}` | Aynı değerlendirme, kural başına insan-okur açıklama ile |
| POST | `/rag/query` | Case 8'in RAG'ı — serbest soru (`{"question": "...", "transaction_id": null}`); `transaction_id` verilirse o işlemin kural verdict'ine göre açıklama üretir |
| GET | `/agent/{transaction_id}` | Case 9'un tam multi-agent orkestrasyonu (feature engineering → anomaly scoring → rule engine → koşullu RAG açıklaması) |

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
- **Dependency Injection:** `dependency_injector` — `RuleEngineContainer` (Case 7),
  `RAGContainer` (Case 8), ve bu ikisini FastAPI'nin `@inject`/`Provide[...]` mekanizmasıyla
  birleştiren üst-seviye `ApiContainer` (Case 10, `src/container.py`).
- **Notebook'lar:** her case'in kendi `notebooks/case_0N_*.ipynb`'i, çalıştırılmış çıktılarla —
  altyapının nasıl inşa edildiğinin ve doğrulandığının tam kaydı.

## Proje yapısı

```
src/
  config.py              # tüm ayarlar (pydantic-settings, .env'den)
  database/               # SQLAlchemy modelleri + Repository-pattern CRUD (artifact, RAG knowledge base)
  services/
    analyzers/              # Case 1/2 — veri kalitesi, dağılım, cardinality analizleri
    features/                 # Case 3 — causal (sızıntısız) feature engineering
    anomaly/                    # Case 4/5/6 — 4 katmanlı anomali skorlama + birleştirme + context adjustment
    rules/                        # Case 7 — configurable if-then rule engine
    rag/                            # Case 8 — RAG pipeline
    agents/                           # Case 9 — LangGraph multi-agent orkestrasyon
  routes/, schemas/                     # Case 10 — FastAPI endpoint'leri
  container.py                            # Case 10 — üst-seviye DI container
main.py                                    # FastAPI giriş noktası
notebooks/                                   # her case'in çalıştırılmış defteri
data/knowledge_base/                           # RAG için örnek policy dökümanları
```
