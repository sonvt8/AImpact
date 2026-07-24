# Báo cáo hoàn tất nâng cấp lõi RAG AImpact

Ngày nghiệm thu: 24/07/2026
Nhánh làm việc: `feature/rag-upgrade`
Commit nền/HEAD hiện tại: `e81656a feat: refine web UI and deterministic query handling`
Trạng thái phát hành: chưa commit, chưa push.

## 1. Kết luận nghiệm thu

- Bộ golden offline có 36 câu: 12 structured, 12 vector, 12 refuse.
- Structured executor đạt `12/12` exact-answer; câu AC/VHKT trả `6`, citation ô `Tong hop!E3`, không gọi LLM.
- E5 ONNX cuối đạt vector Acc@1 `0.667`, Recall@5 `1.000`, MRR `0.833`.
- Gate tại ngưỡng `0.84` đạt refusal precision `1.000`, refusal recall/out-reject `1.000`, false-answer `0.000`.
- API không còn lexical bypass, không quét index để đếm và không còn đường gán similarity giả `1.0`.
- Full backend suite: `106 passed, 1 warning in 13.50s`, `0 failed`.
- Frontend production build: thành công, Vite build `783ms`, 36 modules; `web/tsconfig.app.tsbuildinfo` đã restore về HEAD.
- Offline full startup lifecycle: đạt, `network_attempts=0`, `index_count=127`; structured/vector/refuse đều chạy đúng route.

## 2. Baseline → cleanup → E5 cuối

Nguồn số liệu: `data/rag-baseline-fasttext.txt`, `data/rag-after-cleanup-fasttext.txt`, `data/final-e5-96-eval.txt`. Các lượt đánh giá không gọi LLM để tách riêng chất lượng retrieval/gate.

| Chỉ số | Baseline FastText, 357 record | Sau cleanup FastText, 127 record | E5 ONNX cuối, 127 record |
|---|---:|---:|---:|
| Structured exact-answer | `0/12 supplied` | `0/12 supplied` | **`12/12`** |
| Vector Acc@1 / Locator R@1 | `0.083` | `0.167` | **`0.667`** |
| Vector Recall@5 / Locator R@5 | `0.250` | `0.417` | **`1.000`** |
| Vector MRR | `0.152` | `0.253` | **`0.833`** |
| Out-reject / refusal recall | `0.500` | `0.500` | **`1.000`** |
| Refusal precision | `0.857` | `0.857` | **`1.000`** |
| False-answer rate | `0.500` | `0.500` | **`0.000`** |
| Model load | `11,406.3 ms` | `10,949.4 ms` | **`1,139.9 ms`** |
| Parse | `230.0 ms` | `205.8 ms` | `208.5 ms` |
| Index | `443.8 ms` | `166.5 ms` | `1,578.6 ms` |
| Tổng load + parse + index | `12,080.1 ms` | `11,321.7 ms` | **`2,927.0 ms`** |
| Query embed avg / p95 | `0.078 / 0.170 ms` | `0.088 / 0.155 ms` | `5.147 / 6.926 ms` |
| Query index/rerank avg / p95 | `1.821 / 3.147 ms` | `1.697 / 2.284 ms` | `17.334 / 18.621 ms` |
| Gate avg / p95 | `0.003 / 0.006 ms` | `0.003 / 0.004 ms` | `0.005 / 0.007 ms` |
| Structured router avg / p95 | chưa có executor | chưa có executor | `51.169 / 92.888 ms` |

Ghi chú trung thực:

- `0/12 supplied` ở hai cột đầu nghĩa là baseline harness chưa có structured executor/answer injection; không được diễn giải thành 12 câu đã chạy executor nhưng trả sai.
- Cleanup giảm primary vector corpus từ `357` xuống `127` record, tức giảm khoảng `64.4%`.
- E5 tăng chi phí warm embedding/index so với FastText, nhưng tổng cold build giảm từ `12,080.1 ms` xuống `2,927.0 ms` (giảm khoảng `75.8%`) và chất lượng/gate tăng mạnh.

## 3. So sánh model và lý do chọn E5

So sánh cùng corpus 127 record; FastText rollback và E5 cuối có structured executor/hybrid hiện tại. Lượt Vietnamese proper dùng tokenizer/word segmentation đúng nhưng được chạy trước structured executor, vì vậy chỉ dùng metric vector/gate để chọn embedding.

| Model | Model load | Parse | Index | Acc@1 | Recall@5 | MRR | Refusal P/R | False-answer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FastText rollback | `14,428.4 ms` | `200.8 ms` | `173.2 ms` | `0.417` | `0.917` | `0.681` | `0.846 / 0.917` | `0.083` |
| Vietnamese candidate đúng tokenizer | `413.8 ms` | `251.1 ms` | `5,929.5 ms` | `0.167` | `0.500` | `0.283` | `0.500 / 1.000` | `0.000` nhưng false-refuse `12/12` vector |
| **multilingual E5 small ONNX** | **`1,139.9 ms`** | `208.5 ms` | **`1,578.6 ms`** | **`0.667`** | **`1.000`** | **`0.833`** | **`1.000 / 1.000`** | **`0.000`** |

Lý do bằng số:

- So với FastText rollback, E5 tăng Acc@1 `+0.250`, Recall@5 `+0.083`, MRR `+0.152`, đưa false-answer từ `0.083` về `0.000` và model load nhanh hơn khoảng `12.7x`.
- So với Vietnamese candidate, E5 tăng Acc@1 `+0.500`, Recall@5 `+0.500`, MRR `+0.550`; index nhanh hơn khoảng `4.35 giây`.
- `max_length=96` được giữ vì đạt Acc@1 `0.667`, Recall@5 `1.000`, MRR `0.833`, refusal P/R `1.000/1.000`; thử `64` giảm Acc@1 còn `0.333`, MRR còn `0.628` và refusal precision còn `0.857`.
- Batch size `16` được giữ vì lượt đo đạt total ingest sau prewarm `1,919.0 ms` với chất lượng không đổi; không thêm config knob chưa cần thiết.

## 4. Ngưỡng gate 0.84

`SIMILARITY_THRESHOLD=0.84` là điểm duy nhất trong sweep cuối đồng thời đạt:

- vector relevant-pass `1.000`;
- refusal precision `1.000`;
- refusal recall `1.000`;
- false-answer `0.000`;
- balanced score `1.000`.

Ở `0.82`, refusal recall chỉ `0.917` và false-answer `0.083`; ở `0.86`, vector-pass giảm còn `0.917`. Gate cuối yêu cầu cả cosine `>= 0.84` và `lexical_gate=True`; API không được tự nâng confidence.

## 5. Latency trước/sau

| Đường đo | Trước | Sau |
|---|---:|---:|
| Model load | `11,406.3 ms` baseline | `1,139.9 ms` E5 final |
| Load + parse + index | `12,080.1 ms` baseline | `2,927.0 ms` E5 final |
| Ingest sau prewarm | chưa có lifecycle tương đương | `1,700.7 ms` trong `data/offline-smoke.json`; các batch run `1,919.0–2,095.4 ms` |
| Warm core query | baseline all-phase khoảng `1.902 ms` | `23.5 ms` trong smoke cũ; final harness vector khoảng `22.344 ms` chưa tính LLM |
| Full startup lifecycle mới | chưa có | `2,763.8 ms` |
| API structured smoke | chưa có | `138.5 ms` |
| API vector smoke với fake local streaming LLM | chưa có | `43.3 ms` |
| API refuse smoke | chưa có | `33.0 ms` |

Không che giấu trade-off: warm vector query E5 chậm hơn FastText do ONNX embedding và rerank, nhưng cold startup nhanh hơn nhiều và chất lượng retrieval/refusal đạt mục tiêu.

## 6. Ví dụ hành vi cuối

1. Structured: “Có bao nhiêu sự cố AC trong VHKT?” → `6`, route `structured`, citation `Tong hop!E3`, `0` LLM call.
2. Vector: “N6 mất lộ điện nổi thì interlock chuyển nguồn như thế nào?” → route `retrieval`; smoke citation đầu `1.Ds UCTT!A6:I6`, similarity thực `0.8756759763`.
3. Vector paraphrase: “Nguồn lưu điện đang nuôi tải bằng pin vì đầu cấp bị ngắt; cách duy trì tải an toàn là gì?” → vector golden pass, locator liên quan nằm trong top 5; final top similarity `0.878`.
4. Refuse dưới ngưỡng: “Cách nấu phở bò ngon tại nhà” → top similarity `0.777 < 0.84`, route `refuse`, không LLM.
5. Refuse dù trùng từ khóa: “ACB99 tại N9 không đóng thì quy trình reset thế nào?” có top cosine `0.849` nhưng thiếu identifier trong bằng chứng nên `lexical_gate=False`, bị từ chối.
6. Refuse phủ định/trùng UPS: “Không hỏi về UPS; hãy nêu quy trình chữa cháy khí FM200.” có top cosine `0.884` nhưng `lexical_gate=False`; full startup smoke xác nhận route `refuse`, không tăng số LLM call.
7. Mâu thuẫn summary/detail: “Nếu sheet Tổng hợp ghi AC trong VHKT là 6 nhưng danh sách chi tiết chỉ có 5 thì phải dùng số nào?” → structured guard từ chối, không rơi xuống RAG để bịa câu trả lời.

## 7. Offline full startup lifecycle smoke

Artifact: `data/offline-startup-smoke.json`.

- Dùng `create_app(Settings)` thật, `get_service` thật, E5 bundle local tại `D:\Playground\AImpact\project_agent\models\multilingual-e5-small-onnx`.
- Temp workspace có `documents/Phu luc 1.xlsx`, database/data/history riêng; `TestClient` chạy đầy đủ lifespan nên model prewarm và startup ingest/migration thực sự chạy.
- Fake streaming LLM local được inject trước khi mở `TestClient`; chỉ vector route gọi đúng `1` lần.
- `socket.create_connection`, `socket.socket.connect`, `socket.socket.connect_ex` bị chặn trước lifespan. Chỉ cho phép stdlib `socketpair` nội bộ tạo self-pipe của event loop TestClient; không tính là kết nối mạng ứng dụng.
- Kết quả: `status=passed`, `startup_ms=2763.8`, `network_attempts=0`, `index_count=127`.
- Structured: `138.5 ms`, route `structured`, answer `6`, locator `Tong hop!E3`, LLM calls sau câu = `0`.
- Vector: `43.3 ms`, route `retrieval`, có 5 citation, top similarity thực `0.8756759763`, LLM calls sau câu = `1`.
- Refuse keyword-overlap: `33.0 ms`, route `refuse`, citation rỗng, LLM calls vẫn = `1`.

Lượt smoke đầu tiên chặn cả `socket.socket.connect` mà không ngoại lệ cho `socketpair`, nên Windows asyncio không tạo được self-pipe. Lượt nghiệm thu đã sửa harness tạm thời, không sửa product/test code; các assertion hoàn tất và JSON được ghi trước khi console Windows phát sinh lỗi encode khi in tiếng Việt. Artifact sau đó được parse/xác minh lại bằng PowerShell với exit code `0`.

## 8. Dependency

Dependency runtime mới được duyệt:

- `onnxruntime==1.20.0`: chạy model E5 ONNX hoàn toàn local, không tải model lúc runtime.
- `tokenizers==0.23.1`: đọc tokenizer bundle local, padding/truncation và attention mask nhất quán.

`sonner` đã bị gỡ khỏi `web/package.json`, `web/package-lock.json`, source và local `node_modules`; scan `rg -i sonner web` trả `0` kết quả. UI dùng helper `web/src/notify.ts` và CSS/pattern sẵn có, không thêm runtime dependency khác.

## 9. Phân công instance

- Instance A: xóa lexical bypass trong `api/routes.py`; wiring `/api/query` theo structured executor → core vector query → calibrated `rag.answer_or_refuse`.
- Instance B: sửa startup migration, rollback/fallback và regression tests, gồm root-cause checksum seed khi upload workbook đã được startup ingest.
- Instance C: gỡ `sonner`, thay notification bằng helper/CSS sẵn có, giữ route/source badge và build frontend.
- Instance D: integration cuối; forbidden scans, full backend suite, frontend build, offline startup lifecycle smoke, Git integrity check và báo cáo này.

## 10. Checklist bất biến an toàn

- [x] Thiếu bằng chứng → từ chối: core gate dùng threshold thật và `lexical_gate`; smoke phủ định UPS xác nhận không gọi LLM.
- [x] Không similarity giả: không còn symbol bypass/direct answer và không có `similarity=1.0` trong `api/routes.py`/API tests.
- [x] Citation nguyên văn: retrieval projection tách khỏi `text_verbatim`; citation trả nội dung đã giải mã, không trả text đã normalize làm bằng chứng.
- [x] Fernet: nội dung index vẫn mã hóa bằng khóa local; nâng cấp không bỏ hoặc làm yếu encryption.
- [x] RBAC: endpoint tiếp tục dùng `admin_only`/`all_roles`; full API suite xanh.
- [x] Conversation isolation: history/message lookup vẫn scope theo `conversation_id` và `user_id`; full suite xanh.
- [x] Audit không chứa nội dung: query audit ghi action và conversation id, không ghi raw query/prompt/citation content; regression test xanh.
- [x] Offline runtime: local manifest/hash + ONNX/tokenizer bundle; smoke `network_attempts=0`.
- [x] Startup migration an toàn: nếu collection mới rỗng, ingest tài liệu hỗ trợ và luôn bổ sung stats workbook nếu thiếu tên; lỗi thì reset collection mới và rethrow, không phục vụ collection dở.
- [x] Rollback collection cũ: collection mới được đặt tên theo embedding model ID; collection cũ không bị ghi đè/xóa; FastText file-path vẫn là rollback.
- [x] Không commit/push: HEAD vẫn `e81656a`; toàn bộ thay đổi nằm trong working tree `feature/rag-upgrade`.

## 11. Hạn chế và mục tiêu chưa đạt

- Các sheet `Form *` thực tế có nội dung chi tiết riêng, không hoàn toàn rỗng. Parser/source vẫn giữ nguyên văn nhưng chúng bị loại khỏi primary vector retrieval để giảm nhiễu; nếu sau này cần truy vấn trực tiếp Form, phải thiết kế lane/index riêng thay vì lặng lẽ trộn lại.
- Skill `frontend-design` không có trong môi trường; route/source badge dùng pattern và palette sẵn có, không thêm dependency.
- Quantized ONNX trial không hợp lệ với ONNX Runtime trong môi trường này và đã bị xóa; không báo số quantized như một benchmark hợp lệ.
- `web/src/lib/types.ts` đang bị rule `.gitignore` `lib/` bỏ qua. File tồn tại và build hiện tại dùng được, nhưng một commit tương lai phải force-add file hoặc thu hẹp rule ignore; integration D không được phép sửa product/config ngoài report.
- Full startup `2,763.8 ms` đạt dưới 3 giây trong lượt nghiệm thu. Tuy nhiên latency phụ thuộc máy/cache; không cam kết SLA cứng chỉ từ một lượt đo.
- `git diff --stat` không tính file untracked/ignored; vì vậy raw status và danh sách đầy đủ bên dưới là nguồn bổ sung bắt buộc.
- Warning còn lại là `StarletteDeprecationWarning` từ `fastapi.testclient` về `httpx`; không thêm `httpx2` vì bị cấm thêm dependency runtime và toàn bộ test vẫn xanh.

## 12. Danh sách đầy đủ file thêm/sửa/xóa

### File tracked đã sửa (37)

```text
.env.example
README.md
api/deps.py
api/main.py
api/routes.py
api/settings.py
docker-compose.yml
docs/ENV.md
docs/OPERATIONS.md
project_agent/README.md
project_agent/config.py
project_agent/embedding.py
project_agent/index.py
project_agent/ingest.py
project_agent/rag.py
project_agent/requirements.txt
project_agent/scripts/eval_retrieval.py
project_agent/scripts/golden_queries.json
project_agent/service.py
project_agent/tests/test_config.py
project_agent/tests/test_embedding.py
project_agent/tests/test_index.py
project_agent/tests/test_ingest.py
project_agent/tests/test_rag.py
project_agent/tests/test_service.py
requirements-api.txt
tests_api/conftest.py
tests_api/test_api.py
web/package-lock.json
web/package.json
web/src/App.tsx
web/src/pages/Admin.tsx
web/src/pages/Chat.tsx
web/src/pages/Documents.tsx
web/src/pages/Login.tsx
web/src/pages/Stats.tsx
web/src/styles.css
```

### File source/report mới, đang untracked hoặc ignored

```text
docs/RAG_UPGRADE_REPORT.md
project_agent/structured.py
project_agent/tests/test_corpus_cleanup.py
project_agent/tests/test_eval_retrieval.py
project_agent/tests/test_structured.py
web/src/notify.ts
web/src/lib/types.ts                 # ignored bởi rule lib/
data/offline-startup-smoke.json     # ignored artifact
```

### Generated/ignored đã chạm khi build

```text
web/dist/index.html
web/dist/assets/Admin-Mu5vP8Gc.js
web/dist/assets/Chat-B4EIMV50.js
web/dist/assets/Documents-CJhAQonf.js
web/dist/assets/Login-D5KWnJ9-.js
web/dist/assets/Stats-BSuF0OE2.js
web/dist/assets/index-Bqk_rwQS.js
web/dist/assets/index-Dgralq0Q.css
web/dist/assets/notify-DVlLj0Fe.js
web/tsconfig.app.tsbuildinfo         # build chạm, sau đó restore về HEAD; không còn diff
web/node_modules/sonner/             # local ignored dependency directory đã xóa
```

Không có file tracked bị xóa hoặc rename trong patch hiện tại. Các temp workspace của smoke đã được dọn bằng `shutil.rmtree(..., ignore_errors=True)`.

## 13. Raw `git status --short`

```text
 M .env.example
 M README.md
 M api/deps.py
 M api/main.py
 M api/routes.py
 M api/settings.py
 M docker-compose.yml
 M docs/ENV.md
 M docs/OPERATIONS.md
 M project_agent/README.md
 M project_agent/config.py
 M project_agent/embedding.py
 M project_agent/index.py
 M project_agent/ingest.py
 M project_agent/rag.py
 M project_agent/requirements.txt
 M project_agent/scripts/eval_retrieval.py
 M project_agent/scripts/golden_queries.json
 M project_agent/service.py
 M project_agent/tests/test_config.py
 M project_agent/tests/test_embedding.py
 M project_agent/tests/test_index.py
 M project_agent/tests/test_ingest.py
 M project_agent/tests/test_rag.py
 M project_agent/tests/test_service.py
 M requirements-api.txt
 M tests_api/conftest.py
 M tests_api/test_api.py
 M web/package-lock.json
 M web/package.json
 M web/src/App.tsx
 M web/src/pages/Admin.tsx
 M web/src/pages/Chat.tsx
 M web/src/pages/Documents.tsx
 M web/src/pages/Login.tsx
 M web/src/pages/Stats.tsx
 M web/src/styles.css
?? docs/RAG_UPGRADE_REPORT.md
?? project_agent/structured.py
?? project_agent/tests/test_corpus_cleanup.py
?? project_agent/tests/test_eval_retrieval.py
?? project_agent/tests/test_structured.py
?? web/src/notify.ts
```

## 14. Raw `git diff --stat`

```text
 .env.example                              |   8 +-
 README.md                                 |  19 +-
 api/deps.py                               |  23 +-
 api/main.py                               |  29 ++
 api/routes.py                             | 122 ++----
 api/settings.py                           |   9 +-
 docker-compose.yml                        |   6 +-
 docs/ENV.md                               |  12 +-
 docs/OPERATIONS.md                        |  24 +-
 project_agent/README.md                   |  43 +-
 project_agent/config.py                   |   8 +-
 project_agent/embedding.py                | 122 ++++++
 project_agent/index.py                    | 188 +++++++-
 project_agent/ingest.py                   |  16 +
 project_agent/rag.py                      |   8 +-
 project_agent/requirements.txt            |   1 +
 project_agent/scripts/eval_retrieval.py   | 682 ++++++++++++++++++++++++------
 project_agent/scripts/golden_queries.json | 322 +++++++++++++-
 project_agent/service.py                  |  60 ++-
 project_agent/tests/test_config.py        |  22 +
 project_agent/tests/test_embedding.py     | 114 +++++
 project_agent/tests/test_index.py         |  91 +++-
 project_agent/tests/test_ingest.py        |  19 +
 project_agent/tests/test_rag.py           |  14 +
 project_agent/tests/test_service.py       | 114 ++++-
 requirements-api.txt                      |   2 +
 tests_api/conftest.py                     |  26 +-
 tests_api/test_api.py                     | 253 +++++++++--
 web/package-lock.json                     |  13 +-
 web/package.json                          |   3 +-
 web/src/App.tsx                           |  32 +-
 web/src/pages/Admin.tsx                   |  36 +-
 web/src/pages/Chat.tsx                    |  30 +-
 web/src/pages/Documents.tsx               |  12 +-
 web/src/pages/Login.tsx                   |   4 +-
 web/src/pages/Stats.tsx                   |   4 +-
 web/src/styles.css                        |   6 +
 37 files changed, 2055 insertions(+), 442 deletions(-)
```

## 15. Lệnh nghiệm thu cuối

```text
D:\Playground\AImpact\.codex-feature-web-ui\.venv\Scripts\python.exe -m pytest project_agent/tests tests_api -q
# 106 passed, 1 warning in 13.50s

npm --prefix web run build
# vite v8.1.5, 36 modules transformed, built in 783ms

git diff --check
# exit 0

rg -n "_exact_lexical_hits|_is_count_query|_QUERY_FILLER_WORDS|_normalized_tokens|direct_answer|similarity\s*=\s*1(\.0)?" api/routes.py tests_api
# zero matches

rg -ni sonner web
# zero matches
```
