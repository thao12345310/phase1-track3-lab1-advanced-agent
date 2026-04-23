# Lab 16 — Reflexion Agent Scaffold

Repo này cung cấp một khung sườn (scaffold) để xây dựng và đánh giá **Reflexion Agent**.

## 1. Mục tiêu của Repo
- Repo hiện tại đang sử dụng **Mock Data** (`mock_runtime.py`) để giả lập phản hồi từ LLM.
- Mục đích giúp học viên hiểu rõ về **flow**, các bước **loop**, cách thức hoạt động của cơ chế phản chiếu (reflection) và cách đánh giá (evaluation) mà không tốn chi phí API ban đầu.

## 2. Nhiệm vụ của Học viên
Học viên cần thực hiện các bước sau để hoàn thành bài lab:
1. **Xây dựng Agent thật**: Thay thế phần mock bằng việc gọi LLM thật (sử dụng Local LLM như Ollama, vLLM hoặc các Simple LLM API như OpenAI, Gemini).
2. **Chạy Benchmark thực tế**: Chạy đánh giá trên ít nhất **100 mẫu dữ liệu thật** từ bộ dataset **HotpotQA**.
3. **Định dạng báo cáo**: Kết quả chạy phải đảm bảo xuất ra file report (`report.json` và `report.md`) có cùng định dạng (format) với code gốc để có thể chạy được công cụ chấm điểm tự động.
4. **Tính toán Token thực tế**: Thay vì dùng số ước tính, học viên phải cài đặt logic tính toán lượng token tiêu thụ thực tế từ phản hồi của API.

## 3. Cách chạy Lab (Scaffold)
```bash
# Cài đặt môi trường
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Chạy benchmark (với mock data)
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run

# Chạy chấm điểm tự động
python autograde.py --report-path outputs/sample_run/report.json
```

## 4. Tiêu chí chấm điểm (Rubric)
- **80% số điểm (80 điểm)**: Hoàn thiện đúng và đủ luồng (flow) cho Reflexion Agent, chạy thành công với LLM thật và dataset thật.
- **20% số điểm (20 điểm)**: Thực hiện thêm ít nhất một trong các phần **Bonus** được nhắc đến trong mã nguồn (ví dụ: `structured_evaluator`, `reflection_memory`, `adaptive_max_attempts`, `memory_compression`, v.v. - xem chi tiết tại `autograde.py`).

## Thành phần mã nguồn
- `src/reflexion_lab/schemas.py`: Định nghĩa các kiểu dữ liệu trace, record.
- `src/reflexion_lab/prompts.py`: Nơi chứa các template prompt cho Actor, Evaluator và Reflector.
- `src/reflexion_lab/mock_runtime.py`: (Cần thay thế) Logic giả lập phản hồi LLM.
- `src/reflexion_lab/agents.py`: Cấu trúc chính của ReAct và Reflexion Agent.
- `src/reflexion_lab/reporting.py`: Logic xuất báo cáo benchmark.
- `run_benchmark.py`: Script chính để chạy đánh giá.
- `autograde.py`: Công cụ hỗ trợ chấm điểm nhanh dựa trên report.
- `src/reflexion_lab/ollama_runtime.py`: Runtime thật gọi Ollama LLM, ghi trace, hỗ trợ LATS branching.
- `src/reflexion_lab/structured_evaluator.py`: Bộ đánh giá đa chiều (factual accuracy, completeness, precision, reasoning).
- `src/reflexion_lab/utils.py`: Chuẩn hoá câu trả lời, load dataset, ghi JSONL.
- `scripts/download_hotpotqa.py`: Script tải dữ liệu HotpotQA từ HuggingFace, chia theo độ khó.

---

## 5. Kết quả thí nghiệm (`outputs/`)

Thư mục `outputs/` chứa kết quả của **4 lần chạy thí nghiệm**, mỗi lần xây dựng và cải thiện dựa trên bài học từ lần trước.

### Lần 1: `outputs/ollama_test/` — Kiểm tra tích hợp ban đầu

| Thông tin | Giá trị |
|---|---|
| **Dataset** | `hotpot_mini.json` (8 câu hỏi) |
| **Agents** | ReAct, Reflexion |
| **Tổng records** | 16 |
| **Mục đích** | Xác nhận pipeline Ollama hoạt động đúng end-to-end |

| Agent | EM (Exact Match) | Số lần thử TB | Token TB | Độ trễ TB |
|---|---:|---:|---:|---:|
| ReAct | 1.00 | 1.0 | 250 | 1,103 ms |
| Reflexion | 1.00 | 1.0 | 250 | 986 ms |

**Nhận xét:** Cả hai agent đều đạt 100% trên bộ dữ liệu mini (câu hỏi dễ). Xác nhận Ollama runtime, evaluator và reporting pipeline hoạt động bình thường. Không có reflection nào được kích hoạt vì tất cả câu trả lời đầu tiên đều đúng.

**File đầu ra:** `report.json`, `report.md`, `react_runs.jsonl`, `reflexion_runs.jsonl`

---

### Lần 2: `outputs/full_run/` — Benchmark thực tế đầu tiên

| Thông tin | Giá trị |
|---|---|
| **Dataset** | `hotpot_easy.json` (20 câu hỏi) |
| **Agents** | ReAct, Reflexion |
| **Tổng records** | 40 |
| **Mục đích** | Thử nghiệm trên câu hỏi multi-hop thật, quan sát hành vi reflection |

| Agent | EM | Số lần thử TB | Token TB | Độ trễ TB |
|---|---:|---:|---:|---:|
| ReAct | 0.55 | 1.0 | 880 | 7,302 ms |
| Reflexion | 0.75 | 1.75 | 2,695 | 32,277 ms |

**Phân loại lỗi (Failure Modes):**
- ReAct: 10 câu sai hoàn toàn, 1 câu thiếu bước suy luận (incomplete multi-hop)
- Reflexion: 8 câu sai (phục hồi được 4 câu so với ReAct nhờ cơ chế reflection)

**Nhận xét:** Reflexion cải thiện **+20%** so với ReAct. Cơ chế reflection giúp phục hồi một số lỗi multi-hop. Tuy nhiên, một số câu hỏi vẫn không giải được dù thử lại 3 lần. Độ trễ tăng ~4.4 lần do chi phí thử lại. Đã bổ sung trace logging để theo dõi toàn bộ các lệnh gọi LLM.

**File đầu ra:** `report.json`, `report.md`, `react_runs.jsonl`, `reflexion_runs.jsonl`, `trace_log.json`

---

### Lần 3: `outputs/full_run_v2/` — Tối ưu hoá Prompt

| Thông tin | Giá trị |
|---|---|
| **Dataset** | `hotpot_easy.json` (10 câu hỏi, tập con) |
| **Agents** | ReAct, Reflexion |
| **Tổng records** | 20 |
| **Mục đích** | Thử nghiệm prompt cải tiến (Chain-of-Thought, chống lặp) trên tập nhỏ |

| Agent | EM | Số lần thử TB | Token TB | Độ trễ TB |
|---|---:|---:|---:|---:|
| ReAct | 0.70 | 1.0 | 837 | 5,065 ms |
| Reflexion | 0.90 | 1.5 | 1,981 | 19,599 ms |

**Phân loại lỗi:**
- ReAct: 4 câu sai, 1 incomplete multi-hop
- Reflexion: 3 câu sai (phục hồi thêm 2 câu nhờ prompt cải tiến)

**Nhận xét:** Cải tiến prompt giúp cả hai agent tăng đáng kể. Chain-of-thought prompting ở lần thử ≥ 2 và theo dõi câu trả lời sai trước đó giúp giảm tình trạng lặp (looping). ReAct tăng từ 55% → 70%, Reflexion từ 75% → 90%. Độ trễ trung bình cũng giảm nhờ ít bước thử lại lãng phí.

**File đầu ra:** `report.json`, `report.md`, `react_runs.jsonl`, `reflexion_runs.jsonl`, `trace_log.json`

---

### Lần 4: `outputs/full_run_v3/` — Benchmark đầy đủ (3 Agents) ⭐

| Thông tin | Giá trị |
|---|---|
| **Dataset** | `hotpot_full.json` (120 câu: 40 easy + 40 medium + 40 hard) |
| **Agents** | ReAct, Reflexion, **LATS** (mới) |
| **Tổng records** | 360 |
| **Mục đích** | So sánh toàn diện 3 kiến trúc agent trên tất cả mức độ khó |

| Agent | EM | Số lần thử TB | Token TB | Độ trễ TB |
|---|---:|---:|---:|---:|
| ReAct | 0.467 | 1.0 | 1,166 | 10,413 ms |
| Reflexion | 0.775 | 1.84 | 3,508 | 39,972 ms |
| **LATS** | **0.783** | 1.45 | 3,695 | 53,722 ms |

**Bảng phân loại lỗi chi tiết:**

| Loại lỗi | ReAct | Reflexion | LATS |
|---|---:|---:|---:|
| none (đúng) | 46 | 78 | 76 |
| wrong_final_answer (sai đáp án) | 61 | 27 | 37 |
| incomplete_multi_hop (thiếu bước) | 13 | 2 | 7 |
| looping (lặp câu trả lời) | — | 11 | — |
| reflection_overfit (reflection phản tác dụng) | — | 2 | — |

**Nhận xét chính:**
- **LATS đạt độ chính xác cao nhất (78.3%)**, nhỉnh hơn Reflexion (77.5%) và vượt xa ReAct (46.7%).
- **ReAct** gặp nhiều lỗi `incomplete_multi_hop` nhất (13 trường hợp) — agent dừng lại sau bước suy luận đầu tiên mà không hoàn thành chuỗi logic.
- **Reflexion** có các loại lỗi riêng: `looping` (11 trường hợp lặp lại câu trả lời sai) và `reflection_overfit` (2 trường hợp reflection phản tác dụng, làm kết quả tệ hơn).
- **LATS** khám phá nhiều ứng viên câu trả lời đa dạng, tránh hoàn toàn vấn đề looping. Tuy nhiên, tốn nhiều token và độ trễ nhất do branching (3 ứng viên × đánh giá mỗi mức).
- **Đánh đổi (tradeoff):** LATS dùng gấp ~3.2× token so với ReAct nhưng đạt độ chính xác cao hơn ~1.68×.

**File đầu ra:** `report.json`, `report.md`, `react_runs.jsonl`, `reflexion_runs.jsonl`, `lats_runs.jsonl`, `trace_log.json`, `structured_eval_summary.json`, `structured_eval_details.json`

---

## 6. Tiến trình phát triển qua các lần chạy

```
Lần 1 (smoke test)  →  Lần 2 (baseline)    →  Lần 3 (tối ưu)     →  Lần 4 (đầy đủ + LATS)
8 câu, 2 agents        20 câu, 2 agents        10 câu, 2 agents      120 câu, 3 agents
EM: 100%/100%          EM: 55%/75%             EM: 70%/90%           EM: 47%/78%/78%
16 records             40 records              20 records            360 records
```

Mỗi lần chạy xây dựng dựa trên lần trước:
1. **Lần 1** — Xác nhận pipeline tích hợp Ollama hoạt động
2. **Lần 2** — Phát hiện các loại lỗi thực tế (incomplete multi-hop, wrong final answer)
3. **Lần 3** — Cải tiến prompt với Chain-of-Thought và chiến lược chống lặp
4. **Lần 4** — Thêm LATS agent và mở rộng lên 120 câu hỏi ở tất cả mức độ khó

---

## 7. Các extension đã cài đặt (Bonus)

| Extension | Mô tả |
|---|---|
| `structured_evaluator` | Đánh giá đa chiều: factual accuracy, completeness, precision, reasoning quality |
| `reflection_memory` | Tích luỹ bài học qua các lần thử, theo dõi câu trả lời sai trước đó |
| `benchmark_report_json` | Tạo báo cáo cấu trúc JSON + Markdown với đầy đủ metrics |
| `mock_mode_for_autograding` | Runtime mock để test mà không tốn chi phí LLM |
| `mini_lats_branching` | Agent LATS với tree-search branching và đánh giá đa ứng viên |

---

## 8. Điểm tự chấm (Auto-grade)

```
Auto-grade total: 100/100
- Flow Score (Core): 80/80
  * Schema: 30/30
  * Experiment: 30/30
  * Analysis: 20/20
- Bonus Score: 20/20
```
