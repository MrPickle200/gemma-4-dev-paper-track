# IDEA.md: Micro-Model Native Harness cho Gemma 4 (E2B / E4B)

## 1. Giả thuyết nghiên cứu (Core Hypothesis)
Các mô hình nhỏ có kiến trúc Per-Layer Embeddings (Gemma 4 E2B/E4B) thất bại trong tác vụ SWE-bench không hoàn toàn do thiếu khả năng suy luận logic, mà do hai điểm nghẽn hệ thống:
1. **Context Degradation:** Suy giảm chất lượng chú ý nhanh chóng khi ngữ cảnh bị lấp đầy bởi kết quả tìm kiếm thô (`grep`/`cat`) và log lỗi dài.
2. **Syntactic Fragility:** Tỷ lệ lỗi cú pháp cao khi sinh định dạng phức tạp (JSON tool calls, unified diffs dựa trên số dòng).

**Giải pháp:** Chuyển dịch gánh nặng từ *Model Reasoning* sang *Deterministic Harness* thông qua việc giới hạn không gian hành động bằng AST, ép khuôn đầu ra (Constrained Decoding) và thu gọn ngữ cảnh theo pha (Episodic Context Compaction).

---

## 2. Thiết kế kiến trúc Harness (3 Modules)

### Module 1: AST-Guided Navigation (Thay thế `grep` / `cat`)
Dùng `tree-sitter` xây dựng bộ công cụ điều hướng codebase ở mức ký hiệu (symbol-level) để giảm lượng token nạp vào context:
- `get_repo_map(path)`: Trả về khung xương của file/thư mục (chỉ tên class, hàm, chữ ký hàm và docstring, ẩn toàn bộ phần thân hàm).
- `read_symbol(file, symbol_name)`: Chỉ nạp mã nguồn của đúng hàm hoặc class được chỉ định.
- `find_references(symbol_name)`: Trả về danh sách vị trí gọi hàm trong repo.

### Module 2: AST-Guarded Editing (Thay thế `git apply` / `sed`)
Loại bỏ hoàn toàn việc sửa code theo số dòng (line numbers):
- **Structured Output:** Ép mô hình xuất đúng schema `{"file": str, "target_symbol": str, "search_block": str, "replace_block": str}` bằng constrained decoding. Trên `llama.cpp` dùng `response_format={"type": "json_schema", ...}` — tương đương `guided_json` của vLLM/SGLang, đã kiểm chứng chạy được ở Bước 1.
- **In-Memory Syntax Gate:** Áp dụng thay thế trên bộ nhớ tạm và chạy `ast.parse()` (với Python) hoặc `tree-sitter` check. Nếu có `SyntaxError` hoặc `IndentationError`, chặn việc ghi xuống ổ cứng và trả về thông báo lỗi kèm dòng bị sai để mô hình sửa lại ngay trong turn tiếp theo (không làm hỏng trạng thái repo).

### Module 3: Episodic Context Compaction (Bộ nhớ phân tầng)
Chia quy trình của Agent thành 2 pha tách biệt với cơ chế tự động dọn dẹp context:
- **Pha 1 (Localization):** Agent dùng công cụ điều hướng để tìm vị trí lỗi. Kết thúc pha bằng việc gọi `lock_targets(files, symbols, root_cause_note)`.
- **Compaction Trigger:** Ngay khi `lock_targets` được gọi, Harness **xóa toàn bộ lịch sử tìm kiếm nháp** trước đó, khởi tạo một context mới sạch hoàn toàn chỉ gồm: *Issue Description + Mã nguồn của các `symbols` đã khóa + `root_cause_note`*.
- **Pha 2 (Patching & Verification):** Agent thực hiện sửa code và chạy `pytest` trên không gian ngữ cảnh đã được rút gọn.

---

## 3. Thiết lập thực nghiệm (Weekend MVP Scope)

- **Models:** `Gemma-4-E2B-IT` và `Gemma-4-E4B-IT`, chạy qua **`llama.cpp`** (`llama-server`, CUDA) với GGUF Q4_0 QAT, `temperature=0.0`, `max_tokens=16384`, tối đa 15 turns/task.
  - *Lý do thay vLLM:* GPU của máy này chỉ có 6 GB VRAM, không đủ chứa trọng số bf16 (E2B 10.25 GB, E4B 16 GB), và vLLM chỉ chạy trên Linux. Harness chỉ giao tiếp qua API OpenAI-compatible nên có thể thay lại vLLM khi có GPU lớn hơn.
  - *Tham số đo được (E2B Q4_0):* nạp model ~10 s, sinh ~85 tok/s, prefill chưa cache ~13 tok/s.
  - *Thinking mode:* template mặc định `enable_thinking=false` nhưng tự bật khi request có `tools` hoặc message đầu là `system`/`developer`. CoT nằm ở `reasoning_content`, câu trả lời ở `content` — harness phải log cả hai và ghim cờ này theo từng pha.
- **Benchmark:** **SWE-bench Verified Mini (N = 50)**.
  - *Tiêu chí lọc:* Các task từ `SWE-bench Verified` có bản vá chuẩn (Gold patch) chỉ sửa trên 1 file, dưới 50 dòng code, thuộc các repo có thời gian chạy test nhanh (`pytest`, `requests`, `sympy`, `scikit-learn`).

### Ma trận Ablation (Chạy trên cả E2B và E4B)
1. **`B0 (Baseline)`:** Bash tool tiêu chuẩn (`grep`, `find`, `cat`) + sửa file bằng `search/replace` thô + giữ nguyên lịch sử hội thoại.
2. **`M1 (+AST-Edit)`:** `B0` + Constrained JSON Decoding + In-Memory Syntax Gate.
3. **`M2 (+AST-Nav)`:** `M1` + Thay thế tìm kiếm bash bằng bộ công cụ `tree-sitter`.
4. **`M3 (Full Micro-Harness)`:** `M2` + Episodic Context Compaction giữa pha Localization và Patching.

---

## 4. Các chỉ số đo lường (Metrics cho Paper)
Từ file log `.jsonl` của các lượt chạy, trích xuất 4 nhóm số liệu chính để đưa vào bài báo:
1. **Resolve Rate (%):** Tỷ lệ giải quyết thành công issue qua bộ test chuẩn của SWE-bench.
2. **Malformed Edit Rate (%):** Tỷ lệ các lệnh sửa code bị lỗi cú pháp hoặc không khớp khối `search` (chứng minh hiệu quả của `M1`).
3. **Peak Context Length & Total Tokens:** Lượng token tối đa trong một turn và tổng token tiêu thụ trên mỗi task (chứng minh hiệu quả của `M2` và `M3`).
4. **Error Rate vs. Context Depth (Biểu đồ phân tích):** Tỷ lệ gọi sai công cụ/lặp vô hạn của E2B/E4B theo độ dài context (2K, 4K, 8K, 12K, 16K) để làm rõ giới hạn thực tế của kiến trúc PLE.

---

## 5. Danh sách việc cần làm (Action Items)

- [x] **Bước 1 (Data & Infra):** Tạo môi trường ảo (`.venv`, Python 3.11.9) và hoạt động trong đó. Viết script lọc 50 mẫu từ `SWE-bench Verified` ra file `subset_50.json` (117/500 mẫu đạt tiêu chí, lấy 50 với `seed=0`). Khởi chạy **`llama-server`** (llama.cpp, CUDA) cho Gemma 4 E2B và E4B — đã PASS acceptance test `scripts/smoke_test.py`.
- [x] **Bước 2 (Core Loop):** Viết vòng lặp Agent cơ bản hỗ trợ xuất log `.jsonl` (lưu đầy đủ prompt, response, token count, tool status ở mỗi turn). Hoàn tất: `harness/` + `scripts/run_task.py`; mỗi run ghi `logs/runs/<run_id>.jsonl` và `logs/patches/<run_id>.patch`. Đã chạy 9 runs trên E2B (xem `PROGRESS.md`).
- [ ] **Bước 3 (Implement Modules):**
  - [ ] Code module `AST-Edit` (schema JSON cho llama.cpp `response_format` + cổng `ast.parse` trong bộ nhớ).
  - [ ] Code module `AST-Nav` bằng `tree_sitter` (3 hàm: `get_repo_map`, `read_symbol`, `find_references`).
  - [ ] Code hàm `compact_context()` khi chuyển pha.
- [ ] **Bước 4 (Execution):** Chạy batch 4 cấu hình (`B0`, `M1`, `M2`, `M3`) $\times$ 2 models trên 50 tasks.
- [ ] **Bước 5 (Evaluation & Plotting):** Chạy Docker evaluation của SWE-bench cho các bản vá sinh ra; chạy script tổng hợp bảng Ablation và vẽ biểu đồ suy giảm hiệu năng theo độ dài context.