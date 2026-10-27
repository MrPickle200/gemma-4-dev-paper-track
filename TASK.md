# TASK.md: Triển khai Cấu hình M4 (Zero-Turn AST-Semantic Seeding) & Hoàn thiện Micro-Harness

## Bối cảnh & Mục tiêu

Sau khi hoàn thành các cấu hình `B0` $\rightarrow$ `M3` trên tập 50 tasks của `SWE-bench Verified`:
- `E4B` tăng trưởng tốt từ **6% (`B0`) $\rightarrow$ 12% (`M1`) $\rightarrow$ 16% (`M2`, `M3`)**.
- Tuy nhiên, `E2B` đạt **10% ở `M2`** nhưng bị tụt xuống **6% ở `M3`** (do mô hình 2B định vị thiếu hoặc sai symbol ở Pha 1 trước khi gọi `lock_targets`), đồng thời lỗi `SYMBOL_NOT_FOUND` vẫn chiếm tỷ trọng lớn nhất ở cả `M2` (45.1%) và `M3` (65.1%).

**Mục tiêu của đợt thực nghiệm tiếp theo (`M4`):**
Tích hợp mô hình nhúng **Embedding Gemma 2** theo cơ chế **Zero-Turn Seeding (chạy 1 lần duy nhất ở Turn 0 trước khi LLM bắt đầu vòng lặp)**. Không cung cấp thêm tool tìm kiếm cho LLM để tránh làm phình không gian hành động và tránh tốn thêm turns.

---

## Phần 1: Thiết kế kỹ thuật của `M4` (Zero-Turn AST-Semantic Seeding)

### 1.1. Đơn vị đánh chỉ mục (AST-Symbol Level Indexing)
Không cắt nhỏ file theo số lượng ký tự/token cố định. Tận dụng bộ parser `tree-sitter` sẵn có của `M2`/`M3` để trích xuất toàn bộ các **AST Symbols** (functions, classes, methods) trong worktree của repository.
- Bỏ qua các thư mục test (`tests/`, `test_*.py`), thư mục ẩn và docs (trừ khi trong đề bài nhắc trực tiếp đến tên file test).
- Với mỗi symbol, tạo một chuỗi văn bản đại diện (`document_text`) theo định dạng chuẩn:
  ```text
  File: {relative_file_path}
  Symbol: {relative_file_path}::{qualified_symbol_name}
  Signature: {def_or_class_signature}
  Docstring: {first_300_chars_of_docstring}
  Body Preview: {first_10_non_empty_lines_of_body}
  ```

### 1.2. Cơ chế Hybrid Retrieval (Embedding Gemma + Lexical Boost) tại Turn 0
Khi bắt đầu một task, dùng chính `problem_statement` (Issue Description) làm câu truy vấn (`query`):
1. **Dense Retrieval (Embedding Gemma 2):**
   - Mã hóa toàn bộ `document_text` của các AST symbols và `problem_statement` thành các vector chuẩn hóa ($L_2$-normalized).
   - Tính điểm tương đồng cosine: $S_{\text{dense}} = \mathbf{q} \cdot \mathbf{d}_i$.
2. **Lexical / Identifier Boost (BM25 hoặc Exact Identifier Match):**
   - Trích xuất các định danh kỹ thuật trong `problem_statement` (các từ nằm trong dấu backtick `` `...` ``, CamelCase, hoặc `snake_case` như `fit_predict`, `CountVectorizer`, `Session.request`).
   - Tính điểm khớp từ khóa $S_{\text{lex}}$ (hoặc dùng Reciprocal Rank Fusion - RRF kết hợp thứ hạng giữa Dense Embedding và BM25 với $k=60$).
3. **Chọn Top-$K$ ($K = 8$) Candidate Symbols:**
   - Lấy ra 8 symbols có điểm kết hợp cao nhất.

### 1.3. Bơm ngữ cảnh vào Prompt mở đầu (Turn-0 Prompt Injection)
Ở cấu hình `M4`, bộ công cụ (`tools_offered`) và cơ chế `lock_targets` giữ nguyên 100% như `M3`. Điểm khác biệt duy nhất là nối thêm khối **`[AUTO-RETRIEVED CANDIDATE SYMBOLS]`** vào cuối message chứa đề bài ở Turn 0:

```text
=== AUTO-RETRIEVED CANDIDATE SYMBOLS (Turn-0 AST-Semantic Seeding) ===
The following existing symbols in the repository were automatically ranked as most relevant to the issue description. Use exact IDs from this list when calling `read_symbol` or `lock_targets`:

1. `sklearn/mixture/base.py::BaseMixture.fit_predict`
   Signature: def fit_predict(self, X, y=None):
   Summary: Estimate model parameters using X and predict the labels for X.

2. `sklearn/mixture/base.py::BaseMixture._e_step`
   Signature: def _e_step(self, X):
   ...
======================================================================
```

### 1.4. Cơ chế Caching & Tối ưu VRAM cho GPU Laptop
- **Disk Cache theo Task/Repo:** Lưu ma trận vector và metadata của từng task vào `cache/embeddings/{instance_id}.npz` (hoặc `.pkl`). Nếu chạy lại hoặc chạy chung giữa `E2B` và `E4B` trên cùng 50 tasks, chỉ cần tính embedding **đúng 1 lần duy nhất cho mỗi task** (tổng cộng 50 lần index), thời gian tra cứu ở lần chạy thứ hai là $<10\text{ms}$.
- **Tách biệt VRAM (Zero Runtime VRAM Conflict):**
  - Bước tiền xử lý (Pre-compute Embeddings cho 50 tasks) được chạy thành một bước độc lập (`scripts/build_embedding_cache.py`) **trước khi** khởi chạy vLLM / `llama.cpp` server cho `E2B`/`E4B`, hoặc chạy trên CPU/GPU tách rời.
  - Nhờ đó, trong suốt quá trình chạy Agent (`M4`), VRAM dành cho LLM không bị giảm đi bất kỳ MB nào, tránh hoàn toàn lỗi `shared KV pool 500`.

---

## Phần 2: Danh sách công việc cho Coding Agent (Implementation Checklist)

### Bước 1: Hoàn thiện các bản vá nền tảng của Harness (Nếu chưa merge hết)
Đảm bảo các bản vá thực thi từ đợt chẩn đoán trước đã được áp dụng đồng nhất cho `M4`:
- [ ] **`resolve_repo_path`:** Tự động khử tiền tố lặp (như `psf/requests/...` $\rightarrow$ `requests/...`, `sympy/sympy/...` $\rightarrow$ `sympy/...`).
- [ ] **`apply_edit` Normalization:** Chuẩn hóa `\r\n` và `\\r\\n` thành `\n`; hỗ trợ khớp không phân biệt khoảng trắng cuối dòng (`.rstrip()`).
- [ ] **Pre-Compaction Gate (`lock_targets`):** Từ chối `lock_targets` nếu danh sách `symbols` không trích xuất được mã nguồn hợp lệ (tránh hiện tượng nén rỗng khiến model phải tìm kiếm lại ở Pha 2).
- [ ] **Hard Loop-Breaker:** Cảnh báo ngay nếu turn $t$ lặp lại y hệt `(tool_name, args)` của turn $t-1$.

### Bước 2: Xây dựng Module `harness/seeding.py` & Pre-compute Cache
- [ ] Viết script `scripts/build_embedding_cache.py`:
  - Duyệt qua 50 tasks trong tập `SWE-bench Verified` đang dùng.
  - Với mỗi worktree, dùng bộ parser `tree-sitter` trích xuất danh sách toàn bộ AST symbols.
  - Chạy mô hình **Embedding Gemma** (kết hợp BM25 / Identifier matching) đối chiếu với `problem_statement` của task đó.
  - Lưu Top-20 symbols kèm điểm số, chữ ký hàm, dòng đầu docstring và thời gian thực thi (`indexing_time_s`, `query_time_ms`) vào `cache/seeding/{instance_id}.json`.
- [ ] Viết hàm kiểm tra nhanh **Recall@K của bước Seeding** ngay sau khi tạo cache:
  - Đối chiếu Top-5 và Top-8 symbols trong `cache/seeding/{instance_id}.json` với file và hàm thực tế nằm trong **Gold Patch** của SWE-bench.
  - In ra chỉ số: `File Hit Rate @ 5`, `File Hit Rate @ 8`, và `Symbol Hit Rate @ 8` (số liệu này sẽ dùng làm một bảng riêng rất giá trị trong Paper).

### Bước 3: Tích hợp biến thể `M4` vào `harness/agent.py`
- [ ] Thêm cấu hình `M4` vào danh sách variants (`B0`, `M1`, `M2`, `M3`, `M4`).
- [ ] Khi `variant == "M4"`:
  - Kế thừa toàn bộ cấu hình của `M3` (AST-Nav + AST-Edit + Episodic Compaction).
  - Tại Turn 0, đọc Top-8 symbols từ `cache/seeding/{instance_id}.json` và chèn khối `[AUTO-RETRIEVED CANDIDATE SYMBOLS]` vào user prompt đầu tiên.
  - Ghi log trường `"seeded_symbols": [...]` vào bản ghi đầu tiên của file `.jsonl` để phục vụ phân tích.

### Bước 4: Chạy thực nghiệm & Đánh giá (`E2B` và `E4B`)
- [ ] Chạy batch 50 tasks cho `gemma-4-E2B` ở cấu hình `M4`.
- [ ] Chạy batch 50 tasks cho `gemma-4-E4B` ở cấu hình `M4`.
- [ ] Chạy SWE-bench Docker evaluator cho toàn bộ các bản vá sinh ra ở `M4` và cập nhật vào bảng tổng hợp Bước 5.

---

## Phần 3: Các chỉ số cần xuất ra sau khi chạy xong (Reporting Spec)

Cập nhật script tổng hợp để xuất ra 3 bảng so sánh phục vụ trực tiếp cho bài báo:

1. **Bảng 1 — Hiệu năng định vị của Embedding Gemma ở Turn 0 (Zero-Turn Localization Recall):**
   - `Top-1 / Top-5 / Top-8 File Recall (%)` và `Symbol Recall (%)` trên 50 tasks.
   - Thời gian trung bình để index 1 repo và dung lượng VRAM/RAM tiêu thụ của Embedding Gemma.

2. **Bảng 2 — So sánh hành vi Pha 1 giữa `M3` và `M4`:**
   - Số turns trung bình trước khi gọi `lock_targets` (kỳ vọng giảm từ `5.8 turns` ở `M3` xuống `2-3 turns` ở `M4`).
   - Tỷ lệ lỗi `SYMBOL_NOT_FOUND` (kỳ vọng giảm mạnh nhờ model copy trực tiếp ID từ danh sách gợi ý ở Turn 0).
   - Tỷ lệ gọi lại tool tìm kiếm sau khi Compaction (`Post-compaction nav calls`).

3. **Bảng 3 — Bảng tổng hợp Bước 5 mở rộng (`B0` $\rightarrow$ `M4` cho cả `E2B` và `E4B`):**
   - Đầy đủ các cột: `Resolved`, `Resolve rate (%)`, `Edited`, `Real edits`, `Malformed`, `Mean tok`, `Finish`, và danh sách `Resolved IDs`.