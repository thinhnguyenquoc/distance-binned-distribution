# Related Work & Exploratory Experiments Archive

Thư mục này chỉ lưu các pilot, legacy runner và biến thể không còn thuộc
canonical paper pipeline. Các runner hoặc contract test được gọi bởi paper
provenance, research contract hoặc certification phải nằm trong `src/experiment/`
và `tests/`.

---

## Danh mục các thực nghiệm đã chuyển:

### 1. `src_experiment/`
* **Pilot & Redundant Experiments**:
  * `run_convergence_pilot.py`: Thí điểm đo tốc độ hội tụ epoch.
  * `run_unified_placebo.py`: Phiên bản archive (runner canonical chính thức đặt tại `src/experiment/run_unified_placebo.py`).
* **Historical analysis helpers**:
  * `audit_finite_sample_yd_v1.py`: Kiểm toán mẫu hữu hạn đời cũ.
  * `compute_qstar.py`: Helper tính tỷ lệ tương đương quan sát cho các kết quả archive.

### 2. `legacy_and_pilots/`
* `legacy/`: Mã nguồn huấn luyện `run_e1.py` đời đầu.
* `audit_temp.py`: Script kiểm toán tạm thời tại thư mục gốc.
