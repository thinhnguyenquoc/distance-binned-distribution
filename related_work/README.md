# Related Work & Exploratory Experiments Archive

Thư mục này lưu trữ toàn bộ các mã nguồn thực nghiệm **không nằm trong phạm vi hoặc không được đưa vào bài báo hiện tại**, nhằm giúp thư mục `src/` và `tests/` của dự án sạch sẽ, dễ dàng đối chiếu mã nguồn cho Paper.

---

## Danh mục các thực nghiệm đã chuyển:

### 1. `src_experiment/`
* **Direct OD / Partial OD Equivalence**:
  * `run_direct_od_equivalence_v1.py`: Thực nghiệm bộc lộ $p\%$ số cặp OD ground-truth để tìm điểm giao cắt tương đương $p_{\text{eq}}$.
  * `run_partial_od_equivalence_v2.py`: Phiên bản v2 của thí nghiệm kiểm tra tương đương OD trực tiếp.
  * `audit_direct_od_v1.py`: Script kiểm toán kết quả cho direct OD.
* **Finite-Sample / Multinomial Sampling Robustness ($q^*$)**:
  * `run_sampling_robustness.py`: Thử nghiệm lấy mẫu đa thức với số lượng mẫu $m \in [10^2, 10^7]$ chuyến đi.
  * `run_finite_sample_yd_robustness.py`: Kiểm tra độ bền vững của phân phối $Y_D$ khi lấy mẫu hữu hạn.
  * `audit_finite_sample_yd_v1.py`: Kiểm toán độ tin cậy mẫu hữu hạn.
  * `compute_qstar.py`: Tính tỷ lệ tương đương quan sát $q^* = m^* / T_{\text{total}}$.
* **Pilot & Redundant Experiments**:
  * `run_convergence_pilot.py`: Thí điểm đo tốc độ hội tụ epoch.
  * `run_unified_placebo.py`: Phiên bản gộp placebo cũ (kết quả bài báo dùng `run_placebo_matched_v2.py`).

### 2. `tests/`
* `test_direct_od_equivalence_v1_contract.py`: Kiểm thử hợp đồng cho Direct OD.
* `test_partial_od_equivalence_v2_contract.py`: Kiểm thử hợp đồng cho Partial OD v2.
* `test_finite_sample_yd_robustness_contract.py`: Kiểm thử hợp đồng cho Finite Sample.

### 3. `legacy_and_pilots/`
* `legacy/`: Mã nguồn huấn luyện `run_e1.py` đời đầu.
* `audit_temp.py`: Script kiểm toán tạm thời tại thư mục gốc.
