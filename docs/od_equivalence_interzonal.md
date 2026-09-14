# Direct-OD và Partial-OD sau khi chuyển sang liên vùng

Hai script mặc định dùng `results/interzonal_only/data`, checkpoint trong
`results/interzonal_only/artifacts/checkpoints`, và lưu kết quả vào hai thư mục
`direct_od_equivalence_v1` và `partial_od_equivalence_v2` dưới cùng thư mục artifacts.
Có thể chỉ định `--data-root`, `--checkpoint-dir`, `--output-dir`.

Trước khi chạy, code kiểm tra inventory dữ liệu theo `interzonal_protocol.json`
và đủ 15 checkpoint GNN. Mỗi checkpoint phải có seed, fold, backbone, hash split,
`training_support=positive_interzonal` và `training_data_sha256` khớp hash protocol.
Checkpoint lịch sử thiếu metadata bị từ chối. Không sửa metadata chỉ vì checkpoint
nằm trong thư mục có tên interzonal. Cần xác minh lịch sử huấn luyện trước khi xây
cơ chế chuyển đổi có bằng chứng hoặc tạo checkpoint mới có provenance đầy đủ.

`source_manifest.json` khóa đường dẫn, hash dữ liệu và hash checkpoint của cả bộ.
Thông tin này cũng nằm trong signature chọn lambda, progress và manifest từng fold.
Resume và tổng hợp phải khớp nguồn. Thư mục cũ không có source manifest bị từ chối,
không tự động tiếp nhận. Fold đã có dữ liệu yêu cầu `--resume` hoặc thư mục mới.
Không sao chép các artifact cũ vào thư mục chạy mới.

Sau khi nguồn checkpoint đã được xác minh, chạy:

```sh
.venv/bin/python src/experiment/run_direct_od_equivalence_v1.py --resume
.venv/bin/python src/experiment/run_partial_od_equivalence_v2.py --resume
```

Các lệnh trên chạy thực nghiệm đầy đủ. Việc sửa code không tự chạy các lệnh này.

---

> [!NOTE]
> **Trạng thái bài báo (2026-09-14):**
> Hai thực nghiệm này (`direct_od_equivalence_v1` và `partial_od_equivalence_v2`) hiện tại **không dùng trực tiếp cho các bảng/hình chính trong bài báo**.
> Chúng được lưu trữ như một phân tích kiểm toán / độ nhạy bổ trợ (Audit / Information Equivalence Analysis / Phân tích D) để sử dụng khi trả lời phản biện (reviewer rebuttal) hoặc đưa vào Phụ lục mở rộng về tính tương đương thông tin giữa $Y_D$ và quan sát OD trực tiếp.
> Tạm hoãn chạy hai thực nghiệm này cho đến khi có yêu cầu cụ thể và sau khi nguồn checkpoint liên vùng đã được huấn luyện / xác minh provenance đầy đủ.

