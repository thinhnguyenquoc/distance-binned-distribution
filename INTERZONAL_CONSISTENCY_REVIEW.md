# Kiểm tra tính nhất quán của thực nghiệm liên vùng

Giao thức mới dùng các cặp có i khác j, khoảng cách dương và luồng dương cho cả GNN, MLP và Gravity. Không thay số liệu cũ trong paper bằng kết quả chạy thử.

## Kết quả kiểm tra dữ liệu và phép hiệu chỉnh

Đã đối chiếu đủ 50 thành phố giữa dữ liệu gốc và bản đã lọc. Có 6.065.339 cặp liên vùng dương. Trên tập này, chỉ số nguồn và đích, lưu lượng, khoảng cách, đặc trưng nút và tọa độ đều giữ nguyên. Các thành phố vẫn theo manifest 5 fold đã khóa, mỗi fold có 35 thành phố huấn luyện, 5 validation và 10 kiểm tra.

Script `scripts/audit_interzonal_consistency.py` kiểm tra Y_D bằng cách cộng lưu lượng trực tiếp theo nhóm ở cả 50 thành phố kiểm tra. Đồng thời kiểm tra q=0 giữ nguyên dự báo, q=1 khớp phân phối mục tiêu và bảo toàn tổng lưu lượng dự báo. Báo cáo máy đọc được nằm tại `results/interzonal_only/consistency_audit.json`.

Việc loại OD nội vùng không loại tract hoặc cạnh tự nối của đồ thị địa lý. Đồ thị vẫn được xây dựng từ địa lý, không từ nhãn luồng OD.

## Các điểm đã sửa trong lần rà soát

Checkpoint neural mới có `training_support` và `training_data_sha256`. Đường chạy mới từ chối checkpoint không khớp và kiểm tra trực tiếp tập nhãn trước huấn luyện. Đã chạy thử huấn luyện và lưu/nạp checkpoint cho cả GNN và MLP trên Raleigh, đồng thời xác nhận nhãn nội vùng và dấu dữ liệu không khớp bị từ chối.

Ở đường chạy mới, GNN/MLP dùng khoảng cách gốc để tính Y_D và hiệu chỉnh, giống Gravity và bước xác định biên nhóm. Đường chạy cũ từng khôi phục khoảng cách bằng expm1(log1p(d)), có thể khác vài chữ số cuối do float32 và làm đổi nhóm khi sát biên. Tùy chọn mới giữ hành vi lịch sử của đường chạy cũ nhưng thống nhất đường chạy liên vùng.

Sau khi đủ kết quả của các mô hình được yêu cầu, `backbone_comparison.json` áp dụng cùng thống kê cấp thành phố: trung bình CPC trước và sau, trung bình chênh lệch, bootstrap 10.000 lượt phân tầng theo fold, Wilcoxon hai phía và số thành phố cải thiện. Các giá trị p là giá trị gốc cho phép so sánh trước/sau trong từng mô hình, không phải kiểm định so sánh kiến trúc. Bộ tổng hợp từ chối thành phố thiếu, trùng hoặc sai fold. Neural được lấy trung bình qua seeds trước khi phân tích cấp thành phố.

## Phạm vi chưa hoàn tất

Chưa huấn luyện đầy đủ 5 fold và 3 seeds cho hai mô hình neural. Chưa thể kết luận mức tăng CPC của giao thức mới hoặc cập nhật các bảng, hình và kết luận định lượng trong paper.

Script chính mới chạy hiệu chỉnh oracle K=8 và tổng hợp so sánh ba mô hình. Nó không tự chạy toàn bộ placebo, độ nhạy K, nhiễu và phân tích cơ chế của paper. Các script lịch sử có thể mặc định đọc checkpoint/kết quả cũ. Không dùng chúng để tạo kết luận cho giao thức mới nếu chưa chuyển toàn bộ đầu vào và kiểm tra nguồn gốc.

Đã bổ sung tham số đường dẫn cho hai phân tích K và nhiễu. Sau khi có đủ checkpoint mới, có thể chạy riêng:

```bash
MPLBACKEND=Agg .venv/bin/python -m src.experiment.run_k_sensitivity_v1 --data-root results/interzonal_only/data --checkpoint-dir results/interzonal_only/artifacts/checkpoints --output-dir results/interzonal_only/k_sensitivity
MPLBACKEND=Agg .venv/bin/python -m src.experiment.run_noise_robustness --data-root results/interzonal_only/data --checkpoint-dir results/interzonal_only/artifacts/checkpoints --output-dir results/interzonal_only/noise
```

Các phân tích này chưa được chạy đầy đủ với checkpoint mới. Placebo và phân tích cơ chế vẫn cần chuyển đường dẫn và kiểm tra riêng trước khi tái tạo toàn bộ paper.
