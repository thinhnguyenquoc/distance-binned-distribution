# Rà soát lý thuyết, công thức và giao thức liên vùng

Phạm vi: đối chiếu mục 3 và phụ lục phương pháp với mã nguồn. Không thay số liệu thực nghiệm, không khởi chạy huấn luyện hoặc placebo.

## Các thành phần đã xác nhận

| Thành phần | Đánh giá và điều kiện |
|---|---|
| Tập hỗ trợ | Omega gồm i khác j, khoảng cách dương và nhãn nguyên dương. Điều kiện i khác j một mình chưa loại cặp khác vùng có cùng tọa độ tâm. |
| Gravity | exp(G) là hệ số quy mô. Trong không gian log, G là hệ số chặn và -alpha là hệ số của log khoảng cách. Dân số và khoảng cách có ngưỡng chặn trong code. |
| NB và ZTNB | Công thức Gamma, xác suất tại 0 và chuẩn hóa trên luồng dương đúng. mu là trung bình nền, phi là tham số kích thước/phân tán với phương sai mu+mu²/phi. |
| Miền giá trị | t=0 là giá trị luồng, i=j là danh tính cặp vùng. Cắt cụt NB tại 0 không thay thế thao tác loại OD nội vùng. |
| Mất mát | Âm log hợp lý trung bình trên Omega của từng thành phố nguồn. Neural học mu theo cặp và phi dùng chung. |
| Dự báo | Kỳ vọng ZTNB bằng mu/(1-p0). Dự báo liên tục dương dù nhãn nguyên dương. Code dùng epsilon và ngưỡng chặn nên chỉ xấp xỉ công thức lý tưởng ở sát giới hạn số học. |
| Y_D | Tỷ trọng khối lượng luồng, không phải tỷ lệ số cặp. Các biên lấy từ khoảng cách của tập nguồn với mỗi cặp có trọng số bằng nhau. |
| Nhóm rỗng | Tập hoạt động dựa trên sự tồn tại của cặp trong Omega. Nhóm không có cặp không có hệ số cần áp dụng. |
| Hiệu chỉnh | q=1 cho tỷ số tỷ trọng mục tiêu/dự báo. q tổng quát cần hệ số chuẩn hóa Z. Tổng khối lượng được bảo toàn. |
| Thứ hạng | Hệ số dương bảo toàn tỷ lệ và thứ hạng trong từng nhóm. Không bảo toàn thứ hạng giữa các nhóm. Target bằng 0 ở nhóm hoạt động làm mất tính dương khi q>0. |
| CPC | CPC = 1 - L1/(tổng thật+tổng dự báo). Bảo toàn tổng dự báo không bảo đảm giảm L1. Oracle không phải cận trên CPC. |
| Dose matching | Center log-ratio và ghép RMS đúng cho cùng tập hoạt động. Hoán vị log-ratio bảo toàn RMS, không phải hoán vị trực tiếp histogram. |
| Nhiễu | Exponential tilting tạo tỷ trọng dương và chuẩn hóa tổng bằng 1. TV là khoảng cách giữa phân phối, không phải tỷ lệ chuyến đi bị đo sai. |
| County | Phân hoạch theo county nguồn là phân hoạch các cặp. Cộng lại dự báo các nhóm bảo toàn tổng thành phố nếu từng nhóm bảo toàn tổng. Không được xem county là dữ liệu OD độc lập. |

## Bổ sung vào paper

Đã thống nhất ký hiệu mathcal A_c trong các công thức, làm rõ phân vị trùng và số nhóm thực tế, điều kiện mẫu số dương, trường hợp target có tỷ trọng 0, và q=0. Đã bổ sung chứng minh CPC theo L1 để giới hạn đúng ý nghĩa oracle.

Đã bổ sung phân biệt giả thuyết một phía/hai phía, điều kiện diễn giải Wilcoxon, và công thức kiểm định tương quan từng phần với bậc tự do n-k-2. R² đa biến không phải phần phương sai giải thích của riêng d_pre và không xác lập quan hệ nhân quả.

## Những chỗ code còn chưa khớp hoàn toàn, cần xử lý trước khi tái xuất kết quả

1. `run_unified_placebo.py` vẫn xác định nhóm hoạt động bằng ngưỡng tỷ trọng target. Cần dùng trực tiếp các cặp và biên nhóm, nhất là với nhóm có tỷ trọng rất nhỏ. Với chỉ một nhóm hoạt động, tập hoán vị khác đồng nhất rỗng và cần xử lý suy biến rõ ràng thay vì lấy trung bình danh sách rỗng.
2. Hàm bootstrap placebo hiện lấy mẫu gộp thành phố, chưa phân tầng theo fold. Bảng cũng dùng kiểm định một phía cho hàng target trong khi paper quy định hai phía cho hiệu ứng chính. Cần tách thống kê đối chứng so với baseline và target so với đối chứng.
3. `audit_dpre_mechanism.py` dùng p-value Pearson trên phần dư mà chưa trừ số biến kiểm soát. Cần tính Student với n-k-2 và dùng hàm survival để tránh p-value bằng 0 do phép trừ 1-CDF.
4. Báo cáo placebo còn nhãn Upper Bound và diễn giải viết cố định. Đây không phải hệ quả của toán tử và không nên đưa vào kết luận khoa học.
5. Một số đường chạy phân tích dùng expm1(log1p(distance)) thay vì khoảng cách gốc. Cần thống nhất khoảng cách và thứ tự cặp giữa binning, Y_D và hiệu chỉnh trước khi chứng nhận các đầu ra cùng giao thức.
6. Các công thức donor/hoán vị lý tưởng đúng khi tỷ trọng dương. Code dùng ngưỡng nhỏ cho log và dự báo, nên kiểm tra đối chiếu với toán tử chuẩn cần bao phủ các đối chứng và trường hợp suy biến, không chỉ phân phối target.

## Kiểm tra công thức

`tests/test_paper_formula_contract.py`: 4 kiểm tra đạt. Đối chiếu NB/ZTNB với scipy ở nhiều mu/phi, kiểm tra chuẩn hóa và kỳ vọng, kiểm tra q=0/0.2/0.5/1 với nhóm rỗng, kiểm tra target bằng 0, và phản ví dụ oracle khớp tỷ trọng nhưng CPC giảm.

Kết luận: lõi toán học đúng với các giả định đã nêu. Chưa thể chứng nhận toàn bộ pipeline thống kê hiện có là khớp hoàn toàn với paper cho đến khi xử lý các điểm triển khai trên. Giữ nguyên kết quả hiện có trong thời gian này.
