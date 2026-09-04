# Improving Zero-Shot Origin–Destination Flow Intensity Reconstruction via Target-City Distance-Binned Mobility Distributions

*(Tiếng Việt: **Cải thiện tái tạo cường độ luồng OD zero-shot bằng phân phối di chuyển theo khoảng cách của thành phố mục tiêu**)*

---

## Abstract

Origin–destination (OD) flow matrices are important inputs for transportation analytics and urban planning, yet granular target-city flow intensities are often difficult to obtain. Cross-city zero-shot models that leverage urban context and geographic distance can reconstruct mobility flows without observing target-city OD data. This study investigates whether a low-dimensional oracle aggregate observation—the target city's distance-binned mobility distribution—can improve zero-shot OD intensity reconstruction beyond a frozen neural model. In the canonical experiment, this distribution is deterministically derived from the target city's reference OD flows and only provides the proportion of total trip volume falling within each distance interval, without revealing the intensity of any individual OD pair. The trained neural backbone and model parameters remain fixed, while the aggregate distribution is used solely at inference time to reallocate predicted flow volume across distance intervals on the known positive interzonal support. We evaluate the framework under city-level 5-fold cross-validation across 50 U.S. metropolitan datasets. City-level calibration produces a small but relatively consistent improvement, increasing the mean Common Part of Commuters (CPC) by 0.00354, with a 95% bootstrap confidence interval of the improvement spanning from 0.0026 to 0.0045, and improving performance in 45 of the 50 cities. The improvement diminishes as the resolution and quality of the observed distribution decline. The conclusions apply only to intensity reconstruction on known positive interzonal support and do not extend to link prediction or identification of zero-flow pairs. Overall, the target city's distance-binned mobility distribution provides a low-dimensional aggregate signal that yields modest, consistent improvements for a frozen cross-city model in this support-conditioned benchmark.

**Keywords:** origin–destination matrix; OD intensity reconstruction; distance-binned distribution; zero-shot; cross-city transfer learning; aggregate observations; spatial mobility.

---

## Tóm tắt

Ma trận nguồn–đích (origin–destination, OD) là đầu vào quan trọng cho phân tích giao thông và quy hoạch đô thị, nhưng dữ liệu chi tiết về cường độ luồng OD của thành phố mục tiêu thường khó thu thập. Các nghiên cứu sử dụng ngữ cảnh đô thị và khoảng cách địa lý đã phát triển các baseline cross-city zero-shot có khả năng dự báo luồng di chuyển mà không sử dụng dữ liệu quan sát về cường độ OD của thành phố mục tiêu. Nghiên cứu này xem xét liệu phân phối di chuyển theo khoảng cách của thành phố mục tiêu có thể cải thiện việc tái tạo cường độ luồng OD liên vùng trên tập hỗ trợ dương đã biết thông qua hiệu chỉnh kết quả của một baseline zero-shot hay không.

Phân phối di chuyển này được tổng hợp từ dữ liệu quan sát luồng của chính thành phố mục tiêu và chỉ cung cấp tỷ trọng khối lượng luồng theo các khoảng khoảng cách, không cung cấp cường độ của từng cặp OD cụ thể. Trong thí nghiệm chính, phương pháp được đánh giá bằng quy trình kiểm định chéo 5-fold trên 50 vùng đô thị của Hoa Kỳ. Hiệu chỉnh ở cấp thành phố tạo ra mức cải thiện nhỏ nhưng nhất quán: CPC trung bình tăng 0.00354 (CI 95%: $[+0{.}0026; +0{.}0045]$), với kết quả có cải thiện cho baseline zero-shot tại 45 thành phố. Đồng thời, mức cải thiện cũng giảm khi độ phân giải hoặc chất lượng của phân phối quan sát dần suy giảm.

Nghiên cứu giới hạn phạm vi tạp trung đánh giá ở các cặp OD đã biết có luồng di chuyển. Các cặp OD chưa được quan sát chưa có số liệu di chuyển không được đưa vào các phép đánh giá.

**Từ khóa:** ma trận nguồn–đích; tái tạo cường độ OD; phân phối di chuyển theo khoảng cách; zero-shot; học chuyển giao giữa các thành phố; quan sát tổng hợp; di chuyển không gian.
