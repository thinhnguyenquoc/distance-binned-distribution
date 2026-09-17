# Đề xuất nghiên cứu: Mô hình hóa độ phân tán chuyển giao và có cấu trúc không gian cho luồng OD (Transferable & Spatially Structured Dispersion for OD Flows)

**Trạng thái**: Đã lưu trữ ý tưởng nghiên cứu độc lập (Post-current paper proposal)  
**Ngày khởi tạo**: 2026-09-17  
**Dựa trên cơ sở thực nghiệm**: Kiểm định phân phối ZTNB trên 6.065.339 quan sát liên vùng thuộc 50 đô thị (`INTERZONAL_COUNT_DISTRIBUTION_EVIDENCE.md`).

---

## 1. Đánh giá học thuật & Điểm độc đáo (Academic Assessment & Novelty)

### 1.1 Khoảng trống nghiên cứu (Research Gap)
* Hầu hết các mô hình sinh luồng di chuyển đô thị (Gravity, DeepGravity, Urban GNN) chỉ tập trung dự báo **kỳ vọng điểm (point expectation / mean flow)** $\hat{T}_{ij} = \mathbb{E}[T_{ij}]$.
* Khi mô hình hóa xác suất (probabilistic/count modeling), các nghiên cứu thường mặc định dùng Poisson (equidispersed) hoặc Negative Binomial với một tham số phân tán toàn cục $\phi$ duy nhất.
* **Vấn đề thực nghiệm**: Kiểm định trên 50 thành phố cho thấy overdispersion không phải là nhiễu đồng nhất (homogeneous noise):
  * Tỷ số $\operatorname{Var}/\operatorname{Mean}$ thay đổi từ **708 đến 2.648** giữa các thành phố.
  * Giảm mạnh theo khoảng cách: từ **2.288** (<5 km) xuống **58** (>50 km).
  * Tăng vọt theo quy mô luồng: các luồng $\ge 500$ có $\operatorname{Var}/\operatorname{Mean} \approx 1.592$.

### 1.2 Luận điểm đóng góp cốt lõi (Core Contribution)
> **"Dispersion của luồng OD không phải là nhiễu ngẫu nhiên vô hướng, mà mang cấu trúc không gian và liên thành phố có thể học được; cấu trúc này có thể dự đoán zero-shot cho các đô thị chưa từng quan sát chỉ từ đặc trưng đô thị và cự ly, giúp nâng cao vượt trội chất lượng ước lượng bất định (uncertainty estimation) trong dự báo lưu lượng đô thị."**

---

## 2. Câu hỏi nghiên cứu (Research Questions)

1. **RQ1 (Cấu trúc không gian & Bối cảnh)**: Mức độ overdispersion của các luồng liên vùng biến thiên như thế nào theo cự ly ($D_{ij}$), quy mô phát xạ ($O_i, D_j$), mật độ tract và phân cấp đô thị?
2. **RQ2 (Khả năng chuyển giao Zero-Shot)**: Liệu cấu trúc dispersion của một thành phố chưa từng thấy có thể dự báo được mà **hoàn toàn không dùng nhãn OD** của thành phố đó, chỉ dựa trên đặc trưng nút (dân số, việc làm, POI, đường sá) và cự ly?
3. **RQ3 (Giá trị mô hình hóa xác suất)**: Việc mô hình hóa $\phi$ có cấu trúc đem lại cải thiện định lượng như thế nào đối với các chỉ số đánh giá phân phối xác suất (NLL, CRPS, độ bao phủ khoảng tin cậy / Prediction Interval Coverage Probability - PICP, Winkler Score) so với mô hình ZTNB dùng $\phi$ toàn cục?

---

## 3. Hệ thống mô hình đối chứng (Model Hierarchy)

Nghiên cứu so sánh 4 cấp độ tham số hóa độ phân tán $\phi$:

* **Cấp 0 (Baseline hiện tại - Global $\phi$)**:
  $$\phi_{ij} = \phi_{\text{global}} \quad (\text{1 scalar toàn cục cho toàn bộ tập dữ liệu/fold})$$
* **Cấp 1 (City-specific $\phi_c$)**:
  $$\phi_{ij} = \phi_{c} \quad (\text{mỗi thành phố có 1 mức dispersion riêng, học zero-shot qua đặc trưng vĩ mô đô thị } \mathbf{Z}_c)$$
* **Cấp 2 (Distance-parameterized $\phi(d)$)**:
  $$\phi_{ij} = f_{\theta}(D_{ij}) \quad (\text{hàm đơn điệu hoặc MLP cự ly biểu diễn suy giảm overdispersion theo km})$$
* **Cấp 3 (Feature-dependent Neural Dispersion $\phi_{ij}$)**:
  $$\mu_{ij}, \phi_{ij} = g_{\psi}(\mathbf{h}_i, \mathbf{h}_j, D_{ij}) \quad (\text{mạng nơ-ron dự báo đồng thời kỳ vọng } \mu_{ij} \text{ và dispersion } \phi_{ij})$$

---

## 4. Thiết kế thực nghiệm & Giao thức đánh giá (Experimental Protocol)

* **Phân chia dữ liệu**: Giữ nguyên 5-fold cross-city đã khóa (35 train / 5 val / 10 test trên 50 MSAs).
* **Nguyên tắc Zero-Shot nghiêm ngặt**: Trong từng fold, mô hình học bộ tham số dự báo $\hat{\phi}$ từ 35 thành phố train. Khi đánh giá trên 10 thành phố test, hoàn toàn không được nhìn thấy OD labels của thành phố test.
* **Bộ chỉ số đánh giá (Evaluation Metrics)**:
  1. **Negative Log-Likelihood (NLL)**: Đánh giá độ khớp thực sự của phân phối xác suất ZTNB.
  2. **Continuous Ranked Probability Score (CRPS)**: Đo lường chất lượng toàn diện của phân phối tích lũy dự báo.
  3. **Prediction Interval Calibration (PICP & Winkler Score ở 90% và 95%)**: Kiểm tra xem khoảng tin cậy do mô hình sinh ra có đạt đúng độ bao phủ danh định hay bị under-confident / over-confident.
  4. **CPC & RMSE (Point Prediction)**: Kiểm tra xem việc ước lượng chính xác dispersion $\phi_{ij}$ có giúp tinh chỉnh kỳ vọng có điều kiện $\mathbb{E}[T \mid T \ge 1] = \frac{\mu}{1 - P(0;\mu,\phi)}$ và cải thiện dự báo luồng điểm hay không.

---

## 5. Kế hoạch triển khai khi thực hiện (Action Items)

1. Giữ nguyên phạm vi bài báo hiện tại (Distance-Binned Distribution Calibration với ZTNB global $\phi$).
2. Sau khi hoàn thành bài báo hiện tại, khởi động repo/nhánh riêng:
   - Viết module `HeteroskedasticZTNBHead` dự báo song song $(\mu_{ij}, \log \phi_{ij})$.
   - Xây dựng pipeline đo CRPS và Winkler score trên 50 held-out cities.
   - Chạy cross-city evaluation đối chứng giữa 4 cấp độ $\phi$.
