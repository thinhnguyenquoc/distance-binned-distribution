# E1: Methodological Framing & Statistical Defense Guide

> **Mục đích**: Tài liệu chuẩn hóa lý thuyết, công thức toán học và khung diễn đạt phương pháp luận cho thực nghiệm **E1 (Oracle Aggregated-Distance Existence Test)** nhằm đảm bảo tính chặt chẽ, minh bạch và chống phản biện khoa học (peer-review defense).

---

## 1. Bản Chất Hàm Mục Tiêu: City-Balanced Objective

### 1.1 Công Thức Toán Học
Hàm mất mát huấn luyện cho một fold $f$ gồm $C = 35$ thành phố huấn luyện:

$$\mathcal{L}_{\text{train}}(\theta) = \frac{1}{C} \sum_{c=1}^{C} \left[ \frac{1}{|\Omega_c^+|} \sum_{(i,j) \in \Omega_c^+} \ell_{ij}^{(c)}(\theta) \right]$$

trong đó:
- $\Omega_c^+ = \{(i,j) \in \Omega_c : i \neq j, D_{ij} > 0\}$ là tập các cặp liên vùng (interzonal candidate pairs).
- $\ell_{ij}^{(c)}(\theta) = -\log P_{\text{ZTNB}}(T_{ij}; \mu_{ij}(\theta), \phi)$ là Zero-Truncated Negative Binomial negative log-likelihood.

### 1.2 Phân Biệt City-Balanced vs Pair-Balanced
- **City-Balanced (Macro-Averaged)**: Mỗi thành phố nhận hệ số ngoài $\frac{1}{C}$ ngang nhau trong hàm mục tiêu lý thuyết.
- **Pair-Balanced (Micro-Averaged)**:
  $$\mathcal{L}_{\text{pair}}(\theta) = \frac{1}{\sum_{c} |\Omega_c^+|} \sum_{c=1}^C \sum_{(i,j) \in \Omega_c^+} \ell_{ij}^{(c)}(\theta)$$
  Trong công thức pair-balanced, các đại đô thị có số cặp lớn ($|\Omega_c^+| \approx N_c(N_c-1)$ lên tới hàng triệu cặp) sẽ áp đảo hoàn toàn các đô thị quy mô nhỏ.

### 1.3 Quy Tắc Diễn Đạt Chính Xác (Airtight Phrasing)
> [!IMPORTANT]
> - **NÊN VIẾT**: *"Each city receives an equal coefficient ($\frac{1}{C}$) in the macro-averaged training objective."*
> - **KHÔNG NÊN VIẾT**: *"Every city has exactly equal influence on the learned parameters."*

**Lý do**: Trong thực tế tối ưu hóa:
1. Các thành phố có phân phối dòng di chuyển khác nhau, phương sai khác nhau, và gradient norm khác nhau.
2. Quá trình huấn luyện thực thi theo cơ chế Online/Stochastic Gradient Descent từng thành phố (per-city step kèm gradient clipping $\le 5.0$).

---

## 2. Thống Kê Suy Luận: Fold-Stratified Bootstrap & Giới Hạn Phụ Thuộc

### 2.1 Bản Chất Của Fold-Stratified Bootstrap
Quy trình tái lấy mẫu 10,000 lần (10,000 resamples):
1. Với mỗi fold $f \in \{1, \dots, 5\}$, lấy mẫu có hoàn lại (with replacement) đúng 10 đô thị từ 10 test cities của fold đó.
2. Tổng hợp 50 đô thị giả định để tính bootstrap mean của $\Delta_c^{\text{target}}$, $\Delta_c^{\text{wrong}}$, và $\Delta_c^{\text{specificity}}$.

### 2.2 Các Chiều Phụ Thuộc (Remaining Covariance & Dependencies)
Fold-stratified bootstrap bảo toàn cơ cấu phân tầng (strata ratio) và phản ánh tính dị biệt giữa các fold, nhưng **chưa khử hoàn toàn**:
1. **Shared-Backbone Covariance**: 10 đô thị trong cùng một test fold được đánh giá dựa trên cùng một trọng số mô hình đã đóng băng $\theta^{(f)}$.
2. **Overlapping-Training-Set Dependence**: Giữa hai fold $f_1 \neq f_2$, tập train 35 đô thị có độ trùng lặp $\frac{30}{35} \approx 85.7\%$.

### 2.3 Quy Tắc Diễn Đạt Chính Xác (Airtight Phrasing)
> [!IMPORTANT]
> - **NÊN VIẾT**: *"Fold-stratified bootstrap preserves the cross-validation fold structure when estimating uncertainty across out-of-fold city-level effects, though it does not fully model shared-backbone and overlapping-training-set covariance."*
> - **KHÔNG NÊN VIẾT**: *"Fold-stratified bootstrap eliminates all statistical dependencies between observations."*

---

## 3. Bin Edges: Pair-Weighted vs City-Balanced

### 3.1 Thiết Kế Hiện Tại: Pair-Weighted Quantiles ($F_{\text{pair}}$)
Hệ phân vị $K_{\text{move}} = 8$ bins được trích xuất từ toàn bộ candidate pairs của 35 training cities:

$$F_{\text{pair}}(d) = \frac{\sum_{c \in \text{Train}} \sum_{(i,j) \in \Omega_c^+} \mathbf{1}(D_{ij} \le d)}{\sum_{c \in \text{Train}} |\Omega_c^+|}$$

- **Ưu điểm**: Đảm bảo mỗi khoảng cách địa lý có mật độ cặp quan sát đồng đều trên toàn bộ không gian dữ liệu huấn luyện; tránh tình trạng các bin bị thoái hóa hoặc thiếu mẫu.
- **Đặc điểm**: Thành phố lớn đóng góp nhiều cặp hơn vào việc định hình biên khoảng cách.

### 3.2 Phương Án Đối Sánh: City-Balanced Quantiles ($F_{\text{city}}$)
$$F_{\text{city}}(d) = \frac{1}{C} \sum_{c \in \text{Train}} \left[ \frac{1}{|\Omega_c^+|} \sum_{(i,j) \in \Omega_c^+} \mathbf{1}(D_{ij} \le d) \right]$$

- **Định vị phương pháp**: Giữ $F_{\text{pair}}$ là **Primary Protocol**, và có thể sử dụng $F_{\text{city}}$ như một **Sensitivity Analysis** bổ trợ nếu cần kiểm chứng độ vững.

---

## 4. Phân Phối $Y_D$: Trip-Weighted (TLD) vs Pair Count

Vector $Y_D$ của mỗi thành phố $c$ trên $K_{\text{move}} = 8$ bins:

$$Y_{D,k} = \frac{\sum_{(i,j) \in \Omega_{c,k}^+} T_{ij}}{\sum_{(i,j) \in \Omega_c^+} \sum T_{ij}}$$

- **Bản chất khoa học**: $Y_D$ bắt buộc phải là **Trip-Weighted** vì đại lượng nghiên cứu là **Hàm phân phối chiều dài chuyến đi thực tế (Empirical Trip Length Distribution - TLD)**.
- Nếu gộp theo số lượng cặp (Pair-Weighted), nó chỉ phản ánh hình học đô thị (Urban Geometry), mất đi thông tin về quy luật di chuyển thực tế.

---

## 5. Tính Đối Xứng & Công Bằng Trong Hiệu Chuẩn (Calibration Symmetry)

Toán tử hiệu chuẩn Closed-Form ($q = 1.0$):

$$T_{ij}^{(\text{cal})} = T_{ij}^{(0)} \cdot \left[ \frac{Y_{D,k} / \hat{Y}_k}{\sum_{m} \hat{Y}_m (Y_{D,m} / \hat{Y}_m)} \right]$$

| Điều kiện kiểm định | Đối tượng thông tin | Toán tử | Bin Edges | Ngưỡng sai số dung sai | Siêu tham số điều chỉnh |
|---|---|:---:|:---:|:---:|:---:|
| **Condition B (Treatment)** | $Y_{D,c}^{\text{target}}$ | Closed-form ($q=1.0$) | $\mathcal{B}_f$ | $10^{-5}$ | *Không có (None)* |
| **Condition C (Placebo)** | $\bar{Y}_{D,d}^{\text{wrong}}$ (9 donors) | Closed-form ($q=1.0$) | $\mathcal{B}_f$ | $10^{-5}$ | *Không có (None)* |

> **Khẳng định công bằng**: Hoàn toàn không có bất kỳ bước tinh chỉnh (hyperparameter tuning) hay thiên vị nào cho Target condition so với Wrong Donor Placebo.

---

## 6. Phát Biểu Phương Pháp Luận Tiêu Chuẩn (Camera-Ready Formal Statement)

Trích đoạn mẫu chuẩn hóa cho phần *Methodology / Statistical Analysis* trong bài báo:

```markdown
> "E1 adopts a city-balanced training objective and city-level macro-averaging for model 
> selection, validation, and out-of-fold evaluation. Distance-bin boundaries (K_move = 8) 
> are constructed from pair-weighted training distances, whereas each city's distance 
> distribution (Y_D) is trip-weighted by definition to represent the empirical trip length 
> distribution. Model adaptation via closed-form calibration operates symmetrically across 
> treatment and multi-donor placebo conditions under identical numerical tolerances. 
> While fold-stratified bootstrap resampling preserves the stratified fold composition across 
> the five cross-validation folds, it does not fully eliminate covariance arising from 
> shared backbone parameters within folds or overlapping training sets across folds."
```

---

## 7. Bộ Câu Hỏi & Trả Lời Phản Biện (Peer-Review Defense Q&A)

### Q1: Tại sao không tính loss trung bình trên toàn bộ cặp (Pair-Weighted Training)?
**Trả lời**: Nếu tối ưu hóa theo pair-weighted, các siêu đô thị lớn (như Chicago, Houston) sẽ chiếm $>70\%$ tổng gradient, khiến mô hình overfit vào hình thái đô thị lớn và mất khả năng tổng quát hóa (zero-shot transfer) sang các đô thị vừa và nhỏ. City-balanced objective đảm bảo mọi hình thái đô thị đều có trọng số đại diện ngang nhau trong mục tiêu tối ưu.

### Q2: Tại sao 9 wrong donors lại dùng trung bình cộng thay vì 1 donor ngẫu nhiên?
**Trả lời**: Đánh giá trên trung bình của toàn bộ 9 wrong donors còn lại trong cùng test fold ($\bar{\Delta}_c^{\text{wrong}} = \frac{1}{9}\sum_{d \neq c}\Delta_{c,d}^{\text{wrong}}$) triệt tiêu hoàn toàn phương sai do việc lựa chọn donor chủ quan (arbitrary donor selection), phản ánh chính xác hiệu ứng điều hòa cấu trúc khoảng cách chung (generic distance regularization).

### Q3: Bootstrap có bảo đảm tính độc lập hoàn toàn giữa 50 thành phố không?
**Trả lời**: Không. Các thành phố trong cùng fold chia sẻ cùng một backbone $\theta^{(f)}$, và tập huấn luyện giữa các fold có sự chồng lấn. Fold-stratified bootstrap bảo toàn tỷ lệ phân tầng của cross-validation để ước lượng độ không chắc chắn out-of-fold ở cấp độ đô thị, nhưng không mô hình hóa toàn bộ hiệp phương sai tham số của backbone.
