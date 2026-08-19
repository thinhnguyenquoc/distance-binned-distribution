Nếu chọn **Missing = unknown**, thiết kế phù hợp nhất là:

> Đánh giá giá trị bổ sung của (Y_D) trong dự báo cường độ OD trên tập các cặp được quan sát, với cùng observation support cho cả (M_0) và (M_1).

Nghiên cứu không còn tuyên bố tái tạo toàn bộ ma trận OD, mà trở thành **support-conditioned held-out-city OD intensity reconstruction**.

## 1. Câu hỏi nghiên cứu mới

### Tiếng Việt

> Phân phối khoảng cách theo khoảng của thành phố mục tiêu có cải thiện dự báo cường độ của các luồng OD được quan sát so với baseline liên thành phố, khi hai điều kiện cùng sử dụng bối cảnh đô thị, khoảng cách địa lý và observation support hay không?

### Tiếng Anh

> Does a target-city observed distance-binned mobility distribution improve observed OD-flow intensity reconstruction beyond a cross-city baseline under the same urban context, pairwise geographic distance, and observation support?

Điểm thay đổi quan trọng là thêm:

[
\text{same observation support}.
]

## 2. Định nghĩa dữ liệu

Với thành phố (c):

[
\Omega_c^{all}={(i,j):i\neq j}.
]

Chia thành:

[
\Omega_c^{all}
==============

\Omega_c^{obs}\cup\Omega_c^{mis},
]

trong đó:

* (\Omega_c^{obs}): pair có giá trị flow được công bố;
* (\Omega_c^{mis}): pair không có giá trị;
* không đặt (T_{ij}=0) trên (\Omega_c^{mis}).

Định nghĩa observation indicator:

[
R_{ij}=
\begin{cases}
1,&(i,j)\in\Omega_c^{obs},\
0,&(i,j)\in\Omega_c^{mis}.
\end{cases}
]

Nghiên cứu chỉ ước lượng:

[
T_{ij}\mid R_{ij}=1.
]

Không đưa ra kết luận định lượng về (T_{ij}) khi (R_{ij}=0).

## 3. Chế độ thông tin ở test city

| Thông tin                            |     (M_0)     |     (M_1)     |
| ------------------------------------ | :-----------: | :-----------: |
| Urban features (X_c)                 |       Có      |       Có      |
| Urban graph (G_c)                    |       Có      |       Có      |
| Pairwise distance (D_{ij})           |       Có      |       Có      |
| Observation support (\Omega_c^{obs}) |       Có      |       Có      |
| Individual flow (T_{ij})             |     Không     |     Không     |
| Target (Y_D^{obs})                   |     Không     |       Có      |
| Ground-truth OD để đánh giá          | Sau inference | Sau inference |

Do cả hai điều kiện đều biết (\Omega_c^{obs}), so sánh giữa (M_0) và (M_1) cô lập đúng đóng góp của (Y_D).

Đây là:

* zero-shot đối với **target OD magnitudes**;
* không phải pure zero-shot đối với **OD support**.

## 4. Tính (Y_D)

Chỉ tính trên các pair được quan sát:

[
Y^{obs}_{D,k}
=============

\frac{
\sum_{(i,j)\in\Omega_c^{obs}}
T_{ij}\mathbf1(D_{ij}\in B_k)
}{
\sum_{(i,j)\in\Omega_c^{obs}}T_{ij}
}.
]

Phải gọi nó là:

> observed-support distance-binned distribution.

Không được gọi là true full-city (Y_D), vì phần (\Omega^{mis}) chưa biết.

## 5. Kiến trúc mô hình

### Baseline (M_0)

Urban GNN sinh embedding vùng:

[
h_i=\operatorname{GNN}(X_c,G_c)_i.
]

Pairwise decoder:

[
\widehat T^{(0)}_{ij}
=====================

f_\theta(h_i,h_j,\log D_{ij},T^{grav}_{ij}),
\qquad (i,j)\in\Omega_c^{obs}.
]

`pair_o_idx` và `pair_d_idx` được phép sử dụng để chọn (h_i,h_j), nhưng:

* không đưa ID dưới dạng biến số vào decoder;
* không dùng trainable city-specific pair embedding;
* không truyền (T_{ij}) target vào model.

### Calibration (M_1)

Từ baseline, tính predicted bin distribution:

[
\widehat Y^{(0)}_{D,k}
======================

\frac{
\sum_{(i,j)\in\Omega_c^{obs}}
\widehat T^{(0)}*{ij}\mathbf1(D*{ij}\in B_k)
}{
\sum_{(i,j)\in\Omega_c^{obs}}\widehat T^{(0)}_{ij}
}.
]

Hiệu chỉnh:

[
\widehat T^{(1)}_{ij}
=====================

\widehat T^{(0)}*{ij}
\left(
\frac{Y^{obs}*{D,k}}
{\widehat Y^{(0)}*{D,k}}
\right)^q,
\qquad D*{ij}\in B_k.
]

Chính sách chọn (q) phải:

* được cố định từ validation cities; hoặc
* chỉ sử dụng sai lệch (Y_D), không sử dụng test CPC hay individual test OD.

## 6. Hàm loss

Nếu tất cả observed pairs đều có flow dương, ZTNB phù hợp:

[
T_{ij}\mid R_{ij}=1,T_{ij}>0
\sim
\operatorname{ZTNB}(\mu_{ij},\phi).
]


Không đưa (\Omega^{mis}) vào loss như negative examples.

## 7. Protocol 5-fold

Giữ cấu trúc:

* 35 train cities;
* 5 validation cities;
* 10 test cities;
* Fold 1 exploratory;
* Folds 2–5 confirmatory.

Tại mỗi test city:

1. Đọc (X_c,G_c,D_{ij}) và (\Omega_c^{obs}).
2. Tạm khóa toàn bộ individual (T_{ij}).
3. Frozen backbone sinh (M_0).
4. Cung cấp (Y_D^{obs}) và sinh (M_1).
5. Sinh thêm wrong-donor placebo.
6. Lưu predictions.
7. Chỉ sau đó mở (T_{ij}) để tính metrics.

## 8. Estimand chính

Với từng thành phố:

[
\Delta_c^{obs}
==============

CPC_{\Omega^{obs}}
\left(\widehat T_c^{(1)},T_c\right)
-----------------------------------

CPC_{\Omega^{obs}}
\left(\widehat T_c^{(0)},T_c\right).
]

Estimand confirmatory:

[
\Delta_{\mathrm{primary}}
=========================

\frac{1}{40}
\sum_{c\in\text{Folds 2--5}}
\Delta_c^{obs}.
]

Giả thuyết:

[
H_0:E\leq0,
\qquad
H_1:E>0.
]

Primary statistics:

* mean (\Delta^{obs});
* fold-stratified bootstrap 95% CI;
* one-sided paired Wilcoxon hoặc permutation test;
* `Target-over-M0 win rate`.

## 9. Specificity là kết quả phụ

[
\Delta_{\mathrm{spec},c}
========================

## CPC(M_{1,\mathrm{target}},T)

\frac1J\sum_{d=1}^{J}
CPC(M_{1,\mathrm{donor}_d},T).
]

Specificity trả lời liệu (Y_D) có mang đặc trưng thành phố hay không. Nó không thay thế (\Delta_c^{obs}=M_1-M_0).

Thứ tự báo cáo:

1. (M_1-M_0): estimand chính;
2. Target-over-(M_0) win rate;
3. Target-versus-placebo specificity;
4. Placebo degradation.

## 10. Metrics

Primary:

[
CPC_{\Omega^{obs}}.
]

Secondary:

* MAE và RMSE trên observed flows;
* Spearman correlation;
* CPC theo từng distance bin;
* inflow/outflow CPC trên observed support;
* relative error của tổng flow;
* (Y_D)-fit chỉ là manipulation check.

Không báo cáo:

* support precision/recall;
* full-matrix CPC;
* performance trên missing pairs;

vì không có ground truth cho (\Omega^{mis}).

## 11. Phân tích missingness bắt buộc

Báo cáo cho từng city:

[
\rho_c=
\frac{|\Omega_c^{obs}|}{N_c(N_c-1)}.
]

Kiểm tra (\Delta_c^{obs}) theo:

* observation coverage (\rho_c);
* số zone;
* khoảng cách;
* flow magnitude;
* tỷ lệ short-distance/long-distance observed pairs.

Điều này giúp xác định liệu hiệu quả của (Y_D) chỉ xuất hiện ở các thành phố có observation support dày hay không.

## 12. Xử lý ba seed

Không xem (3\times40=120) là 120 quan sát độc lập. Với mỗi city, lấy trung bình theo seed:

[
\bar\Delta_c
============

\frac13
\sum_{s\in{42,2024,3000}}
\Delta_{c,s}.
]

Sau đó bootstrap và kiểm định trên 40 giá trị (\bar\Delta_c)
Run max 200 epoc
12 epoc not improve stop early
learning rate from: 3.2e^-3



Chay Y_D cho city level, county level va subzone level


Các kết quả ba seed hiện tại có thể được giữ nếu audit xác nhận:

* chỉ dùng target pair support;
* không dùng target individual flows trong (M_0);
* (M_1) chỉ nhận (Y_D);
* (q,m) không được chọn bằng test CPC.

## Kết luận khoa học được phép

> Conditional on the supplied observation support, target-city (Y_D^{obs}) provides a small but reproducible improvement in reconstructing observed positive OD-flow intensities beyond a frozen cross-city urban-context baseline.

Không được kết luận:

> (Y_D) cải thiện việc tái tạo toàn bộ ma trận OD.

Thiết kế này có internal validity tốt cho việc đo **marginal value của (Y_D) trên observed support**, nhưng external validity chỉ giới hạn ở các cặp được quan sát. Các missing pairs vẫn phải giữ trạng thái unknown.
