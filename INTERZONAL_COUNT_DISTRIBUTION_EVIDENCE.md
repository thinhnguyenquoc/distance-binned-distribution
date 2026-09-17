# Bằng chứng thực nghiệm: phân phối lưu lượng interzone

## Kết luận ngắn

Trên dữ liệu hiện tại trong `data/`, các lưu lượng interzone dương có
**overdispersion rất mạnh** và phù hợp với mô hình **zero-truncated negative
binomial (ZTNB)** hơn rất nhiều so với **zero-truncated Poisson (ZTP)**.

Kết luận này áp dụng cho phân phối có điều kiện trên các cặp interzone đã quan
sát với `trip_count >= 1`; nó không phải là kết luận về xác suất xuất hiện số
0 trong toàn bộ ma trận OD.

## Dữ liệu và bộ lọc

Phân tích dùng 50 thư mục thành phố trong `data/*/pairs/` và ghép
`od.csv` với `distance.csv` theo `(o_idx, d_idx)`. Giữ đúng support của
pipeline interzone:

```text
o_idx != d_idx
distance_km > 0
trip_count >= 1
```

Tổng số quan sát dương sau lọc là **6,065,339**, trên **50 thành phố**.

## Thống kê mô tả

| Thống kê | Giá trị |
|---|---:|
| Số quan sát pooled | 6,065,339 |
| Trung bình | 114.9950 |
| Phương sai mẫu | 193,459.9162 |
| Phương sai / trung bình | **1,682.3327** |
| Thành phố có variance/mean > 1 | **50/50** |
| Variance/mean nhỏ nhất theo thành phố | 708.0333 |
| Median variance/mean theo thành phố | 1,720.6400 |
| Variance/mean lớn nhất theo thành phố | 2,648.5053 |

Nếu dữ liệu Poisson phù hợp, kỳ vọng cơ bản là
$\operatorname{Var}(T) \approx \operatorname{E}(T)$). Tỷ số pooled **1,682 lần**
và việc cả 50 thành phố đều lớn hơn 1 là bằng chứng mô tả trực tiếp chống lại
Poisson.

## So sánh likelihood

Hai mô hình đều được ước lượng trên cùng tập quan sát dương và đều được hiệu
chỉnh zero-truncation:

| Mô hình | Ước lượng / chỉ số |
|---|---:|
| ZTP: $\hat{\lambda}$ | 114.9950 |
| ZTP: log-likelihood | -1,176,972,597.85 |
| ZTNB: mean nền $\hat{\mu}$ | 16.8344 |
| ZTNB: size $\hat{\phi}$ | 0.02418 |
| ZTNB: log-likelihood | -30,127,194.53 |
| ZTNB: AIC | 60,254,393.06 |
| ZTP: AIC | 2,353,945,201.70 |
| ZTNB - ZTP: log-likelihood | **+1,146,845,403.32** |
| Likelihood-ratio statistic | **2,293,690,806.63** |
| LR p-value | **< 1e-300** |

ZTNB có AIC thấp hơn khoảng **2.294 tỷ điểm**. Tham số `size` rất nhỏ cũng
phù hợp với overdispersion mạnh; với NB nền,

$$\operatorname{Var}(T)=\mu+\frac{\mu^2}{\phi}.$$

## Diễn giải và giới hạn

1. Đây là bằng chứng rất mạnh rằng Poisson equidispersion không mô tả được
   các **positive interzone counts** hiện tại.
2. Việc dùng ZTNB là cần thiết vì support phân tích đã loại các quan sát
   `trip_count <= 0`; do đó phải chuẩn hóa theo $1-P_{NB}(0)$.
3. Không được diễn giải kết quả này là bằng chứng rằng toàn bộ OD demand có
   phân phối ZTNB, vì các cặp bị thiếu/zero không nằm trong support này.
4. Kiểm định LR được báo cáo như chỉ báo thực nghiệm; Poisson là trường hợp
   biên của NB khi $\phi \to \infty$, nên p-value chi-square thông thường có
   một caveat lý thuyết. Caveat này không thay đổi kết luận vì chênh lệch
   likelihood là cực lớn.

## Nguồn và khả năng tái lập

- Nguồn dữ liệu: `data/*/pairs/od.csv` và `data/*/pairs/distance.csv`.
- Định nghĩa support: `PROTOCOL_CONTRACT.md` và
  `run_interzonal_experiment.py`.
- Likelihood ZTNB được triển khai tại `src/loss/ztnb.py`.
- Các con số trong báo cáo được tính từ dữ liệu hiện tại ngày 2026-09-17;
  nếu dữ liệu nguồn thay đổi, cần chạy lại phép tính trước khi dùng số liệu
  trong bản thảo.