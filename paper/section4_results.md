# Section 4: Empirical Results

In this section, we present the empirical evaluation designed to answer **RQ1** and **RQ2**. Across all experiments, our objective is not to propose a novel calibration algorithm, but to employ a closed-form, mass-preserving calibration operator as an **experimental instrument** to quantify the information value of target-city aggregate distance distributions ($Y_D$).

All evaluations are conducted under a strict 5-fold cross-validation protocol (10 held-out test cities per fold, totaling $N=50$ metropolitan areas across the United States) on the observed positive interzonal support $\Omega_c^+ = \{(i, j) \mid i \ne j, D_{ij} > 0, T_{ij} \ge 1\}$. The headline metric is the Common Part of Commuters (CPC) on interzonal flows, evaluated relative to the frozen zero-shot cross-city baseline $M_0$.

---

## 4.1 Target Distance Information Improves Zero-Shot OD Reconstruction

In the primary experiment, incorporating the oracle target-city distance-binned mobility distribution increases the mean interzonal CPC across 50 US metropolitan areas from 0.71281 for the zero-shot baseline ($M_0$) to 0.71635 after calibration ($M_1$). The resulting mean improvement is $\Delta\text{CPC} = +0.00354$, with a 95% confidence interval from the fold-stratified hierarchical bootstrap of $[+0.0026, +0.0045]$. The fact that the entire confidence interval lies strictly above zero demonstrates that the estimated mean CPC improvement is positive under the bootstrap protocol employed.

*(Tiếng Việt: Trong thí nghiệm chính, việc bổ sung phân phối di chuyển theo nhóm khoảng cách oracle của thành phố mục tiêu làm CPC liên vùng trung bình trên 50 thành phố Hoa Kỳ tăng từ 0.71281 ở mô hình zero-shot cơ sở ($M_0$) lên 0.71635 sau hiệu chỉnh ($M_1$). Mức cải thiện trung bình đạt $\Delta\mathrm{CPC}=+0.00354$, với khoảng tin cậy 95% từ fold-stratified hierarchical bootstrap là $[+0.0026,+0.0045]$. Toàn bộ khoảng tin cậy nằm phía trên 0, cho thấy mức cải thiện CPC trung bình được ước lượng là dương dưới giao thức bootstrap đã sử dụng.)*

As shown in Figure 2, the improvement is not concentrated in a small subset of cities but appears across the majority of evaluated metropolitan areas. Specifically, CPC increases after calibration in 45 out of 50 cities (90.0%). The median change, $\Delta\text{CPC} = +0.00195$, is also positive, although the magnitude of improvement varies considerably across cities. The remaining 5 cities exhibit lower CPC following calibration, indicating that the benefit of target distance information does not materialize in every instance. Overall, the city-level distribution indicates that the improvement, while modest in magnitude, is broadly consistent across the evaluation set.

*(Tiếng Việt: Theo Hình 2, mức cải thiện không chỉ tập trung ở một số ít thành phố mà xuất hiện trên phần lớn các thành phố được đánh giá. Cụ thể, CPC tăng sau hiệu chỉnh ở 45 trong 50 thành phố (90.0%). Trung vị $\Delta\mathrm{CPC}=+0.00195$ cũng nằm phía dương, mặc dù mức cải thiện khác nhau đáng kể giữa các thành phố. Năm thành phố còn lại có CPC giảm sau hiệu chỉnh, cho thấy lợi ích của thông tin khoảng cách không xuất hiện ở mọi trường hợp. Nhìn chung, phân bố theo thành phố cho thấy mức cải thiện có quy mô nhỏ nhưng khá nhất quán trên tập đánh giá.)*

To further evaluate whether this improvement is systematic, we conduct a two-sided paired Wilcoxon signed-rank test comparing $M_0$ and $M_1$ outcomes across the 50 cities. The test yields $p = 1.93 \times 10^{-9}$, providing strong evidence against the null hypothesis of no systematic difference between the two conditions. Combining these findings, the oracle target-city distance-binned mobility distribution provides a modest but consistent improvement over the zero-shot baseline across the vast majority of evaluated cities.

*(Tiếng Việt: Để kiểm tra thêm liệu xu hướng cải thiện này có mang tính hệ thống hay không, chúng tôi sử dụng kiểm định Wilcoxon signed-rank hai phía trên các cặp kết quả $M_0$ và $M_1$ của 50 thành phố. Kiểm định cho $p=1.93\times10^{-9}$, cung cấp bằng chứng mạnh chống lại giả thuyết không có sự thay đổi có hệ thống giữa hai điều kiện. Kết hợp các kết quả trên, phân phối di chuyển theo nhóm khoảng cách oracle của thành phố mục tiêu mang lại một mức cải thiện nhỏ nhưng nhất quán trên phần lớn các thành phố được đánh giá so với mô hình zero-shot cơ sở.)*

---

![Figure 2](figures/fig2_per_city_delta_cpc.png)
**Figure 2 | City-level improvement in interzonal CPC from oracle target-distance calibration.** Bars show the per-city performance change $\Delta\text{CPC}_c = \text{CPC}(M_{1,c}) - \text{CPC}(M_{0,c})$ for $N=50$ held-out test cities, ordered from lowest to highest. The dashed green line indicates the mean improvement ($+0.00354$) and the dotted orange line indicates the median improvement ($+0.00195$). Overall, 45 of 50 cities (90.0%) exhibit positive gains, with the primary fold-stratified 95% confidence interval spanning $[+0.0026, +0.0045]$.

---

### Table 1: Primary Zero-Shot Flow Reconstruction Benchmark ($N=50$ Cities, $K=8$ Bins)

| Model Condition | Mean Interzonal CPC | Median CPC | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | City Win Rate | Wilcoxon $p$ (Two-Sided) |
|---|---|---|---|---|---|---|
| **Zero-Shot Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **Calibrated Model ($M_1$)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | **$[+0.0026, +0.0045]$** | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ |

*Note: Evaluated on observed positive interzonal support $\Omega_c^+$. Confidence interval computed via $B=10,000$ fold-stratified bootstrap over cities. Seed-averaged across 3 independent model seeds.*
