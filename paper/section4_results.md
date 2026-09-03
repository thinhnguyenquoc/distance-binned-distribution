# Section 4: Empirical Results

In this section, we present the empirical evaluation designed to answer **RQ1** and **RQ2**. Across all experiments, our objective is not to propose a novel calibration algorithm, but to employ a closed-form, mass-preserving calibration operator as an **experimental instrument** to quantify the information value of target-city aggregate distance distributions ($Y_D$).

All evaluations are conducted under 5-fold cross-validation across the $N=50$ evaluated U.S. metropolitan areas on the observed positive interzonal support $\Omega_c^+ = \{(i, j) \mid i \ne j, D_{ij} > 0, T_{ij} \ge 1\}$. The headline metric is the Common Part of Commuters (CPC) on interzonal flows, evaluated relative to the frozen zero-shot cross-city baseline $M_0$.

---

## 4.1 Does $Y_D$ improve zero-shot OD reconstruction?

In the primary experiment, incorporating the oracle target-city distance-binned mobility distribution increased the mean interzonal CPC across 50 U.S. cities from 0.71281 for the zero-shot baseline ($M_0$) to 0.71635 after calibration ($M_1$). This corresponds to a mean improvement of $\Delta\mathrm{CPC}=+0.00354$, with a 95% confidence interval of $[+0.0026,+0.0045]$ obtained from the fold-stratified hierarchical bootstrap. Because the entire confidence interval lies above zero, the estimated mean improvement remains positive under the adopted bootstrap procedure.

*(Tiếng Việt: Trong thí nghiệm chính, việc bổ sung phân phối di chuyển theo nhóm khoảng cách oracle của thành phố mục tiêu làm CPC liên vùng trung bình trên 50 thành phố Hoa Kỳ tăng từ 0.71281 ở mô hình zero-shot cơ sở ($M_0$) lên 0.71635 sau hiệu chỉnh ($M_1$). Mức cải thiện trung bình đạt $\Delta\mathrm{CPC}=+0.00354$, với khoảng tin cậy 95% từ fold-stratified hierarchical bootstrap là $[+0.0026,+0.0045]$. Toàn bộ khoảng tin cậy nằm phía trên 0, cho thấy mức cải thiện CPC trung bình được ước lượng là dương dưới giao thức bootstrap đã sử dụng.)*

As shown in Figure 2, the improvement was not concentrated in a small subset of cities but was observed across most of the evaluation set. Specifically, CPC increased after calibration in 45 of 50 cities (90.0%). The median city-level change was also positive ($\Delta\mathrm{CPC}=+0.00195$), although the magnitude of improvement varied considerably across cities. The remaining five cities exhibited lower CPC after calibration, indicating that the benefit of target distance information did not occur in every case. Overall, the city-level distribution shows that the improvement was modest in magnitude but broadly consistent across the evaluated cities.

*(Tiếng Việt: Theo Hình 2, mức cải thiện không chỉ tập trung ở một số ít thành phố mà xuất hiện trên phần lớn các thành phố được đánh giá. Cụ thể, CPC tăng sau hiệu chỉnh ở 45 trong 50 thành phố (90.0%). Trung vị $\Delta\mathrm{CPC}=+0.00195$ cũng nằm phía dương, mặc dù mức cải thiện khác nhau đáng kể giữa các thành phố. Năm thành phố còn lại có CPC giảm sau hiệu chỉnh, cho thấy lợi ích của thông tin khoảng cách không xuất hiện ở mọi trường hợp. Nhìn chung, phân bố theo thành phố cho thấy mức cải thiện có quy mô nhỏ nhưng khá nhất quán trên tập đánh giá.)*

To further assess whether this pattern represented a systematic paired difference, we applied a two-sided Wilcoxon signed-rank test to the $M_0$ and $M_1$ results across the 50 cities. The test yielded $p=1.93\times10^{-9}$, providing strong evidence against the null hypothesis of no systematic paired difference between the two conditions. Taken together, these results indicate that the oracle target-city distance-binned mobility distribution provides a modest but consistent improvement over the zero-shot baseline across most evaluated cities.

*(Tiếng Việt: Để kiểm tra thêm liệu xu hướng cải thiện này có mang tính hệ thống hay không, chúng tôi sử dụng kiểm định Wilcoxon signed-rank hai phía trên các cặp kết quả $M_0$ và $M_1$ của 50 thành phố. Kiểm định cho $p=1.93\times10^{-9}$, cung cấp bằng chứng mạnh chống lại giả thuyết không có sự thay đổi có hệ thống giữa hai điều kiện. Kết hợp các kết quả trên, phân phối di chuyển theo nhóm khoảng cách oracle của thành phố mục tiêu mang lại một mức cải thiện nhỏ nhưng nhất quán trên phần lớn các thành phố được đánh giá so với mô hình zero-shot cơ sở.)*

---

![Figure 2](figures/fig2_main_per_city.png)
**Figure 2 | City-level improvement in interzonal CPC from oracle target-distance calibration.** Bars show the per-city performance change $\Delta\text{CPC}_c = \text{CPC}(M_{1,c}) - \text{CPC}(M_{0,c})$ for the $N=50$ evaluated test cities, ordered from lowest to highest. The dashed green line indicates the mean improvement ($+0.00354$) and the dotted orange line indicates the median improvement ($+0.00195$). Overall, 45 of 50 cities (90.0%) exhibit positive gains, with the primary fold-stratified 95% confidence interval spanning $[+0.0026, +0.0045]$.

---

### Table 1: Primary Zero-Shot Flow Reconstruction Benchmark ($N=50$ Cities, $K=8$ Bins)

| Model Condition | Mean Interzonal CPC | Median CPC | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | City Win Rate | Wilcoxon $p$ (Two-Sided) |
|---|---|---|---|---|---|---|
| **Zero-Shot Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **Calibrated Model ($M_1$)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | **$[+0.0026, +0.0045]$** | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ |

*Note: Evaluated on observed positive interzonal support $\Omega_c^+$. Confidence interval computed via $B=10,000$ fold-stratified bootstrap over cities. Seed-averaged across 3 independent model seeds.*

---

## 4.2 Is the gain genuinely target-specific and structurally meaningful?

Although the results in Section 4.1 demonstrate that calibration using the target-city distance-binned distribution ($Y_D$) improves CPC, they do not yet establish whether this improvement genuinely stems from target-specific distance information or is simply an artifact of the calibration process itself. To test this, we compare calibration using the true target-city $Y_D$ against calibration using distributions from other cities. To ensure a fair comparison, donor distributions from other cities are dose-matched so that they induce the exact same intervention magnitude ($D_T$) as the target-city distribution. When applying the true target-city $Y_D$, the mean CPC improvement reaches $\Delta\mathrm{CPC} = +0.003539$. In contrast, when using dose-matched donor distributions from other cities, the mean CPC change is only $\Delta\mathrm{CPC} = -0.000091$, representing virtually no improvement. The performance difference between the two conditions is $+0.003630$, with a 95% confidence interval of $[+0.00287, +0.00445]$. A one-sided Wilcoxon signed-rank test comparing target calibration against dose-matched wrong-city calibration yields $p = 2.19 \times 10^{-11}$. This result demonstrates that when the magnitude of calibration is controlled at the same level, donor distance distributions from other cities fail to replicate the performance gains achieved with the target city's own distribution. In other words, the benefit of calibration does not arise merely from altering predictions, but depends on whether the distance information is well matched to the target city.

*(Tiếng Việt: Mặc dù kết quả ở Mục 4.1 cho thấy việc hiệu chỉnh bằng phân phối di chuyển theo nhóm khoảng cách $Y_D$ của thành phố mục tiêu giúp cải thiện CPC, kết quả đó vẫn chưa cho biết liệu mức cải thiện có thực sự đến từ thông tin khoảng cách đặc thù của thành phố mục tiêu hay chỉ đơn giản là hệ quả của quá trình hiệu chỉnh. Để kiểm tra điều này, chúng tôi so sánh trường hợp sử dụng đúng $Y_D$ của thành phố mục tiêu với trường hợp sử dụng phân phối của các thành phố khác. Để bảo đảm so sánh công bằng, các phân phối từ thành phố khác được điều chỉnh sao cho tạo ra cùng mức độ can thiệp $D_T$ như trường hợp sử dụng thông tin của thành phố mục tiêu. Khi sử dụng đúng $Y_D$ của thành phố mục tiêu, mức cải thiện CPC trung bình đạt $\Delta\mathrm{CPC}=+0.003539$. Ngược lại, khi sử dụng các phân phối từ thành phố khác nhưng đã được khớp cùng mức độ can thiệp, mức thay đổi CPC trung bình chỉ là $\Delta\mathrm{CPC}=-0.000091$, tức gần như không mang lại cải thiện. Chênh lệch về mức cải thiện giữa hai điều kiện đạt $+0.003630$, với khoảng tin cậy 95% là $[+0.00287,+0.00445]$. Kiểm định Wilcoxon signed-rank một phía khi so sánh trường hợp sử dụng đúng thông tin của thành phố mục tiêu với trường hợp sử dụng thông tin từ thành phố khác cho $p=2.19\times10^{-11}$. Kết quả này cho thấy rằng khi mức độ hiệu chỉnh được kiểm soát ở cùng một mức, việc sử dụng phân phối khoảng cách của các thành phố khác không tái tạo được mức cải thiện đạt được khi sử dụng phân phối của chính thành phố mục tiêu. Nói cách khác, lợi ích của quá trình hiệu chỉnh không chỉ đến từ việc thay đổi dự báo mà còn phụ thuộc vào việc thông tin khoảng cách được sử dụng có phù hợp với thành phố mục tiêu hay không.)*

Another possibility is that precise knowledge of each target city's distance-binned distribution is unnecessary; instead, an average distribution constructed from training cities might suffice to yield a comparable improvement. Were this the case, the observed benefit would primarily reflect a generic distance-decay regularity rather than city-specific information. However, when applying the average distribution derived from training cities with the same calibration dose, the mean improvement is only $\Delta\mathrm{CPC} = +0.000914$, substantially lower than the $+0.003539$ achieved using the target city's own $Y_D$. The difference between these two conditions is $+0.002626$, with a 95% confidence interval of $[+0.00197, +0.00336]$ and a one-sided Wilcoxon test yielding $p = 4.03 \times 10^{-11}$. This indicates that while a generic distance-decay regularity can produce a small improvement, it does not replicate the gain attained when using the target city's specific distance distribution. This finding supports the role of city-specific information in $Y_D$ in driving the observed improvements.

*(Tiếng Việt: Một khả năng khác là không cần biết chính xác phân phối di chuyển theo khoảng cách của từng thành phố mục tiêu; thay vào đó, một phân phối trung bình được xây dựng từ các thành phố trong tập huấn luyện có thể đã đủ để mang lại mức cải thiện tương tự. Nếu điều này xảy ra, lợi ích quan sát được có thể chủ yếu đến từ một quy luật suy giảm theo khoảng cách mang tính tổng quát, thay vì từ thông tin đặc thù của từng thành phố. Tuy nhiên, khi sử dụng phân phối trung bình của các thành phố huấn luyện với cùng mức độ hiệu chỉnh, mức cải thiện trung bình chỉ đạt $\Delta\mathrm{CPC}=+0.000914$, thấp hơn so với $+0.003539$ khi sử dụng $Y_D$ của chính thành phố mục tiêu. Chênh lệch giữa hai điều kiện là $+0.002626$, với khoảng tin cậy 95% $[+0.00197,+0.00336]$ và kiểm định Wilcoxon một phía cho $p=4.03\times10^{-11}$. Kết quả này cho thấy một quy luật suy giảm theo khoảng cách tổng quát có thể tạo ra một mức cải thiện nhỏ, nhưng không tái tạo được mức cải thiện đạt được khi sử dụng phân phối khoảng cách đặc thù của thành phố mục tiêu. Điều này hỗ trợ vai trò của thông tin đặc thù theo thành phố trong $Y_D$ đối với mức cải thiện quan sát được.)*

In addition to tests using alternative distributions from other sources, we conduct a test by shuffling the distance bin positions within the target city's own $Y_D$. This permutation preserves the original proportions of the distribution but disrupts the relationship between each mobility proportion and its corresponding distance interval, thereby testing whether the distance structure of $Y_D$ is critical for the improvement. Under this condition, CPC decreases on average by $\Delta\mathrm{CPC} = -0.006964$, in contrast to the $+0.003539$ improvement obtained when using the correct $Y_D$. This result provides further evidence that the value of $Y_D$ lies not only in the observed mobility proportions, but also in binding those proportions to their corresponding distance intervals. Combined with the wrong-donor and training-mean placebo controls, these findings reinforce the evidence that the performance improvement is tied to structured, target-specific distance information.

*(Tiếng Việt: Bên cạnh các kiểm tra sử dụng phân phối thay thế từ những nguồn khác, chúng tôi còn thực hiện một phép kiểm tra bằng cách hoán đổi vị trí các khoảng trong chính $Y_D$ của thành phố mục tiêu. Phép hoán đổi này giữ nguyên các tỷ lệ ban đầu của phân phối nhưng phá vỡ mối quan hệ giữa mỗi tỷ lệ di chuyển và khoảng cách tương ứng, qua đó kiểm tra liệu cấu trúc theo khoảng cách của $Y_D$ có quan trọng đối với mức cải thiện hay không. Trong điều kiện này, CPC giảm trung bình với $\Delta\mathrm{CPC}=-0.006964$, trái ngược với mức cải thiện $\Delta\mathrm{CPC}=+0.003539$ khi sử dụng đúng $Y_D$. Kết quả này cung cấp thêm bằng chứng rằng giá trị của $Y_D$ không chỉ nằm ở các tỷ lệ di chuyển được quan sát mà còn ở việc các tỷ lệ đó được gắn đúng với các khoảng cách tương ứng. Kết hợp với các kiểm tra sử dụng phân phối sai thành phố và phân phối trung bình từ tập huấn luyện, kết quả này củng cố bằng chứng rằng mức cải thiện gắn với thông tin khoảng cách có cấu trúc và đặc thù của thành phố mục tiêu.)*

---

![Figure 5](figures/fig5_structural_validity_placebo.png)
**Figure 5 | Fair matched placebo controls.** Comparison of mean reconstruction gain $\Delta\mathrm{CPC}$ across $N=50$ test cities under three conditions from the fair matched placebo branch: (1) authentic target-city distribution ($Y_D$, $+0.00357$, $p < 10^{-8}$); (2) dose-matched cross-city donor placebo ($-0.00009$, not significant); and (3) permuted distance bins ($-0.00669$, $p < 10^{-14}$). Error bars represent 95% fold-stratified bootstrap confidence intervals over city-level values. This robustness visualization is distinct from the primary unified placebo estimates reported in Table 2.

---

### Table 2: Target Specificity and Placebo Controls ($N=50$ Cities; $B_{\text{draw}}=1000$, $B_{\text{boot}}=10,000$)

| Experimental Condition | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | Benefit vs $M_0$ ($p_{\text{2-sided}}$) | Specificity Gain vs Placebo | Specificity 95% CI | Target vs Placebo ($p_{\text{1-sided}}$) | Win Rate ($Target > Placebo$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Oracle Target $Y_D$ (Upper Bound)** | **$+0.003539$** | $[+0.00260, +0.00450]$ | $1.93 \times 10^{-9}$ | — | — | — | **45 / 50 (vs M0)** |
| **2. Dose-Matched Training Donors ($B_{\text{draw}}=1000$)** | **$-0.000091$** | $[-0.00089, +0.00071]$ | $0.4097$ (n.s.) | **$+0.003630$** | $[+0.00287, +0.00445]$ | $\mathbf{2.19 \times 10^{-11}}$ | **46 / 50 (92.0%)** |
| **3. Dose-Matched Fold Train-Mean $Y_D$** | **$+0.000914$** | $[+0.00001, +0.00186]$ | $0.4319$ (n.s.) | **$+0.002626$** | $[+0.00197, +0.00336]$ | $\mathbf{4.03 \times 10^{-11}}$ | **47 / 50 (94.0%)** |
| **4. Raw Test Donors (In-Fold 9-Donor Average, E1)** | **$-0.037721$** | $[-0.04357, -0.03268]$ | $1.78 \times 10^{-15}$ | **$+0.041261$** | $[+0.03641, +0.04688]$ | $8.88 \times 10^{-16}$ | **50 / 50 (100%)** |
| **5. Raw Test Donors ($B_{\text{draw}}=1000$ Draws)** | **$-0.037787$** | $[-0.04358, -0.03278]$ | $1.78 \times 10^{-15}$ | **$+0.041326$** | $[+0.03646, +0.04688]$ | $8.88 \times 10^{-16}$ | **50 / 50 (100%)** |
| **6. Raw Training Donors ($B_{\text{draw}}=1000$ Draws)** | **$-0.035148$** | $[-0.04014, -0.03067]$ | $1.78 \times 10^{-15}$ | **$+0.038687$** | $[+0.03431, +0.04349]$ | $8.88 \times 10^{-16}$ | **50 / 50 (100%)** |
| **7. Raw Fold Train-Mean $Y_D$** | **$-0.017735$** | $[-0.02365, -0.01243]$ | $4.91 \times 10^{-12}$ | **$+0.021275$** | $[+0.01613, +0.02706]$ | $4.44 \times 10^{-15}$ | **48 / 50 (96.0%)** |
| **8. Permuted Target $Y_D$ ($B_{\text{draw}}=1000$ Permutations)** | **$-0.006964$** | $[-0.00914, -0.00512]$ | $1.78 \times 10^{-15}$ | **$+0.010504$** | $[+0.00843, +0.01279]$ | $1.78 \times 10^{-15}$ | **49 / 50 (98.0%)** |

*Note: Evaluated across $N=50$ test cities $\times$ 3 model seeds. $B_{\text{draw}}=1000$ indicates the number of stochastic donor / permutation draws per city; $B_{\text{boot}}=10,000$ denotes fold-stratified bootstrap resamples for 95% CIs. Dose matching scales the L2 log-ratio perturbation norm of donor vectors to match the target city's intervention dose $D_T$. The primary placebo result reported here is the unified training-donor arm (Row 2, $p=2.19\times 10^{-11}$, $46/50$. For dose-matched train-mean (Row 3), the non-parametric Wilcoxon test reflects symmetric positive/negative city ranks ($p=0.4319$, n.s.) despite a slightly positive bootstrap mean CI.*

---

## 4.3 How does the value of $Y_D$ depend on observation resolution and quality?

The contribution of the target-city distance-binned mobility distribution may depend on the amount of structured information preserved during aggregation. We therefore examine two dimensions of observational resolution (distance granularity $K$ and spatial resolution) as well as observational fidelity under synthetic perturbations. These experiments investigate whether retaining finer-grained or higher-fidelity structure within the aggregate observation provides stronger, more effective constraints for zero-shot OD reconstruction.

*(Tiếng Việt: Mức độ đóng góp của phân phối di chuyển theo nhóm khoảng cách tại thành phố mục tiêu có thể phụ thuộc vào lượng thông tin tổng hợp mà quan sát này còn giữ lại được. Vì vậy, chúng tôi xem xét hai khía cạnh của độ phân giải quan sát (độ phân giải theo khoảng cách $K$ và độ phân giải theo không gian) cũng như độ trung thực của quan sát dưới các mức nhiễu tổng hợp. Các thí nghiệm này nhằm kiểm tra xem việc giữ lại nhiều cấu trúc chi tiết và chính xác hơn có cung cấp thêm các ràng buộc hữu ích cho quá trình tái tạo OD hay không.)*

---

### 4.3.1 Higher distance resolution provides more informative constraints

Across the tested values of $K$, the improvement in OD reconstruction increases as the number of distance bins grows. Even at the coarsest resolution ($K=2$), calibration with $Y_D$ improves mean CPC by $+0.00098$ over the frozen zero-shot baseline, with a 95% bootstrap confidence interval of $[+0.00052, +0.00151]$ and positive gains across 39 of 50 cities. The improvement reaches $+0.00354$ CPC at the canonical configuration ($K=8$) and $+0.00639$ CPC at $K=20$. At the highest tested resolution, 46 of 50 cities exhibit better performance than the zero-shot baseline, with the 95% bootstrap confidence interval remaining strictly positive ($[+0.00508, +0.00769]$).

*(Tiếng Việt: Trên các giá trị $K$ đã kiểm tra, mức cải thiện trong tái tạo OD tăng khi số lượng nhóm khoảng cách tăng. Ngay tại độ phân giải thấp nhất ($K=2$), việc hiệu chỉnh bằng $Y_D$ đã cải thiện CPC trung bình $+0.00098$ so với mô hình zero-shot cố định, với khoảng tin cậy bootstrap 95% là $[+0.00052,+0.00151]$, đồng thời cải thiện kết quả ở 39/50 thành phố. Mức cải thiện đạt $+0.00354$ CPC tại cấu hình tham chiếu ($K=8$) và $+0.00639$ CPC tại $K=20$. Ở độ phân giải cao nhất được kiểm tra, 46/50 thành phố có kết quả tốt hơn zero-shot baseline và khoảng tin cậy bootstrap 95% vẫn nằm hoàn toàn trên 0, $[+0.00508,+0.00769]$.)*

### Table 3: Information Resolution Scaling Across Distance Bins ($K \in \{2, 4, \dots, 20\}$)

| Resolution ($K$) | Mean Interzonal CPC | Median CPC | Mean $\Delta\text{CPC}$ | Median $\Delta\text{CPC}$ | 95% Fold-Stratified CI | City Win Rate | Average Gain / Bin ($\Delta\text{CPC}/K$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — | — |
| **$K = 2$** | $0.71379 \pm 0.04441$ | $0.71665$ | **$+0.00098$** | $+0.00034$ | $[+0.00052, +0.00151]$ | **39 / 50 (78.0%)** | $0.000488$ |
| **$K = 4$** | $0.71479 \pm 0.04439$ | $0.71720$ | **$+0.00198$** | $+0.00088$ | $[+0.00125, +0.00279]$ | **39 / 50 (78.0%)** | $0.000494$ |
| **$K = 6$** | $0.71570 \pm 0.04445$ | $0.71784$ | **$+0.00289$** | $+0.00152$ | $[+0.00201, +0.00384]$ | **44 / 50 (88.0%)** | $0.000481$ |
| **$K = 8$ (Anchor)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | $+0.00195$ | $[+0.00262, +0.00447]$ | **45 / 50 (90.0%)** | $0.000442$ |
| **$K = 10$** | $0.71694 \pm 0.04450$ | $0.72007$ | **$+0.00413$** | $+0.00235$ | $[+0.00311, +0.00514]$ | **45 / 50 (90.0%)** | $0.000413$ |
| **$K = 12$** | $0.71761 \pm 0.04453$ | $0.72060$ | **$+0.00480$** | $+0.00288$ | $[+0.00372, +0.00590]$ | **46 / 50 (92.0%)** | $0.000400$ |
| **$K = 14$** | $0.71819 \pm 0.04456$ | $0.72145$ | **$+0.00538$** | $+0.00373$ | $[+0.00424, +0.00654]$ | **45 / 50 (90.0%)** | $0.000384$ |
| **$K = 16$** | $0.71855 \pm 0.04458$ | $0.72205$ | **$+0.00574$** | $+0.00433$ | $[+0.00455, +0.00694]$ | **46 / 50 (92.0%)** | $0.000359$ |
| **$K = 18$** | $0.71884 \pm 0.04460$ | $0.72230$ | **$+0.00603$** | $+0.00458$ | $[+0.00480, +0.00726]$ | **47 / 50 (94.0%)** | $0.000335$ |
| **$K = 20$** | $0.71920 \pm 0.04462$ | $0.72266$ | **$+0.00639$** | $+0.00494$ | $[+0.00508, +0.00769]$ | **46 / 50 (92.0%)** | $0.000319$ |

*Note: Evaluated across $N=50$ test cities $\times$ 3 model seeds on $\Omega_c^+$. Bins are defined by pair-weighted distance quantiles from 35 training cities per fold. Bootstrap confidence intervals computed via $B=10,000$ fold-stratified resamples.*

---

### 4.3.2 County-level calibration yields a small pooled incremental gain
*(Tiếng Việt: **4.3.2. Hiệu chỉnh cấp county tạo ra mức tăng bổ sung pooled nhỏ**)*

Across all 50 cities, county-level calibration yields a small pooled incremental gain over city-level calibration ($\Delta\mathrm{CPC}_{\mathrm{res}}=+0.00014$, 95% CI $[+0.00002,+0.00028]$, Wilcoxon $p=0.0064$). This pooled result must be interpreted in light of the benchmark structure. For 39 single-county cities, $M1_{\mathrm{county}}\equiv M1_{\mathrm{city}}$ by construction, and therefore $\Delta\mathrm{CPC}_{\mathrm{res},c}=0$ exactly. The empirical comparison of finer spatial observation is consequently concentrated in the 11 multi-county cities.

Across the evaluated multi-county subset, county-level calibration produced a small positive average incremental gain (mean $\Delta\mathrm{CPC}_{\mathrm{res}}=+0.00063$), with improvements in 9 of 11 cities. This subgroup result is descriptive unless a separately verified uncertainty estimate is reported. The observed pattern is consistent with the possibility that finer origin-group distance distributions add information in some multi-county metropolitan datasets, but the study does not directly measure or test intra-urban heterogeneity as the mechanism.

*(Tiếng Việt: Trên toàn bộ 50 thành phố, hiệu chỉnh cấp county tạo ra mức tăng bổ sung pooled nhỏ so với hiệu chỉnh cấp city ($\Delta\mathrm{CPC}_{\mathrm{res}}=+0.00014$, khoảng tin cậy 95% $[+0.00002,+0.00028]$, Wilcoxon $p=0.0064$). Kết quả pooled này cần được diễn giải theo cấu trúc của benchmark. Với 39 thành phố single-county, $M1_{\mathrm{county}}\equiv M1_{\mathrm{city}}$ theo cấu trúc, do đó $\Delta\mathrm{CPC}_{\mathrm{res},c}=0$ chính xác. Vì vậy, phép so sánh thực nghiệm về quan sát không gian chi tiết hơn tập trung vào 11 thành phố multi-county.)*

*(Tiếng Việt: Trên nhóm các thành phố multi-county đã đánh giá, hiệu chỉnh cấp county tạo ra mức tăng bổ sung trung bình nhỏ và dương (mean $\Delta\mathrm{CPC}_{\mathrm{res}}=+0.00063$), với 9/11 thành phố cải thiện. Kết quả subgroup này mang tính mô tả nếu chưa có một ước lượng bất định riêng đã được xác minh. Mẫu hình quan sát được phù hợp với khả năng rằng các phân phối theo nhóm origin chi tiết hơn có thể bổ sung thông tin trong một số bộ dữ liệu đô thị multi-county, nhưng nghiên cứu không đo lường hoặc kiểm định trực tiếp tính không đồng nhất nội đô như một cơ chế.)*

---

![Figure 3](figures/fig3_resolution_sensitivity.png)
**Figure 3 | Observational resolution sensitivity.** **(a)** Mean calibration gain $\Delta\mathrm{CPC}$ across $N=50$ test cities as a function of the number of distance bins $K \in \{2, 4, 6, 8, 10, 12, 14, 16, 18, 20\}$ with 95% fold-stratified bootstrap confidence intervals. Gain increases across the tested values while average gain per bin declines. **(b)** Comparison of city-level vs. county-level calibration across the $N=11$ evaluated multi-county metropolitan areas; these subgroup differences are descriptive.

---

### 4.3.3 Synthetic observation noise reduces the value of $Y_D$

Having assessed the impact of observational resolution, we next investigate how calibration efficacy depends on the fidelity of $Y_D$. Specifically, we perturb the target city's distance-binned mobility distribution across varying noise levels ($\epsilon \in [0.00, 0.05]$ Total Variation error), while holding the zero-shot baseline model, evaluation test cities, and calibration procedure strictly identical. This design isolates the effect of estimation errors in $Y_D$ from other sources of model variance.

*(Tiếng Việt: Sau khi đánh giá ảnh hưởng của độ phân giải quan sát, chúng tôi tiếp tục kiểm tra mức độ phụ thuộc của hiệu quả hiệu chỉnh vào chất lượng của $Y_D$. Cụ thể, phân phối di chuyển theo khoảng cách của thành phố mục tiêu được gây nhiễu ở nhiều mức khác nhau ($\epsilon \in [0.00, 0.05]$ sai số Total Variation), trong khi giữ nguyên mô hình zero-shot, tập thành phố đánh giá và toàn bộ quy trình hiệu chỉnh. Thiết kế này cho phép cô lập ảnh hưởng của sai lệch trong $Y_D$ khỏi các nguồn biến thiên khác của mô hình.)*

---

![Figure 4](figures/fig4_noise_dose_response.png)
**Figure 4 | Effect of observation fidelity on calibration benefit across 50 metropolitan areas.** The solid blue curve displays the mean interzonal $\Delta\mathrm{CPC}$ across the 50 evaluated test cities as a function of Total Variation (TV) perturbation magnitude $\epsilon$ in the target-city aggregate distance observation $Y_D$. The shaded band denotes the 95% fold-stratified bootstrap confidence interval. The dashed vertical line marks the empirical signal breakdown crossover threshold ($\epsilon_{\mathrm{cross}} = 4.44\%$ TV error).

---

The empirical results in Figure 4 show monotonic degradation across the tested synthetic noise levels: as perturbation magnitude increases, the CPC gain decreases. The uncorrupted observation yields the largest improvement ($\Delta\mathrm{CPC}=+0.00354$), while the gain falls to $+0.00070$ at $4\%$ TV noise and becomes negative at $5\%$ TV noise ($-0.00087$). Across 1,000 synthetic noise directions, the mean empirical crossover is estimated at $\epsilon_{\mathrm{cross}}=4.44\%$ TV error (95% CI $[4.16\%,4.77\%]$; the across-city summary is $4.39\%$ with 95% CI $[3.66\%,4.94\%]$). This benchmark-specific dose-response pattern shows that utility decreases as the synthetic observation departs from the reference distribution; it does not define a universal tolerance for real-world observations.

*(Tiếng Việt: Kết quả trên Hình 4 cho thấy mức tăng suy giảm đơn điệu qua các mức nhiễu tổng hợp đã kiểm tra. Quan sát không nhiễu tạo ra mức tăng lớn nhất ($+0.00354$); mức tăng giảm còn $+0.00070$ tại sai số TV $4\%$ và trở thành âm tại $5\%$ ($-0.00087$). Trên 1.000 hướng nhiễu tổng hợp, điểm giao cắt trung bình được ước lượng tại $\epsilon_{\mathrm{cross}}=4.44\%$ (khoảng tin cậy 95% $[4.16\%,4.77\%]$). Đây là quan hệ dose-response riêng cho benchmark và thiết kế perturbation này, không phải ngưỡng dung sai phổ quát cho quan sát thực tế.)*

Under this perturbation design, mean calibration gain remains positive at lower tested noise levels (e.g., $+0.00336$ at $1\%$ TV and $+0.00282$ at $2\%$ TV). The decline at higher perturbations also shows that $Y_D$ cannot be treated as beneficial irrespective of observation quality.

*(Tiếng Việt: Trong thiết kế perturbation này, mức tăng trung bình vẫn dương tại các mức nhiễu thấp đã kiểm tra, chẳng hạn $+0.00336$ ở TV $1\%$ và $+0.00282$ ở TV $2\%$. Sự suy giảm ở các mức nhiễu cao hơn cũng cho thấy không thể xem $Y_D$ là có lợi bất kể chất lượng quan sát.)*

---

### Table 4: Perturbation Tolerance and Noise Sensitivity Across Total Variation Error Levels

| TV Noise Level ($\epsilon$) | Mean Calibrated CPC | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | Positive Cities | Degradation vs Clean (Holm-adjusted $p$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\epsilon = 0.00$ (Clean Target $Y_D$)** | $0.71635$ | **$+0.00354$** | $[+0.00261, +0.00451]$ | **45 / 50 (90.0%)** | — |
| **$\epsilon = 0.01$ (1% TV Error)** | $0.71617$ | **$+0.00336$** | $[+0.00243, +0.00432]$ | **44 / 50 (88.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.02$ (2% TV Error)** | $0.71563$ | **$+0.00282$** | $[+0.00189, +0.00379]$ | **36 / 50 (72.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.03$ (3% TV Error)** | $0.71474$ | **$+0.00193$** | $[+0.00100, +0.00290]$ | **28 / 50 (56.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.04$ (4% TV Error)** | $0.71351$ | **$+0.00070$** | $[-0.00025, +0.00167]$ | **18 / 50 (36.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.05$ (5% TV Error)** | $0.71193$ | **$-0.00087$** | $[-0.00183, +0.00012]$ | 17 / 50 (34.0%) | $4.44 \times 10^{-15}$ |

*Note: Evaluated across $N=50$ test cities $\times$ 3 model seeds at $K=8$. Synthetic perturbations use centered Gaussian directions in log-ratio space ($z \sim \mathcal{N}(0, I)$, zero-mean centered) and are scaled numerically via exponential tilting ($p_\sigma \propto p \exp(\sigma z)$) to achieve the specified Total Variation error magnitudes $\epsilon = \frac{1}{2}\sum_k |Y_k - \tilde{Y}_k|$. Degradation $p$-values are family-wise error rate controlled across noise levels via Holm-Bonferroni adjustment. The mean signal breakdown crossover threshold across $B=1,000$ noise directions is $\epsilon_{\text{cross}} = 4.44\%$ [95% CI: 4.16%, 4.77%].*

---

## 4.4 Is the finding robust to training and modeling choices?

The preceding results establish that $Y_D$ provides supplemental structural information for zero-shot OD reconstruction, with its efficacy governed by observation resolution, fidelity, and target specificity. However, it is essential to verify whether the observed performance gains remain stable across stochastic training variations and alternative model architectures. We therefore evaluate multiple independent model seeds and distinct predictive backbones. A separate protocol-specific comparison examines the performance obtained from direct pairwise OD observations.

*(Tiếng Việt: Các kết quả trước cho thấy $Y_D$ cung cấp thông tin bổ sung hữu ích cho dự báo zero-shot, đồng thời mức độ hữu ích này phụ thuộc vào độ phân giải, chất lượng quan sát và tính đặc thù mục tiêu. Tuy nhiên, cần kiểm tra liệu mức cải thiện quan sát được có ổn định trước biến thiên của quá trình huấn luyện và lựa chọn mô hình hay không. Vì vậy, chúng tôi đánh giá nhiều model seeds và các backbone dự báo khác nhau. Một phép so sánh riêng theo protocol kiểm tra hiệu năng thu được từ quan sát trực tiếp các cặp OD.)*

---

### 4.4.1 Stability across independent model initializations

Deep learning architectures may exhibit variability across training runs due to stochastic weight initialization and optimization dynamics. If the benefit of $Y_D$ were confined to a single idiosyncratic model initialization, the empirical effect might merely reflect training noise rather than a stable, systematic contribution from the target observation.

*(Tiếng Việt: Các mô hình học sâu có thể tạo ra kết quả khác nhau giữa các lần huấn luyện do sự ngẫu nhiên trong khởi tạo tham số và quá trình tối ưu. Nếu lợi ích của $Y_D$ chỉ xuất hiện ở một model seed cụ thể, hiệu ứng quan sát được có thể phản ánh biến thiên ngẫu nhiên của quá trình huấn luyện thay vì một đóng góp ổn định từ quan sát mục tiêu.)*

To test this possibility, we evaluate the identical 5-fold cross-city protocol across three independent model seeds (Seeds 1, 10, and 100). For each city and seed, the uncalibrated zero-shot baseline $M_0$ is compared directly against its calibrated counterpart $M_1$, after which performance changes are aggregated across seeds. This matched-pairs design directly assesses the impact of $Y_D$ within the exact same baseline optimization state, isolating the effect of calibration from absolute cross-seed performance variance.

*(Tiếng Việt: Để kiểm tra khả năng này, chúng tôi đánh giá cùng một protocol trên ba model seeds độc lập (Seed 1, 10 và 100). Với mỗi thành phố và mỗi seed, zero-shot baseline $M_0$ được so sánh trực tiếp với phiên bản được hiệu chỉnh bằng $Y_D$, sau đó mức thay đổi CPC được tổng hợp qua các seed. Thiết kế ghép cặp này cho phép đánh giá trực tiếp ảnh hưởng của $Y_D$ trong cùng một trạng thái baseline, thay vì để sự khác biệt về chất lượng tuyệt đối giữa các lần huấn luyện chi phối kết quả.)*

The results in Table 5 demonstrate that the positive performance gain conferred by $Y_D$ is robustly reproduced across all model initializations. Across the three seeds, the mean $\Delta\mathrm{CPC}$ improvement remains consistently positive ($+0.00434$ for Seed 1, $+0.00308$ for Seed 10, and $+0.00320$ for Seed 100), with 95% fold-stratified bootstrap confidence intervals strictly excluding zero in every run ($[+0.00322, +0.00547]$, $[+0.00216, +0.00404]$, and $[+0.00236, +0.00408]$, respectively). City-level win rates remain exceptionally high across all initializations ($82.0\%$, $88.0\%$, and $88.0\%$). Across all 50 cities, the across-seed standard deviation of mean $\Delta\mathrm{CPC}$ is only $\mathrm{SD} = 0.00070$, and the mean per-city seed variance is $\mathrm{SD}_{\mathrm{city}} = 0.00126$.

*(Tiếng Việt: Kết quả cho thấy hướng cải thiện do $Y_D$ mang lại được duy trì qua các model seeds, mặc dù CPC tuyệt đối của từng mô hình có thể thay đổi nhẹ giữa các lần huấn luyện. Điều này cho thấy hiệu ứng của $Y_D$ không phụ thuộc vào một nghiệm tối ưu ngẫu nhiên cụ thể, mà xuất hiện lặp lại khi cùng loại thông tin của thành phố mục tiêu được sử dụng để hiệu chỉnh dự báo zero-shot.)*

---

### Table 5: Model Initialization Robustness Across Independent Seeds ($N=50$ Cities, $K=8$ Bins)

| Model Seed | Mean $M_0$ CPC | Mean $M_1$ CPC | Mean $\Delta\text{CPC}$ | Median $\Delta\text{CPC}$ | 95% Fold-Stratified CI | City Win Rate |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 1** | $0.70861 \pm 0.04492$ | $0.71295 \pm 0.04491$ | **$+0.00434$** | $+0.00207$ | $[+0.00322, +0.00547]$ | **41 / 50 (82.0%)** |
| **Seed 10** | $0.71477 \pm 0.04443$ | $0.71785 \pm 0.04470$ | **$+0.00308$** | $+0.00182$ | $[+0.00216, +0.00404]$ | **44 / 50 (88.0%)** |
| **Seed 100** | $0.71504 \pm 0.04439$ | $0.71824 \pm 0.04471$ | **$+0.00320$** | $+0.00217$ | $[+0.00236, +0.00408]$ | **44 / 50 (88.0%)** |
| **Seed-Averaged (Canonical)** | **$0.71281 \pm 0.04434$** | **$0.71635 \pm 0.04454$** | **$+0.00354$** | **$+0.00195$** | **$[+0.00260, +0.00451]$** | **45 / 50 (90.0%)** |

*Note: Evaluated across all $N=50$ test cities on observed positive interzonal support $\Omega_c^+$. Across-seed standard deviation of mean $\Delta\mathrm{CPC}$ is $\mathrm{SD} = 0.00070$.*

---

### 4.4.2 Performance across neural backbones and classical gravity

Beyond stochastic variation in model initialization, a vital question is whether the benefit of $Y_D$ depends idiosyncratically on a specific neural backbone architecture. We substitute the Urban GNN backbone with a simpler Node-level Multi-Layer Perceptron (Node MLP) without graph message passing, as well as a classical parametric gravity model, while holding the input feature set, 5-fold cross-city evaluation protocol, test cities, and calibration operator strictly identical.

*(Tiếng Việt: Bên cạnh biến thiên do khởi tạo mô hình, một câu hỏi khác là liệu lợi ích của $Y_D$ có chỉ xuất hiện khi sử dụng một kiến trúc backbone cụ thể hay không. Chúng tôi thay backbone Urban GNN bằng một mô hình MLP đơn giản hơn, cũng như một mô hình trọng lực cổ điển, trong khi giữ nguyên tập đặc trưng đầu vào, protocol huấn luyện, tập thành phố đánh giá và cơ chế hiệu chỉnh bằng $Y_D$.)*

The results in Table 6 show that the calibration gain appears across both tested learned neural backbones but is attenuated for the classical gravity baseline. For the Node MLP backbone, calibration improves mean interzonal CPC from $0.70913$ to $0.71242$, yielding $\Delta\mathrm{CPC}=+0.00329$ (95% bootstrap CI $[+0.0025,+0.0042]$, Wilcoxon $p=4.38\times10^{-11}$) with positive gains in 47 of 50 cities (94.0%). For the classical gravity model, calibration produces a marginal, non-significant gain ($\Delta\mathrm{CPC}=+0.00084$, win rate 22/50, Wilcoxon $p=0.3545$). Within the architectures tested, this contrast suggests that distance-binned mass reallocation is more useful when the base model already captures richer non-linear spatial structure.

*(Tiếng Việt: Kết quả tại Bảng 6 cho thấy mức tăng do hiệu chỉnh xuất hiện trên cả hai neural backbone đã kiểm tra nhưng suy giảm trên mô hình trọng lực cổ điển. Với Node MLP, hiệu chỉnh cải thiện CPC trung bình $+0.00329$ ($p=4.38\times10^{-11}$, thắng 47/50 thành phố). Với gravity baseline cổ điển, hiệu chỉnh chỉ tạo ra mức tăng nhỏ không có ý nghĩa thống kê ($+0.00084$, thắng 22/50, $p=0.3545$). Trong phạm vi các kiến trúc đã kiểm tra, sự tương phản này gợi ý rằng tái phân bổ khối lượng theo khoảng cách hữu ích hơn khi mô hình cơ sở đã học được cấu trúc không gian phi tuyến phong phú hơn.)*

---

### Table 6: Backbone Model Generality and Architecture Robustness ($N=50$ Cities, $K=8$ Bins)

| Architecture | Zero-Shot $M_0$ CPC | Calibrated $M_1$ CPC | Mean $\Delta\text{CPC}$ | 95% Bootstrap CI | City Win Rate | Wilcoxon $p$ | $\Delta\text{RMSE}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Urban GNN (Message-Passing)** | $0.71281 \pm 0.04434$ | $0.71635 \pm 0.04454$ | **$+0.00354$** | $[+0.0026, +0.0045]$ | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ | $-2.98$ |
| **Node MLP (No Graph MP)** | $0.70913 \pm 0.04754$ | $0.71242 \pm 0.04737$ | **$+0.00329$** | $[+0.0025, +0.0042]$ | **47 / 50 (94.0%)** | $\mathbf{4.38 \times 10^{-11}}$ | $-2.57$ |
| **Classical 2-Param Gravity** | $0.38868 \pm 0.15312$ | $0.38952 \pm 0.15435$ | $+0.00084$ | $[+0.0002, +0.0016]$ | 22 / 50 (44.0%) | $0.3545$ (n.s.) | $-0.93$ |

*Note: All models evaluated under identical 5-fold cross-validation ($N=50$ test cities $\times$ 3 seeds). Gravity model calibrated using standard maximum likelihood on training folds.*

---

### 4.4.3 Protocol-specific comparison with direct pairwise OD observations

To evaluate whether the observed benefit merely reflects generic target supervision rather than the structured value of distance-aggregated constraints, we compare the reconstruction gain from the $K=8$ distance-binned distribution with direct observations of positive interzonal OD pairs across sampling proportions $p\in[0.10\%,5.0\%]$. Direct-OD performance is evaluated on unseen pairs across all 50 test cities under an OD Fixed-Effect residual adapter (OD-FE).

*(Tiếng Việt: Để kiểm tra xem liệu lợi ích quan sát được có đơn thuần phản ánh việc mô hình nhận thêm target supervision nói chung hay không, chúng tôi so sánh $Y_D$ với các tỷ lệ quan sát OD trực tiếp $p \in [0.10\%, 5.0\%]$ trên các cặp chưa thấy bằng mô hình OD Fixed-Effect adapter.)*

Within this specific OD-FE comparison, Table 7 identifies an interpolated operational crossing near $p_{\mathrm{eq}}\approx0.20\%$ of positive interzonal pairs. Revealing $0.10\%$ of pairs yields an unseen-pair gain of $\Delta\mathrm{CPC}=+0.00180$, below the $+0.00354$ achieved by $Y_D$ (difference $D=-0.00174$, 95% CI $[-0.00279,-0.00068]$). Revealing $0.25\%$ yields $\Delta\mathrm{CPC}=+0.00448$ ($D=+0.00094$). Linear interpolation between these two evaluated points places the crossing at $0.20\%$ (95% bootstrap interval: $[0.133\%,0.287\%]$), corresponding to approximately 35 revealed tract-to-tract flows per city on average. This is an operational comparison under the specified OD-FE adapter, sampling design, support, and metric; it is not a general equivalence between eight aggregate values and OD survey records.

*(Tiếng Việt: Trong phép so sánh OD-FE cụ thể này, Bảng 7 xác định điểm giao cắt vận hành nội suy gần $p_{\mathrm{eq}}\approx0.20\%$ tổng số cặp OD liên vùng dương. Việc tiết lộ $0.10\%$ số cặp mang lại mức tăng $\Delta\mathrm{CPC}=+0.00180$ trên các cặp chưa thấy, thấp hơn mức $+0.00354$ của $Y_D$ (chênh lệch $D=-0.00174$, khoảng tin cậy 95% $[-0.00279,-0.00068]$). Khi tỷ lệ tăng lên $0.25\%$, mức tăng đạt $+0.00448$ ($D=+0.00094$). Nội suy tuyến tính giữa hai điểm đã đánh giá đặt điểm giao cắt tại $0.20\%$ (khoảng bootstrap 95% $[0.133\%,0.287\%]$), tương ứng trung bình khoảng 35 luồng tract-to-tract được tiết lộ trên mỗi thành phố. Đây là so sánh vận hành dưới OD-FE adapter, thiết kế lấy mẫu, support và metric đã nêu; kết quả không thiết lập một quan hệ tương đương chung giữa tám giá trị tổng hợp và dữ liệu khảo sát OD.)*

One interpretation is that the two signals act at different structural scales. A revealed OD value informs one pair, whereas each component of $Y_D$ constrains the total predicted mass of all supported pairs in a distance band. Thus, the eight-bin vector influences many pairwise predictions simultaneously through the shared calibration factor.

*(Tiếng Việt: Sự khác biệt giữa hai loại thông tin nằm ở phạm vi tác động. Một quan sát OD trực tiếp cung cấp thông tin về một cặp cụ thể, trong khi mỗi thành phần của $Y_D$ mô tả tổng khối lượng di chuyển trên một tập lớn các cặp có khoảng cách tương tự. Do đó, mặc dù $Y_D$ có số chiều rất thấp, mỗi thành phần của nó có khả năng ràng buộc đồng thời nhiều dự báo OD thông qua cấu trúc khoảng cách chung.)*

---

### Table 7: Protocol-Specific Direct-OD Performance Comparison ($N=50$ Test Cities, Evaluated on Unseen Pairs)

| Revealed OD Fraction ($p$) | Unseen $M_0$ CPC | Full $Y_D$ Gain ($K=8$) | Direct-OD Gain ($\Delta\text{CPC}$) | Difference vs Full $Y_D$ ($D(p)$) | 95% Bootstrap CI | Cities Direct $\ge$ Full $Y_D$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$0.00\%$** | $0.7128$ | $+0.00354$ | $+0.00000$ | $-0.00354$ | $[-0.00450, -0.00260]$ | 5 / 50 |
| **$0.10\%$** | $0.7128$ | $+0.00354$ | $+0.00180$ | $-0.00174$ | $[-0.00279, -0.00068]$ | 22 / 50 |
| **$0.20\%$ (Interpolated Crossing $p_{\text{eq}}$)** | $0.7128$ | $+0.00354$ | **$+0.00354$** | **$0.00000$** | $[-0.00140, +0.00150]$ | 26 / 50 |
| **$0.25\%$** | $0.7128$ | $+0.00354$ | $+0.00448$ | $+0.00094$ | $[-0.00051, +0.00259]$ | 29 / 50 |
| **$0.50\%$** | $0.7128$ | $+0.00354$ | $+0.00859$ | $+0.00505$ | $[+0.00289, +0.00765]$ | 36 / 50 |
| **$1.00\%$** | $0.7128$ | $+0.00354$ | $+0.01549$ | $+0.01195$ | $[+0.00883, +0.01560]$ | 46 / 50 |
| **$5.00\%$** | $0.7128$ | $+0.00354$ | $+0.04363$ | $+0.04009$ | $[+0.03507, +0.04542]$ | 50 / 50 |

*Note: Evaluated across all $N=50$ test cities on unseen OD pairs. The OD-FE experiment used $B=200$ Monte Carlo replicates per city. Linear interpolation between the 0.10% and 0.25% evaluated conditions places the operational crossing at $p_{\mathrm{eq}}\approx0.20\%$ (95% bootstrap interval $[0.133\%,0.287\%]$; approximately 35 revealed flows per city). The comparison is specific to the OD-FE adapter, sampling protocol, positive support, and CPC metric. It must not be conflated with a distinct partial-OD-to-$Y_D$ calibration formulation, whose comparison with OD-FE is deferred to future work.*

---

### 4.4.4 Synthesis of calibration robustness and stability

The positive calibration gain is reproduced across multiple independent model seeds and both evaluated neural backbones (Urban GNN and Node MLP). The classical gravity baseline exhibits only a small, non-significant change, so the architecture evidence should be interpreted as support for robustness across the two learned neural backbones rather than across all model families. Distance-resolution sensitivity is evaluated using pair-weighted quantile bins derived exclusively from the training cities. Together, these results indicate that the main finding is not attributable to a single parameter initialization or to the Urban GNN architecture alone.

*(Tiếng Việt: Mức tăng do hiệu chỉnh được tái hiện qua nhiều model seeds độc lập và trên cả hai neural backbone đã đánh giá là Urban GNN và Node MLP. Gravity baseline cổ điển chỉ cho mức thay đổi nhỏ, không có ý nghĩa thống kê; vì vậy bằng chứng kiến trúc chỉ hỗ trợ robustness trên hai neural backbone đã kiểm tra, không mở rộng cho mọi họ mô hình. Phân tích độ nhạy theo độ phân giải khoảng cách sử dụng pair-weighted quantile bins được xây dựng hoàn toàn từ các thành phố huấn luyện. Tổng hợp lại, kết quả chính không phải hệ quả riêng của một lần khởi tạo tham số hoặc chỉ của kiến trúc Urban GNN.)*

---

## 4.5 Baseline distance misalignment is strongly associated with city-level calibration gain

Although $Y_D$ confers positive gains across the vast majority of test cities, the magnitude of improvement $\Delta\mathrm{CPC}$ varies substantially across metropolitan environments (e.g., Los Angeles $+0.01543$, Phoenix $+0.01258$, Houston $+0.00976$, whereas other cities exhibit modest changes). This inter-city variation demonstrates that the empirical value of $Y_D$ is inherently conditional.

*(Tiếng Việt: Mặc dù $Y_D$ mang lại mức cải thiện dương trên phần lớn các thành phố, độ lớn của $\Delta\mathrm{CPC}$ không đồng nhất giữa các khu vực mục tiêu. Sự khác biệt này cho thấy giá trị của $Y_D$ mang tính điều kiện và có liên quan đến trạng thái ban đầu của zero-shot baseline tại từng thành phố.)*

To understand the mechanics governing this variation, we first examine an intrinsic property of the calibration operator. For an OD pair $(i,j)$ residing in distance bin $k$, the calibrated flow prediction is given by

$$
\hat{t}_{ij}^{(1)} = w_k \hat{t}_{ij}^{(0)}.
$$

Because all OD pairs within the same bin share the identical scalar multiplier $w_k$, the operator rescales the aggregate flow volume of each bin while leaving pairwise relative proportions strictly invariant:

$$
\frac{\hat{t}_{ij}^{(1)}}{\hat{t}_{uv}^{(1)}} = \frac{\hat{t}_{ij}^{(0)}}{\hat{t}_{uv}^{(0)}} \quad \forall (i,j), (u,v) \in \text{bin } k.
$$

*(Tiếng Việt: Cơ chế hiệu chỉnh nhân tất cả các cặp OD trong cùng một khoảng khoảng cách với cùng một hệ số $w_k$. Do đó, quá trình hiệu chỉnh thay đổi tổng khối lượng di chuyển của từng bin nhưng giữ nguyên tuyệt đối tỷ lệ tương đối giữa các cặp OD bên trong cùng một bin.)*

This mathematical property dictates that bin scaling *cannot* alter intra-bin pair rankings. One might hypothesize that a baseline with superior intra-bin ranking fidelity ($Q_c^{\mathrm{intra}}$) would derive greater benefit from calibration. In this sample, however, the estimated association is small and not statistically distinguishable from zero ($r=+0.046$, $p=0.75$); this null result does not establish that intra-bin fidelity is irrelevant.

*(Tiếng Việt: Giới hạn toán học này cho thấy hiệu chỉnh không thể sửa thứ tự nội bin. Một giả thuyết có thể đặt ra là baseline có chất lượng xếp hạng nội bin ($Q_c^{\mathrm{intra}}$) tốt hơn sẽ hưởng lợi nhiều hơn. Tuy nhiên, trong mẫu hiện tại, liên hệ ước lượng nhỏ và không phân biệt được với 0 về mặt thống kê ($r=+0.046$, $p=0.75$); kết quả null này không chứng minh rằng chất lượng nội bin không quan trọng.)*

By contrast, baseline distance-distribution mismatch $d_{\mathrm{pre}}=\mathrm{TV}(\hat{Y}_D^{(0)},Y_D^{\mathrm{GT}})$ is strongly associated with cross-city gain heterogeneity. As reported in Table 8, $d_{\mathrm{pre}}$ correlates with $\Delta\mathrm{CPC}$ in both Pearson ($r=+0.7995$, $p=3.36\times10^{-12}$) and Spearman analyses ($\rho=+0.7464$, $p=4.92\times10^{-10}$). After controlling for baseline accuracy ($M_0$ CPC), number of tracts ($\log N_{\mathrm{tracts}}$), total pairs ($\log N_{\mathrm{pairs}}$), and mean geographic distance, the partial correlation remains high ($r_{\mathrm{partial}}=+0.7951$, $p=5.35\times10^{-12}$). The multivariate linear model has $R^2=73.7\%$, and the coefficient for $d_{\mathrm{pre}}$ remains positive ($\beta=+0.1487$, $t=+8.70$, $p=4.12\times10^{-11}$). These observational diagnostics support association and mechanism consistency, not a causal effect of $d_{\mathrm{pre}}$.

*(Tiếng Việt: Ngược lại, sai lệch phân phối khoảng cách ban đầu $d_{\mathrm{pre}}=\mathrm{TV}(\hat{Y}_D^{(0)},Y_D^{\mathrm{GT}})$ có liên hệ mạnh với tính không đồng nhất của mức tăng giữa các thành phố (Pearson $r=+0.7995$; partial $r=+0.7951$, $p=5.35\times10^{-12}$). Mô hình hồi quy đa biến có $R^2=73.7\%$ và hệ số của $d_{\mathrm{pre}}$ vẫn dương ($\beta=+0.1487$, $t=+8.70$, $p=4.12\times10^{-11}$). Đây là chẩn đoán liên hệ quan sát phù hợp với cơ chế đề xuất, không phải bằng chứng nhân quả.)*

---

![Figure 6](figures/fig6_mechanistic_dpre.png)
**Figure 6 | Mechanistic diagnostic: Calibration gain increases with baseline distance misalignment.** Scatter plot of baseline distance mismatch $d_{\mathrm{pre}} = \mathrm{TV}(\hat{Y}_D^{(0)}, Y_D^{\mathrm{GT}})$ versus reconstruction gain $\Delta\mathrm{CPC}$ across all $N=50$ test cities. The green line depicts the linear regression fit ($R^2 = 73.7\%$, Pearson $r = +0.7995$, $p = 3.36 \times 10^{-12}$, partial $r = +0.7951$, $p = 5.35 \times 10^{-12}$ controlling for baseline performance and network scale).

---

### Table 8: Mechanistic Regression and Partial Correlation Analysis for Baseline Distance Mismatch ($d_{\text{pre}}$)

| Specification | Control Variables | Metric | Value | $p$-value | Significance |
|---|---|:---:|:---:|:---:|:---:|
| **Raw Bivariate Pearson** | None | $r$ | **$+0.7995$** | $3.36 \times 10^{-12}$ | *** |
| **Raw Bivariate Spearman** | None | $\rho$ | **$+0.7464$** | $4.92 \times 10^{-10}$ | *** |
| **Partial Correlation 1** | Baseline accuracy ($M_0$ CPC) | $r_{\text{part}}$ | **$+0.8067$** | $1.52 \times 10^{-12}$ | *** |
| **Partial Correlation 2** | Network size ($\log N_{\text{tracts}}$) | $r_{\text{part}}$ | **$+0.7936$** | $6.25 \times 10^{-12}$ | *** |
| **Full Partial Correlation** | $M_0 + \log N_{\text{pairs}} + \log N_{\text{tracts}} + \text{MeanDist}$ | $r_{\text{part}}$ | **$+0.7951$** | $\mathbf{5.35 \times 10^{-12}}$ | *** |
| **Multivariate OLS Regression** | All Controls ($R^2 = 73.7\%$) | $\beta(d_{\text{pre}})$ | **$+0.1487$** | $\mathbf{4.12 \times 10^{-11}}$ | *** ($t = +8.70$) |

*Note: Evaluated across all $N=50$ test cities. $d_{\mathrm{pre}} = \mathrm{TV}(\hat{Y}_D^{(0)}, Y_D^{\mathrm{GT}})$ measures the Total Variation error between the zero-shot baseline's distance allocation and ground truth. Multivariate OLS serves as an observational diagnostic for linear association with performance gain heterogeneity rather than a causal model. Significance: *** $p < 0.001$.*

