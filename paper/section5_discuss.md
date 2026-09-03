# Section 5: Discussion

*(Tiếng Việt: **Mục 5: Thảo luận chuyên sâu**)*

In this section, we contextualize our findings within the broader literature on human mobility modeling and spatial transfer learning [@barbosa2018humanmobility; @enaya2026transgm; @lenormand2016comparison; @simini2021deepgravity]. We examine the theoretical mechanisms underlying the information value of aggregate distance distributions, evaluate observational resolution and noise sensitivity under controlled synthetic perturbations, discuss methodological and practical implications for data-scarce urban analytics, and outline key limitations and future research directions.

*(Tiếng Việt: Trong phần này, chúng tôi đặt các phát hiện của nghiên cứu vào bức tranh tổng thể của các nghiên cứu về mô hình hóa di chuyển con người và học chuyển giao không gian [@barbosa2018humanmobility; @enaya2026transgm; @lenormand2016comparison; @simini2021deepgravity]. Chúng tôi phân tích các cơ chế lý thuyết giải thích giá trị thông tin của phân phối cự ly tổng hợp, đánh giá độ phân giải quan sát và độ nhạy đối với nhiễu tổng hợp có kiểm soát, thảo luận ý nghĩa phương pháp luận và thực tiễn cho phân tích đô thị khan hiếm dữ liệu, đồng thời nêu rõ các hạn chế chính và định hướng nghiên cứu tiếp theo.)*

---

## 5.1 Main findings and information value
*(Tiếng Việt: **5.1. Các phát hiện chính và giá trị thông tin**)*

Research on human mobility encompasses diverse data sources, spatial scales, and modeling frameworks, with origin–destination (OD) matrices representing a foundational formulation of spatial interaction at the population level [@barbosa2018humanmobility]. Recent neural mobility architectures demonstrate that geographic context features and learned spatial representations from multiple training regions can effectively support mobility flow prediction in urban areas unseen during model training [@simini2021deepgravity; @guo2025ugnn].

The present study extends this line of inquiry by investigating whether a low-dimensional aggregate observation of the target city—specifically, its distance-binned trip distribution ($Y_D$)—provides actionable supplementary information to a pre-trained, frozen cross-city neural model.

Our empirical benchmark across 50 U.S. metropolitan areas demonstrates that conditioning on the target city's distance distribution yields a **small but statistically significant and consistent improvement** over the zero-shot baseline ($M_0$; Table 1). Across 5-fold cross-validation and three independent model initializations, city-level calibration increases the mean Common Part of Commuters from $0.71281$ to $0.71635$, corresponding to an average gain of $\overline{\Delta\mathrm{CPC}} = +0.00354$ ($95\%\text{ CI: } [+0.0026, +0.0045]$, median $+0.00195$, paired Wilcoxon signed-rank test $W = 83.0, p = 1.93 \times 10^{-9}$). Crucially, positive gains occur in 45 of the 50 evaluated cities (a 90.0% directional win rate).

*(Tiếng Việt: Nghiên cứu về di chuyển con người bao gồm nhiều dạng dữ liệu, thang không gian và mô hình khác nhau, trong đó OD matrices là một biểu diễn quan trọng của tương tác không gian ở cấp độ quần thể [@barbosa2018humanmobility]. Các mô hình neural mobility gần đây cho thấy đặc trưng địa lý và biểu diễn học từ nhiều khu vực có thể hỗ trợ dự báo luồng tại những khu vực không xuất hiện trong huấn luyện [@simini2021deepgravity; @guo2025ugnn]. Nghiên cứu hiện tại mở rộng hướng tiếp cận này bằng cách kiểm tra liệu một quan sát tổng hợp có số chiều thấp của thành phố mục tiêu có cung cấp thông tin bổ sung cho một mô hình cross-city đã được đóng băng hay không. Kết quả thực nghiệm trên 50 thành phố cho thấy $\mathbf{Y}_{D,c}$ tạo ra mức cải thiện nhỏ nhưng có ý nghĩa thống kê và nhất quán (Bảng 1: $\overline{\Delta\mathrm{CPC}} = +0.00354$, $95\%\text{ CI: } [+0.0026, +0.0045]$, $p = 1.93 \times 10^{-9}$, thắng 45/50 thành phố).)*

However, the scientific interpretation of this result warrants careful calibration. The average magnitude of improvement ($\Delta\mathrm{CPC} \approx +0.0035$) is modest in absolute terms; $Y_D$ does not replace granular OD survey data or fundamentally transform baseline fidelity on its own. Rather, it indicates that low-dimensional distance distributions contain useful aggregate structure that pre-trained spatial neural networks cannot infer from cross-city priors and static geographic features alone.

Importantly, the target distribution $\mathbf{Y}_{D,c}$ in our benchmark is synthesized directly from reference OD flows as an **oracle aggregate observation**. Consequently, the current findings assess the potential information ceiling of an ideal, error-free distance distribution. They do not demonstrate real-world deployment performance with noisy, missing, or third-party empirical telemetry streams.

*(Tiếng Việt: Cần lưu ý rằng $\mathbf{Y}_{D,c}$ trong thí nghiệm được tổng hợp từ OD tham chiếu dưới dạng oracle aggregate observation. Vì vậy, kết quả hiện tại đánh giá giá trị thông tin tiềm năng của một phân phối khoảng cách chính xác, chứ chưa chứng minh hiệu quả triển khai với một nguồn quan sát bên ngoài có nhiễu hoặc thiếu dữ liệu.)*

---

## 5.2 Mechanistic explanation: Macro distance reallocation vs intra-bin ranking
*(Tiếng Việt: **5.2. Cơ chế giải thích: Tái phân bổ cự ly vĩ mô và thứ hạng nội khoảng**)*

Travel distance or generalized transportation cost has long been established as the central impedance component of spatial interaction models [@wilson1971family]. Classic calibration frameworks emphasize that empirical distance-decay profiles should be estimated from observed travel patterns rather than assumed fixed across disparate urban environments [@hyman1969calibration]. Contemporary studies further establish that distance-decay curves vary substantially across travel modes, trip purposes, urbanization levels, and socioeconomic contexts [@verma2025distance].

In this study, $Y_D$ is not used to estimate a parametric gravity deterrence function. Instead, it directly supplies the empirical mass proportions required for each distance interval.

While the zero-shot baseline ($M_0$) utilizes tract-level context features and pairwise Haversine distances, it cannot observe how actual travel demand in an unseen target city is partitioned across journey lengths. Our empirical diagnostics indicate a strong positive association between the baseline's initial distance-distribution mismatch ($d_{\text{pre}} = \mathrm{TV}(\hat{Y}_D^{M0}, Y_D)$) and subsequent calibration gain ($\Delta\mathrm{CPC}_c$; Figure 6, Table 8). Even after controlling for baseline accuracy, network size, total pair count, and mean trip distance, the partial correlation remains high ($r_{\text{partial}} = +0.7951, p = 5.35 \times 10^{-12}$, multiple regression $R^2 = 73.7\%$). This pattern is **consistent with an inter-bin mass reallocation mechanism**, whereby calibration delivers larger gains when the baseline's macro distance profile deviates substantially from the target distribution. However, this statistical association represents observational correlation and does not, on its own, establish strict causality.

*(Tiếng Việt: Khoảng cách hoặc chi phí di chuyển từ lâu đã được xem là thành phần impedance trung tâm trong spatial-interaction models [@wilson1971family]. Các phương pháp calibration cổ điển cũng nhấn mạnh rằng hình dạng distance-decay cần được xác định từ thông tin di chuyển quan sát được thay vì được giả định là cố định giữa các bối cảnh [@hyman1969calibration]. Các nghiên cứu gần đây tiếp tục cho thấy distance-decay có thể thay đổi theo phương thức, mục đích chuyến đi, mức độ đô thị hóa và đặc điểm kinh tế–xã hội [@verma2025distance]. Trong nghiên cứu này, $Y_D$ không được dùng để ước lượng một hàm gravity tham số. Thay vào đó, nó cung cấp trực tiếp tỷ lệ khối lượng cần được phân bổ vào từng khoảng cách. Mối liên hệ dương mạnh giữa sai lệch ban đầu $d_{\mathrm{pre}}$ và $\Delta\mathrm{CPC}$ phù hợp với cơ chế tái phân bổ khối lượng liên khoảng ($r_{\text{partial}} = +0.7951, R^2 = 73.7\%$, Hình 6, Bảng 8), nhưng không thiết lập quan hệ nhân quả.)*

Because the calibration operator multiplies all predictions within distance bin $b$ by a common positive scalar $s_{c,b} > 0$, **intra-bin pair rankings are mathematically invariant** (and Kendall's $\tau=1$ for all non-degenerate groups with sufficient pairs). Empirically, intra-bin ranking quality ($Q_c^{\text{intra}}$) exhibits no significant monotonic correlation with calibration gain ($r=+0.046$, $p=0.75$). This non-significant result does not demonstrate that intra-bin ranking is irrelevant; it simply indicates that our sample provides no evidence of a monotonic relationship between baseline intra-bin ranking accuracy and macro calibration benefit. Ultimately, reconstructed flow quality remains bounded by the baseline's internal ranking capacity, as post-hoc distance scaling cannot correct misranked pairs within the same distance interval.

*(Tiếng Việt: Do tất cả cặp trong cùng một khoảng được nhân với cùng một hệ số dương (Mục 3.5), phép hiệu chỉnh bảo toàn thứ tự nội khoảng về mặt toán học. Phân tích thực nghiệm không phát hiện mối tương quan có ý nghĩa giữa chỉ số chất lượng nội khoảng $Q_c^{\mathrm{intra}}$ và mức cải thiện ($r=0.046, p=0.75$). Kết quả không có ý nghĩa thống kê này không chứng minh rằng chất lượng nội khoảng hoàn toàn không quan trọng; nó chỉ cho thấy dữ liệu hiện tại chưa cung cấp bằng chứng về một quan hệ đơn điệu giữa hai đại lượng. Chất lượng cuối cùng vẫn bị giới hạn bởi cấu trúc nội khoảng mà baseline đã dự báo, vì bước hiệu chỉnh không thể sửa thứ tự sai giữa các cặp thuộc cùng một nhóm.)*

---

## 5.3 Observational resolution and diminishing marginal returns
*(Tiếng Việt: **5.3. Độ phân giải thông tin và quy luật lợi suất giảm dần**)*

Prior studies demonstrate that low-dimensional aggregate travel statistics can provide valuable structural information for calibrating constrained models. For instance, median travel time can calibrate single-parameter spatial interaction models when sufficient structural network information is known [@merlin2020medians]. The present study differs by employing the full $K$-bin mass proportion vector to directly adjust predicted OD intensities, rather than inferring a single scalar distance parameter.

Across the tested distance-bin granularities ($K\in\{2,4,6,8,10,12,14,16,18,20\}$; Figure 3a, Table 3), calibration gain increases with resolution. Even at $K=2$, mean CPC improves by $+0.00098$, rising to $+0.00354$ at $K=8$ (canonical) and $+0.00639$ at $K=20$. However, average gain per bin peaks at $K=4$ ($4.94\times10^{-4}/\text{bin}$) and decreases to $3.19\times10^{-4}/\text{bin}$ at $K=20$. One plausible interpretation is that coarse partitions separate broad mobility regimes, whereas finer intervals impose increasingly localized constraints; this interpretation should be tested directly in future work.

Even at $K=20$, the aggregate observation represents a very small dimensionality relative to the number of positive OD pairs ($K / |\Omega_c^+| < 0.1\%$, averaging $\approx 1,757$ positive OD pairs per bin). This result concerns information compression: a low-dimensional summary can still provide structurally useful information for calibration. This dimensionality reduction should not be interpreted as a privacy guarantee. The study does not evaluate re-identification risk, differential privacy, or any release mechanism for $\mathbf{Y}_{D,c}$; therefore, it makes no claim that the aggregate observation is privacy-preserving [@demontjoye2013unique; @houssiau2022differential].

*(Tiếng Việt: Một số nghiên cứu trước cho thấy các thống kê di chuyển tổng hợp có số chiều thấp vẫn có thể chứa thông tin hữu ích cho calibration trong những mô hình giới hạn. Chẳng hạn, median travel time có thể được dùng để hiệu chỉnh một spatial-interaction model đơn tham số khi thông tin cấu trúc cần thiết đã được biết [@merlin2020medians]. Nghiên cứu hiện tại khác với hướng này ở chỗ sử dụng toàn bộ vector tỷ lệ theo $K$ khoảng để hiệu chỉnh trực tiếp cường độ OD dự báo, thay vì suy luận một tham số distance-decay duy nhất. Ngay cả tại $K=20$, quan sát tổng hợp vẫn có số chiều rất nhỏ so với số cặp OD dương ($K / |\Omega_c^+| < 0.1\%$, trung bình khoảng 1.757 cặp OD dương trên mỗi bin). Kết quả này phản ánh khả năng nén thông tin: một thống kê tóm tắt có số chiều thấp vẫn có thể cung cấp thông tin cấu trúc hữu ích cho hiệu chỉnh. Việc giảm số chiều này không nên được diễn giải là một bảo đảm quyền riêng tư. Nghiên cứu không đánh giá rủi ro tái nhận dạng, differential privacy hoặc bất kỳ cơ chế công bố nào cho $\mathbf{Y}_{D,c}$; vì vậy, nghiên cứu không khẳng định quan sát tổng hợp này là privacy-preserving [@demontjoye2013unique; @houssiau2022differential].)*

---

## 5.4 Spatial semantic ordering and synthetic noise breakdown
*(Tiếng Việt: **5.4. Tính đúng thứ tự không gian và ngưỡng phá vỡ do nhiễu**)*

The utility of $Y_D$ depends on its spatial distance semantics under the evaluated conditions. Randomly permuting the bin order while preserving the numerical values causes severe performance degradation ($\Delta\mathrm{CPC}=-0.00696$, a deficit of $0.01050$ relative to target calibration, $p<10^{-14}$; Table 2). This supports the interpretation that the observed benefit is not explained by generic output variance reduction or smoothing alone, but relies on binding mobility proportions to the corresponding physical distance intervals.

Under synthetic Total Variation noise ($\epsilon \in [0\%, 5\%]$), calibration gains degrade monotonically, crossing zero at:

$$\epsilon_{\text{cross}} \approx 4.44\% \quad [95\%\text{ CI: } 4.16\%, 4.77\%]$$

This noise experiment must be interpreted within the broader context of mobility data quality. Sampling bias, coverage limitations, and data processing pipelines can introduce structured distortions that fundamentally alter empirical conclusions [@gallotti2024distorted; @pappalardo2023future]. Consequently, the empirical threshold observed here ($\epsilon_{\text{cross}} \approx 4.44\%$) is specific to our synthetic perturbation design, benchmark dataset, and neural baseline; it does not serve as a universal operational guarantee for all real-world empirical data streams.

*(Tiếng Việt: Trong các điều kiện đã đánh giá, giá trị sử dụng của $Y_D$ gắn với nội dung ngữ nghĩa không gian: hoán vị sai thứ tự các khoảng làm sụt giảm nghiêm trọng CPC ($\Delta\mathrm{CPC}=-0.00696$, $p<10^{-14}$, Bảng 2). Kết quả noise experiment cần được diễn giải trong bối cảnh rộng hơn của chất lượng mobility data. Nguồn dữ liệu, độ phủ mẫu và quy trình xử lý có thể tạo ra các sai lệch làm thay đổi kết luận rút ra từ dữ liệu di chuyển [@gallotti2024distorted; @pappalardo2023future]. Vì vậy, ngưỡng nhiễu quan sát được ($\epsilon_{\text{cross}}\approx4.44\%$, Hình 4) chỉ là một ngưỡng thực nghiệm dưới cơ chế perturbation đã thiết kế, không phải bảo đảm chung cho mọi nguồn dữ liệu thực tế.)*

---

## 5.5 Target specificity vs generic distance decay priors
*(Tiếng Việt: **5.5. Tính đặc thù mục tiêu và các prior suy giảm cự ly phổ quát**)*

The transferability of mobility models across geographic domains is frequently constrained by inter-city divergences in urban scale, spatial topology, and data availability for calibration [@yang2014limits]. Recent transfer learning frameworks likewise establish that the degree of required domain adaptation depends on structural similarity between source and target urban systems [@enaya2026transgm].

Our dose-matched placebo benchmarks evaluate whether target observations convey city-specific idiosyncrasies or merely restate universal distance decay principles (Table 2):
1. **Dose-Matched Wrong Donors**: Applying donor distributions from incorrect cities scaled to the target's intervention dose ($D_T$) produces no systematic gain ($\Delta\mathrm{CPC} = -0.000091, p = 0.4097$). The true target distribution outperforms dose-matched wrong donors in 46 of 50 cities ($+0.003630, p = 2.19 \times 10^{-11}$).
2. **Dose-Matched Training-Mean**: Applying the mean distance profile across training cities yields a marginal change of $+0.000914$, which is **statistically indistinguishable from zero** ($p = 0.4319$). The target-specific distribution outperforms the training-mean profile in 47 of 50 cities ($+0.002626, p = 4.03 \times 10^{-11}$).

Because the baseline model already incorporates pairwise Haversine distances between tracts alongside urban context features, these results demonstrate that static geographic distances and cross-city priors do not fully account for target-city travel distance composition. This supports the city-specific informational value of $Y_D$, while not implying that geographic distance in general is insufficient for spatial mobility modeling.

*(Tiếng Việt: Khả năng chuyển giao của mobility models giữa các khu vực thường bị giới hạn bởi khác biệt về quy mô, cấu trúc không gian và mức độ sẵn có của dữ liệu hiệu chỉnh [@yang2014limits]. Các phương pháp transfer gần đây cũng cho thấy mức độ thích nghi cần thiết phụ thuộc vào sự tương đồng cấu trúc giữa thành phố nguồn và thành phố mục tiêu [@enaya2026transgm]. Do đó, việc target-specific $Y_D$ vượt trội hơn wrong-donor ($\Delta = -0.000091, p=0.4097$) và training-mean observations ($\Delta = +0.000914, p=0.4319$, không phân biệt được với 0; Bảng 2) phù hợp với nhận định rằng một prior cross-city chung chưa thể biểu diễn đầy đủ cấu trúc di chuyển của mọi thành phố.)*

---

## 5.6 Inter-city performance heterogeneity
*(Tiếng Việt: **5.6. Sự không đồng nhất về hiệu quả giữa các thành phố**)*

Prior literature establishes that the comparative performance of trip-distribution models, distance-decay functions, and calibration procedures varies substantially across distinct datasets and spatial scales [@lenormand2016comparison]. Furthermore, empirical distance decay parameters remain inherently context-dependent [@verma2025distance].

In our benchmark, although 45 of 50 cities exhibit positive gains, performance varies across metropolitan areas, with 5 cities exhibiting negative changes (El Paso, Oklahoma City, Jacksonville, Louisville, Long Beach). This inter-city variation is not an anomalous artifact, but reflects the inherent context-dependence of mobility modeling:
- **Low Baseline Misalignment**: In cities where the zero-shot baseline already closely matches the target distance profile ($d_{\text{pre}} \approx 0$), there is minimal room for macro reallocation.
- **Intra-Bin Error Dominance**: In cities where baseline errors stem primarily from misallocating flows among zone pairs within the same distance band rather than across bands, scalar distance calibration cannot rectify the underlying distortion.

Consequently, $Y_D$ calibration should be viewed as a conditioned post-processing tool whose efficacy depends on baseline macro alignment, rather than an unconditional guarantee of improvement.

*(Tiếng Việt: Các benchmark trước đây cho thấy hiệu quả của trip-distribution models, distance-decay functions và calibration procedures thay đổi giữa các bộ dữ liệu và thang không gian [@lenormand2016comparison]. Sự không đồng nhất giữa các thành phố trong nghiên cứu hiện tại (với 45 thành phố tăng và 5 thành phố giảm nhẹ) vì vậy không phải là một ngoại lệ bất thường, mà phản ánh tính phụ thuộc bối cảnh vốn có của mobility modelling [@verma2025distance]. Hiệu chỉnh $Y_D$ là một công cụ suy luận có điều kiện phụ thuộc vào độ lệch cự ly vĩ mô ban đầu của baseline.)*

---

## 5.7 County-level resolution: descriptive evidence and mechanism hypothesis
*(Tiếng Việt: **5.7. Độ phân giải cấp county: bằng chứng mô tả và giả thuyết cơ chế**)*

The spatial resolution experiment examines whether the utility of $Y_D$ changes when the aggregate constraint is supplied at the county rather than city level. The pooled incremental gain across all 50 cities is small ($+0.00014$, 95% CI $[+0.00002,+0.00028]$, $p=0.0064$). This result includes 39 single-county cities, for which county-level and city-level calibration are mathematically identical and the incremental difference is exactly zero by construction.

In the city-level configuration (`M1_city`), a single vector $\mathbf{Y}_{D,c}$ modulates flow mass across distance intervals for the entire metropolis. This operator effectively rectifies average distance decay biases in the baseline, but applies an identical set of scaling multipliers to all origin tracts. Consequently, it cannot accommodate settings where distinct subregions within the same urban area exhibit markedly different distance distributions.

Across the 11 evaluated multi-county cities, the mean incremental gain is $+0.00063$, with improvements in 9 of 11 cities. Because no separately verified uncertainty artifact is reported for this subset, this result is descriptive. Descriptive city-level values for the 11 multi-county datasets are reported in Table S1, while the aggregate spatial-resolution pattern is summarized in Figure 3b. A city-wide distribution applies the same set of distance-bin constraints across all origin tracts, whereas county-level calibration allows the constraints to vary across origin-county groups. This provides a plausible hypothesis for the localized gains observed in the multi-county subset; it is not a direct test that county boundaries capture functional mobility heterogeneity. County membership is an administrative proxy, and the study does not independently measure the degree of intra-urban mobility divergence represented by that proxy.

This formulation does not support a general condition linking county-level aggregation to improved reconstruction. It instead reports a small pooled gain, exact invariance where county grouping adds no partition, and a descriptive positive pattern in the evaluated multi-county subset. The calibration operator only reallocates flow mass between distance intervals or origin-county slices; it leaves the relative ordering of OD pairs within each slice strictly invariant. Consequently, overall accuracy remains bounded by the baseline's capacity to rank zone pairs internally.

Three explicit limitations warrant consideration:
1. **Administrative vs Functional Zoning**: County boundaries are administrative units and are not designed as functional commuting basins or travel communities. Whether functional urban zones or mobility communities produce more informative aggregate constraints requires separate study.
2. **Dataset Footprint Boundary**: County groups comprise only those tracts included within the study city dataset provided by the laboratory, and do not represent total county-wide travel demand extending beyond the study area.
3. **Oracle Aggregate Setting**: County distributions in our benchmark are derived as oracle aggregate observations from reference OD matrices. These results demonstrate the theoretical information ceiling of county-level granularity, but do not prove that equivalent gains would materialize under noisy or incomplete real-world telemetry.

In summary, the county-level experiment provides a small pooled incremental result and descriptive evidence in the evaluated multi-county subset. It motivates, but does not test, the hypothesis that finer origin-group constraints may be useful when they encode information not represented by a city-wide distribution.

*(Tiếng Việt: Thí nghiệm độ phân giải không gian kiểm tra liệu giá trị của $Y_D$ có thay đổi khi ràng buộc tổng hợp được cung cấp ở cấp county thay vì city hay không. Mức tăng bổ sung pooled trên toàn bộ 50 thành phố là nhỏ ($+0.00014$, khoảng tin cậy 95% $[+0.00002,+0.00028]$, $p=0.0064$). Kết quả này bao gồm 39 thành phố single-county, nơi hiệu chỉnh cấp county và cấp city tương đương về mặt toán học, do đó chênh lệch bổ sung bằng 0 chính xác theo cấu trúc.)*

*(Tiếng Việt: Trong cấu hình city-level, một vector $\mathbf{Y}_{D,c}$ duy nhất áp dụng cùng một tập ràng buộc theo khoảng cách cho mọi origin tract. Ngược lại, hiệu chỉnh cấp county cho phép các ràng buộc thay đổi giữa những nhóm origin-county.)*

*(Tiếng Việt: Trên 11 thành phố multi-county đã đánh giá, mức tăng bổ sung trung bình là $+0.00063$, với 9/11 thành phố cải thiện. Do chưa có artifact bất định riêng đã được xác minh cho subgroup này, kết quả mang tính mô tả. Các giá trị mô tả theo thành phố cho 11 bộ dữ liệu multi-county được trình bày trong Bảng S1, còn mẫu hình tổng hợp về độ phân giải không gian được tóm tắt trong Hình 3b. Phân phối cấp city áp dụng cùng một tập ràng buộc theo khoảng cách cho mọi origin tract, trong khi hiệu chỉnh cấp county cho phép các ràng buộc thay đổi giữa những nhóm origin-county. Đây là một giả thuyết hợp lý cho các mức tăng cục bộ quan sát được trong nhóm multi-county; kết quả không phải phép kiểm định trực tiếp rằng ranh giới county biểu diễn tính không đồng nhất chức năng của di chuyển. County membership là một administrative proxy, và nghiên cứu không đo lường độc lập mức độ khác biệt di chuyển nội đô được đại diện bởi proxy này.)*

*(Tiếng Việt: Cách diễn giải này không hỗ trợ một claim tổng quát rằng độ phân giải không gian cao hơn có lợi trong các thành phố không đồng nhất. Thay vào đó, nó báo cáo mức tăng pooled nhỏ, tính bất biến chính xác nơi county grouping không tạo partition mới, và một mẫu hình dương mang tính mô tả trong subgroup multi-county đã đánh giá. Toán tử hiệu chỉnh chỉ tái phân bổ khối lượng luồng giữa các khoảng cách hoặc các lát origin-county; nó giữ nguyên thứ hạng tương đối của các cặp OD trong từng lát, nên độ chính xác tổng thể vẫn bị giới hạn bởi năng lực xếp hạng nội bộ của baseline.)*

*(Tiếng Việt: Các giới hạn chính gồm: (1) County boundaries là đơn vị hành chính, không được thiết kế như các lưu vực đi lại hoặc cộng đồng di chuyển chức năng; việc các vùng đô thị chức năng hoặc mobility communities có tạo ra ràng buộc tổng hợp nhiều thông tin hơn hay không cần nghiên cứu riêng. (2) County groups chỉ gồm các tract thuộc phạm vi dữ liệu city do Lab cung cấp, không biểu diễn toàn bộ nhu cầu di chuyển trên phạm vi county. (3) Các phân phối county trong benchmark được tạo như oracle aggregate observations từ OD reference matrices; kết quả không chứng minh mức tăng tương đương với telemetry thực tế có nhiễu hoặc không đầy đủ.)*

*(Tiếng Việt: Tóm lại, thí nghiệm county-level cung cấp một kết quả incremental pooled nhỏ và bằng chứng mô tả trong subgroup multi-county đã đánh giá. Nó gợi ý, nhưng không kiểm định, giả thuyết rằng các ràng buộc theo nhóm origin chi tiết hơn có thể hữu ích khi chúng mã hóa thông tin chưa được biểu diễn bởi một phân phối cấp city.)*

---

## 5.8 Methodological implications and deployment hypothesis
*(Tiếng Việt: **5.8. Ý nghĩa phương pháp luận và giả thuyết triển khai**)*

Neural mobility frameworks such as Deep Gravity and UGNN illustrate that deep neural networks can synthesize multifaceted geographic information to learn transferable spatial mobility laws [@simini2021deepgravity; @guo2025ugnn]. However, these architectures fundamentally require granular OD observations from source training regions to fit model parameters. The contribution of the present study is not to eliminate the necessity of OD training data, but rather to show that a pre-trained cross-city model can be adjusted at inference time using an aggregate observation of the target city without updating model parameters.

From a methodological perspective, the results show that an accurate target-domain aggregate constraint can adjust a frozen cross-city model at inference time without parameter fine-tuning or end-to-end retraining. This oracle experiment establishes the potential information value of the constraint; whether independently collected aggregate observations can provide comparable utility requires separate empirical validation.

The evaluated framework remains conditioned on the known positive support $\Omega_c^+$. Calibration reallocates predicted mass across distance bins without updating model parameters or creating new OD links. Accordingly, the present results do not establish full-matrix reconstruction capability or operational performance with independently collected telemetry.

*(Tiếng Việt: Các mô hình như Deep Gravity và UGNN cho thấy neural networks có thể kết hợp nhiều dạng thông tin địa lý để học các quy luật mobility có khả năng chuyển giao [@simini2021deepgravity; @guo2025ugnn]. Tuy nhiên, các mô hình này vẫn cần OD observations từ các khu vực nguồn để huấn luyện. Đóng góp của nghiên cứu hiện tại không phải loại bỏ nhu cầu về OD training data, mà là cho thấy một mô hình nguồn đã huấn luyện có thể được điều chỉnh tại inference time bằng một quan sát tổng hợp của thành phố mục tiêu mà không cần cập nhật tham số.)*

*(Tiếng Việt: Về mặt phương pháp, kết quả cho thấy một ràng buộc tổng hợp chính xác tại miền mục tiêu có thể điều chỉnh mô hình cross-city đã đóng băng ở thời điểm suy luận mà không cần fine-tuning tham số hoặc huấn luyện lại end-to-end. Thí nghiệm oracle này xác lập giá trị thông tin tiềm năng của ràng buộc; việc các quan sát tổng hợp thu thập độc lập có mang lại mức hữu ích tương đương hay không cần được kiểm chứng bằng thực nghiệm riêng.)*

*(Tiếng Việt: Framework được đánh giá vẫn có điều kiện trên tập hỗ trợ dương đã biết $\Omega_c^+$. Phép hiệu chỉnh tái phân bổ khối lượng dự báo giữa các khoảng cự ly mà không cập nhật tham số mô hình hoặc tạo liên kết OD mới. Vì vậy, các kết quả hiện tại không thiết lập khả năng tái tạo toàn bộ ma trận hoặc hiệu năng vận hành với telemetry được thu thập độc lập.)*

---

## 5.9 Limitations
*(Tiếng Việt: **5.9. Các giới hạn của nghiên cứu**)*

Several key scope boundaries and methodological limitations must be acknowledged:
1. **Conditioning on Known Positive Support ($\Omega_c^+$)**: The evaluation is conducted on observed positive interzonal pairs ($T_{ij} \ge 1, D_{ij} > 0$). The framework does not address link prediction or the zero-flow identification problem.
2. **One-Dimensional Constraint**: $Y_D$ constrains only scalar distance allocations; it provides no information regarding directional orientation, polycentric attraction hubs, or trip purposes.
3. **Data Quality, Coverage, and Representation**: Human mobility datasets frequently contain substantial coverage biases, representativeness issues, and data processing artifacts that can influence model conclusions [@gallotti2024distorted; @pappalardo2023future]. Our benchmark is evaluated across 50 U.S. metropolitan areas at the census tract level; generalization to international contexts with informal transit systems requires independent empirical validation.
4. **Privacy Scope Boundary**: Aggregating or reducing data resolution does not automatically guarantee formal privacy protection. Individual mobility traces can retain high re-identifiability even after coarse aggregation [@demontjoye2013unique], and providing user-level differential privacy guarantees for aggregate location data remains challenging in practice [@houssiau2022differential]. The present study does not perform a formal privacy analysis on $Y_D$; hence, $Y_D$ should be understood strictly as a low-dimensional aggregate observation, rather than a proven privacy-preserving mechanism.
5. **Synthetic Noise Assumptions**: Noise experiments use centered Gaussian directions in log-ratio space with exponential tilting to reach specified TV magnitudes. Real-world observation errors may exhibit structured demographic or geographic non-randomness not represented by this perturbation design.

*(Tiếng Việt: Mobility datasets có thể chứa sai lệch về độ phủ, tính đại diện và quy trình tiền xử lý [@gallotti2024distorted; @pappalardo2023future]. Ngoài ra, giảm độ phân giải hoặc tổng hợp dữ liệu không tự động tạo ra bảo đảm quyền riêng tư. Mobility traces vẫn có thể chứa thông tin nhận dạng đáng kể sau khi được làm thô [@demontjoye2013unique], và việc cung cấp bảo đảm differential privacy ở cấp người dùng cho dữ liệu vị trí tổng hợp vẫn gặp nhiều khó khăn thực tế [@houssiau2022differential]. Nghiên cứu hiện tại không thực hiện privacy analysis đối với $Y_D$; vì vậy, $Y_D$ chỉ nên được gọi là một quan sát tổng hợp có số chiều thấp, không phải một cơ chế privacy-preserving đã được chứng minh.)*

---

## 5.10 Future research directions
*(Tiếng Việt: **5.10. Các định hướng nghiên cứu tương lai**)*

1. **Multi-Constraint Aggregate Calibration**: A natural extension is coupling $Y_D$ with complementary low-dimensional constraints, such as total origin outflows ($\mathcal{O}_i$) or total destination inflows ($\mathcal{D}_j$). Classical spatial interaction modeling provides a rigorous foundation for simultaneously applying production, attraction, and impedance constraints [@wilson1971family; @ortuzar2011modelling].
2. **Coupling Mechanistic Principles with AI**: Combining mechanistic spatial interaction principles with deep transfer architectures represents an essential frontier for robust, interpretable human mobility modeling [@pappalardo2023future].
3. **Adaptive Target Diagnostics**: Developing pre-inference gating criteria to identify target cities with large initial distance mismatch ($d_{\text{pre}}$), selectively triggering calibration only when expected utility is high.
4. **End-to-End Joint Link-Intensity Modeling**: Coupling cross-city zero-shot link classification with support-conditioned intensity calibration to achieve full-matrix OD reconstruction.
5. **Cross-National Generalization**: Validating the calibration framework on international mobility datasets with diverse transit infrastructures and spatial administrative definitions.
6. **Real-World Aggregate Telemetry Exploration**: While the present study does not use external telemetry datasets, future research may evaluate independently sourced aggregate mobility products—including Meta Movement Distribution if its provenance, geographic units, access conditions, and fitness for this task are established—to assess transfer beyond synthetic noise models.

*(Tiếng Việt: Một hướng phát triển tự nhiên là kết hợp $Y_D$ với các ràng buộc tổng hợp khác, chẳng hạn tổng outflow theo origin hoặc tổng inflow theo destination. Các mô hình spatial interaction cổ điển cung cấp nền tảng cho việc áp dụng đồng thời các ràng buộc sản sinh, thu hút và impedance [@wilson1971family; @ortuzar2011modelling]. Các hướng nghiên cứu gần đây cũng nhấn mạnh giá trị của việc kết hợp mechanistic mobility models với các phương pháp học máy có khả năng mở rộng và diễn giải [@pappalardo2023future]. Future work có thể đánh giá các nguồn quan sát tổng hợp độc lập—bao gồm Meta Movement Distribution nếu provenance, đơn vị địa lý, điều kiện truy cập và mức độ phù hợp được xác lập—nhưng nghiên cứu hiện tại chưa sử dụng telemetry bên ngoài.)*

