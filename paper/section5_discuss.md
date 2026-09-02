# Section 5: Discussion

*(Tiếng Việt: **Mục 5: Thảo luận chuyên sâu**)*

In this section, we contextualize our findings within the broader literature on urban mobility modeling and transfer learning (Barbosa et al., 2018; Enaya et al., 2026; Lenormand et al., 2016; Simini et al., 2012, 2021). We examine the theoretical mechanisms underlying the information value of aggregate distance distributions, evaluate observational resolution and noise sensitivity under controlled synthetic perturbations, discuss methodological and practical implications for data-scarce urban analytics, and outline key limitations and future research directions.

*(Tiếng Việt: Trong phần này, chúng tôi đặt các phát hiện của nghiên cứu vào bức tranh tổng thể của các nghiên cứu về mô hình hóa di chuyển đô thị và học chuyển giao (Barbosa et al., 2018; Enaya et al., 2026; Lenormand et al., 2016; Simini et al., 2012, 2021). Chúng tôi phân tích các cơ chế lý thuyết giải thích giá trị thông tin của phân phối cự ly tổng hợp, đánh giá độ phân giải quan sát và độ nhạy đối với nhiễu tổng hợp có kiểm soát, thảo luận ý nghĩa phương pháp luận và thực tiễn cho phân tích đô thị khan hiếm dữ liệu, đồng thời nêu rõ các hạn chế chính và định hướng nghiên cứu tiếp theo.)*

---

## 5.1 Main findings and information value
*(Tiếng Việt: **5.1. Các phát hiện chính và giá trị thông tin**)*

Our empirical benchmark across 50 held-out U.S. metropolitan areas indicates that conditioning on the target city's distance-binned mobility distribution ($Y_D$) yields a **small but statistically significant and consistent improvement** over the zero-shot baseline ($M_0$). Across 5-fold cross-validation and three independent model initializations, city-level calibration increases the mean Common Part of Commuters from $0.71281$ to $0.71635$, corresponding to a mean gain of $\overline{\Delta\mathrm{CPC}} = +0.00354$ ($95\%\text{ CI: } [+0.0026, +0.0045]$, median $+0.00195$, paired Wilcoxon signed-rank test $W = 83.0, p = 1.93 \times 10^{-9}$). Crucially, positive gains occur in 45 of the 50 evaluated cities (a 90.0% directional win rate).

*(Tiếng Việt: Kết quả cho thấy phân phối khoảng cách của thành phố mục tiêu tạo ra mức cải thiện nhỏ nhưng có ý nghĩa thống kê và tương đối nhất quán trên các thành phố đánh giá. Mức tăng CPC trung bình $+0.00354$ không đại diện cho sự thay đổi lớn về độ chính xác tuyệt đối, nhưng cho thấy $\mathbf{Y}_{D,c}$ chứa thông tin bổ sung chưa được baseline zero-shot khai thác đầy đủ (khoảng tin cậy bootstrap 95%: $[+0.0026, +0.0045]$, trung vị $+0.00195$, kiểm định Wilcoxon ghép cặp $W = 83.0, p = 1.93 \times 10^{-9}$, thắng 45/50 thành phố).)*

However, the scientific interpretation of this result warrants careful calibration. The average magnitude of improvement ($\Delta\mathrm{CPC} \approx +0.0035$) is modest in absolute terms; $Y_D$ does not replace granular OD survey data or fundamentally transform baseline fidelity on its own. Rather, it indicates that low-dimensional distance distributions contain useful aggregate structure that pre-trained spatial neural networks cannot infer from cross-city priors and static geographic features alone.

Importantly, the target distribution $\mathbf{Y}_{D,c}$ in our benchmark is synthesized directly from reference OD flows as an **oracle aggregate observation**. Consequently, the current findings assess the potential information ceiling of an ideal, error-free distance distribution. They do not demonstrate real-world deployment performance with noisy, missing, or third-party empirical telemetry streams.

*(Tiếng Việt: Cần lưu ý rằng $\mathbf{Y}_{D,c}$ trong thí nghiệm được tổng hợp từ OD tham chiếu dưới dạng oracle aggregate observation. Vì vậy, kết quả hiện tại đánh giá giá trị thông tin tiềm năng của một phân phối khoảng cách chính xác, chứ chưa chứng minh hiệu quả triển khai với một nguồn quan sát bên ngoài có nhiễu hoặc thiếu dữ liệu.)*

---

## 5.2 Mechanistic explanation: Macro distance reallocation vs intra-bin ranking
*(Tiếng Việt: **5.2. Cơ chế giải thích: Tái phân bổ cự ly vĩ mô và thứ hạng nội khoảng**)*

The cross-city zero-shot baseline ($M_0$) predicts flow intensities using tract-level context features and inter-tract geographic distances, but cannot observe how total travel demand is partitioned across journey lengths in a specific unseen city. Two metropolitan areas with similar populations and built environments can exhibit markedly different trip length frequencies due to distinct spatial layouts, sub-center distributions, or transit networks (Barbosa et al., 2018; Simini et al., 2021). $Y_D$ supplies a target-specific constraint on this aggregate distance composition.

Our empirical diagnostics indicate a strong positive association between the baseline's initial distance-distribution mismatch ($d_{\text{pre}} = \mathrm{TV}(\hat{Y}_D^{M0}, Y_D)$) and subsequent calibration gain ($\Delta\mathrm{CPC}_c$). Even after controlling for baseline accuracy, network size, urban land area, and mean trip distance, the partial correlation remains high ($r_{\text{partial}} = +0.7951, p = 5.35 \times 10^{-12}$, multiple regression $R^2 = 73.7\%$). This pattern is consistent with an **inter-bin mass reallocation mechanism**, whereby calibration delivers larger gains when the baseline's macro distance profile deviates substantially from the target distribution. However, this statistical association represents observational correlation and does not, on its own, establish strict causality.

*(Tiếng Việt: Mối liên hệ dương mạnh giữa sai lệch phân phối trước hiệu chỉnh $d_{\mathrm{pre}}$ và $\Delta\mathrm{CPC}$, ngay cả sau khi kiểm soát một số đặc điểm thành phố, phù hợp với cơ chế tái phân bổ khối lượng liên khoảng ($r_{\text{partial}} = +0.7951, p = 5.35 \times 10^{-12}$, $R^2 = 73.7\%$). Kết quả này cho thấy hiệu chỉnh thường mang lại lợi ích lớn hơn khi phân phối khoảng cách của baseline khác nhiều hơn so với phân phối mục tiêu. Tuy nhiên, đây vẫn là bằng chứng liên hệ quan sát và không tự nó thiết lập quan hệ nhân quả.)*

Because the calibration operator multiplies all predictions within distance bin $b$ by a common positive scalar $s_{c,b} > 0$, **intra-bin pair rankings are mathematically invariant** (and Kendall's $\tau = 1.00000000$ for all non-degenerate groups with sufficient pairs). Empirically, intra-bin ranking quality ($Q_c^{\text{intra}}$) exhibits no significant monotonic correlation with calibration gain ($r = +0.046, p = 0.75$). This non-significant result does not demonstrate that intra-bin ranking is irrelevant; it simply indicates that our empirical sample provides no evidence of a linear relationship between baseline intra-bin ranking accuracy and macro calibration benefit. Ultimately, reconstructed flow quality remains bounded by the baseline's internal ranking capacity, as post-hoc distance scaling cannot correct misranked pair pairs within the same distance interval.

*(Tiếng Việt: Do tất cả cặp trong cùng một khoảng được nhân với cùng một hệ số dương, phép hiệu chỉnh bảo toàn thứ tự nội khoảng về mặt toán học. Phân tích thực nghiệm không phát hiện mối tương quan có ý nghĩa giữa chỉ số chất lượng nội khoảng $Q_c^{\mathrm{intra}}$ và mức cải thiện ($r=0.046, p=0.75$). Kết quả không có ý nghĩa thống kê này không chứng minh rằng chất lượng nội khoảng hoàn toàn không quan trọng; nó chỉ cho thấy dữ liệu hiện tại chưa cung cấp bằng chứng về một quan hệ đơn điệu giữa hai đại lượng. Chất lượng cuối cùng vẫn bị giới hạn bởi cấu trúc nội khoảng mà baseline đã dự báo, vì bước hiệu chỉnh không thể sửa thứ tự sai giữa các cặp thuộc cùng một nhóm.)*

---

## 5.3 Observational resolution and diminishing marginal returns
*(Tiếng Việt: **5.3. Độ phân giải thông tin và quy luật lợi suất giảm dần**)*

Experiments varying distance bin granularity ($K \in \{2, 4, \dots, 20\}$) demonstrate that calibration gain increases monotonically with distance resolution. Even at $K=2$, mean CPC improves by $+0.00098$, rising to $+0.00354$ at $K=8$ (canonical) and $+0.00639$ at $K=20$. However, marginal returns decline steadily: incremental gain per additional bin peaks at $K=4$ ($4.94 \times 10^{-4}/\text{bin}$) and decreases to $3.19 \times 10^{-4}/\text{bin}$ at $K=20$. Coarse initial partitions separate fundamental mobility regimes (e.g., short local vs long inter-district travel), whereas finer intervals impose localized constraints that become progressively sparser.

Even at $K=20$, the aggregate observation represents a very small dimensionality relative to the number of positive OD pairs ($K / |\Omega_c^+| < 0.1\%$, averaging $\approx 1,757$ positive pairs per bin). This demonstrates that low-dimensional representations can provide structural guidance without requiring individual OD flow intensities during calibration. However, high aggregation does not automatically guarantee privacy. Formal differential privacy guarantees or re-identification risk analyses (e.g., de Montjoye et al., 2013) fall outside the scope of the present study.

*(Tiếng Việt: Ngay cả khi $K=20$, số giá trị tổng hợp vẫn rất nhỏ so với số cặp OD dương ($K / |\Omega_c^+| < 0.1\%$). Điều này cho thấy một biểu diễn có số chiều thấp có thể cung cấp hướng dẫn về cấu trúc khoảng cách mà không cần truyền từng cường độ OD vào bước hiệu chỉnh. Tuy nhiên, mức độ tổng hợp cao không tự động bảo đảm quyền riêng tư. Việc đánh giá rủi ro tái nhận dạng hoặc cung cấp bảo đảm differential privacy nằm ngoài phạm vi nghiên cứu hiện tại.)*

---

## 5.4 Spatial semantic ordering and synthetic noise breakdown
*(Tiếng Việt: **5.4. Tính đúng thứ tự không gian và ngưỡng phá vỡ do nhiễu**)*

The utility of $Y_D$ is strictly contingent upon its spatial distance semantics. Randomly permuting the bin order of $Y_D$ while preserving its numerical values causes severe performance degradation ($\Delta\mathrm{CPC} = -0.00696$, a deficit of $0.01050$ compared to true calibration, $p < 10^{-14}$). This confirms that the observed benefit is not an artifact of generic output variance reduction or smoothing, but relies on binding specific mobility proportions to their true physical distance intervals.

Under synthetic Total Variation noise ($\epsilon \in [0\%, 5\%]$), calibration gains degrade monotonically, crossing zero at:

$$\epsilon_{\text{cross}} \approx 4.45\% \quad [95\%\text{ CI: } 4.16\%, 4.77\%]$$

This value represents an **empirical crossover threshold under the specific synthetic perturbation protocol** employed here, and is conditioned on our benchmark dataset, noise model, $K=8$ configuration, and neural baseline. It is not an operational guarantee that transfers directly to real-world telecom data, survey streams, or out-of-sample metropolitan systems.

*(Tiếng Việt: Trong mô hình nhiễu tổng hợp được sử dụng, lợi ích trung bình chuyển qua 0 tại mức Total Variation error xấp xỉ $4.45\%$ (khoảng tin cậy 95%: $[4.16\%, 4.77\%]$). Đây là ngưỡng chuyển tiếp thực nghiệm phụ thuộc vào benchmark, phương pháp tạo nhiễu, cấu hình $K$ và baseline hiện tại. Nó không phải là một ngưỡng bảo đảm có thể áp dụng trực tiếp cho dữ liệu viễn thông, khảo sát hoặc các thành phố ngoài mẫu.)*

---

## 5.5 Target specificity vs generic distance decay priors
*(Tiếng Việt: **5.5. Tính đặc thù mục tiêu và các prior suy giảm cự ly phổ quát**)*

A central question in spatial transfer learning is whether target observations convey city-specific idiosyncrasies or merely reflect universal distance decay principles (Flowerdew & Aitkin, 1982; Zipf, 1946). Our dose-matched placebo benchmarks evaluate this distinction:
1. **Dose-Matched Wrong Donors**: Applying donor distributions from incorrect cities scaled to the target's intervention dose ($D_T$) produces no systematic gain ($\Delta\mathrm{CPC} = -0.000091, p = 0.4097$). The true target distribution outperforms dose-matched wrong donors in 46 of 50 cities ($+0.003630, p = 2.19 \times 10^{-11}$).
2. **Dose-Matched Training-Mean**: Applying the mean distance profile across training cities yields a marginal change of $+0.000914$, which is **statistically indistinguishable from zero** ($p = 0.4319$). The target-specific distribution outperforms the training-mean profile in 47 of 50 cities ($+0.002626, p = 4.03 \times 10^{-11}$).

Because the baseline model already incorporates pairwise **Haversine distances** between tracts alongside urban context features, these results demonstrate that static geographic distances and cross-city priors do not fully account for target-city travel distance composition. This supports the city-specific informational value of $Y_D$, while not implying that geographic distance in general is insufficient for spatial mobility modeling.

*(Tiếng Việt: Baseline đã sử dụng khoảng cách Haversine giữa các tract cùng với các đặc trưng đô thị và biểu diễn học được từ training cities. Việc target-specific $Y_D$ vượt trội hơn wrong-donor ($\Delta = -0.000091, p=0.4097$) và training-mean observations ($\Delta = +0.000914, p=0.4319$, không phân biệt được với 0) cho thấy các đầu vào này chưa giải thích đầy đủ thành phần khoảng cách của di chuyển tại thành phố mục tiêu. Kết quả hỗ trợ tính đặc thù theo thành phố của $Y_D$, nhưng không chứng minh rằng khoảng cách địa lý nói chung là không đủ cho mọi mô hình di chuyển.)*

---

## 5.6 Inter-city performance heterogeneity
*(Tiếng Việt: **5.6. Sự không đồng nhất về hiệu quả giữa các thành phố**)*

Although 45 of 50 cities exhibit positive gains, performance varies across metropolitan areas, with 5 cities exhibiting negative changes (El Paso, Oklahoma City, Jacksonville, Louisville, Long Beach). This heterogeneity aligns with our mechanistic framework:
- **Low Baseline Misalignment**: In cities where the zero-shot baseline already matches the target distance profile ($d_{\text{pre}} \approx 0$), there is minimal room for macro reallocation.
- **Intra-Bin Error Dominance**: In cities where baseline errors stem primarily from misallocating flows among zone pairs within the same distance band rather than across bands, scalar distance calibration cannot rectify the underlying distortion.

Consequently, $Y_D$ calibration should be viewed as a conditioned post-processing tool whose efficacy depends on baseline macro alignment, rather than an unconditional guarantee of improvement.

*(Tiếng Việt: Mặc dù 45/50 thành phố có $\Delta\mathrm{CPC} > 0$, mức độ cải thiện có sự phân hóa và có 5 thành phố ghi nhận $\Delta\text{CPC} < 0$ (El Paso, Oklahoma City, Jacksonville, Louisville, Long Beach). Sự không đồng nhất này hoàn toàn nhất quán với khung cơ chế $d_{\text{pre}}$: ở các thành phố có baseline đã khớp phân phối cự ly ($d_{\text{pre}} \approx 0$) hoặc nơi sai số chủ yếu nằm ở thứ hạng nội khoảng, hiệu chỉnh khoảng cách một chiều không thể đem lại mức tăng lớn.)*

---

## 5.7 Spatial resolution depends on intra-urban structure
*(Tiếng Việt: **5.7. Giá trị của độ phân giải không gian phụ thuộc vào cấu trúc nội tại của thành phố**)*

The spatial resolution experiment clarifies that the utility of $Y_D$ depends not only on distance bin granularity, but also on the spatial aggregation scale. However, higher spatial resolution does not automatically deliver substantial gains; benefits emerge only when spatial partitioning provides distinct distributional information compared to the city-wide signal.

In the city-level configuration (`M1_city`), a single vector $\mathbf{Y}_{D,c}$ modulates flow mass across distance intervals for the entire metropolis. This operator effectively rectifies average distance decay biases in the baseline, but applies an identical set of scaling multipliers to all origin tracts. Consequently, it cannot accommodate settings where distinct subregions within the same urban area exhibit markedly different distance distributions.

County-level observations (`M1_county`) introduce this localized capacity by supplying a separate distribution for each origin-county group. If the baseline overpredicts long-distance journeys in one county but underpredicts them in another, a single city-wide distribution may allow these localized errors to cancel out in the aggregate sum. In contrast, county-level calibration scales each origin group independently before assembling the unified city-wide prediction. This provides a coherent mechanistic explanation for the localized improvements observed in Kansas City ($+0.0027$), New York ($+0.0021$), and Dallas ($+0.0011$).

Nevertheless, the average gain across the full benchmark remains modest ($+0.00014$). This primarily stems from dataset structure: for the 39 single-county cities, county-level calibration is mathematically identical to city-level calibration and produces zero structural difference ($\Delta\operatorname{CPC}_{\mathrm{res}} \equiv 0$). Even within the 11 multi-county cities, the average incremental gain is $+0.00063$ CPC ($81.8\%$ win rate, 9 of 11 cities). Thus, the evidence does not support a sweeping claim that refining resolution from city to county always delivers large benefits. Instead, it indicates that the marginal value of higher spatial resolution is conditional and concentrated in cities characterized by pronounced intra-urban spatial heterogeneity across origin zones.

This finding also illuminates why $Y_D$ yields focused rather than transformative improvements over the baseline. The calibration operator only reallocates flow mass between distance intervals or origin-county slices; it leaves the relative ordering of OD pairs within each slice strictly invariant. Consequently, overall accuracy remains bounded by the baseline's capacity to rank zone pairs internally. If the baseline misranks pairs within the same distance band, providing finer county-level distributions cannot directly correct that intra-bin distortion.

Three explicit limitations warrant consideration:
1. **Administrative vs Functional Zoning**: County boundaries represent political-administrative jurisdictions rather than functional commuting basins or travel communities (Lenormand et al., 2016). Partitioning based on functional urban zones or mobility communities might achieve greater within-zone behavioral homogeneity, a hypothesis reserved for future inquiry.
2. **Dataset Footprint Boundary**: County groups comprise only those tracts included within the study city dataset provided by the laboratory, and do not represent total county-wide travel demand extending beyond the study area.
3. **Oracle Aggregate Setting**: County distributions in our benchmark are derived as oracle aggregate observations from reference OD matrices. These results demonstrate the theoretical information ceiling of county-level granularity, but do not prove that equivalent gains would materialize under noisy or incomplete real-world telemetry.

In summary, this experiment enriches RQ2 with a nuanced conclusion: refining the spatial resolution of $Y_D$ provides supplementary predictive information when a metropolis encompasses multiple origin zones with heterogeneous distance decay patterns, but average gains are modest and vanish when spatial partitioning yields no new distributional variance.

*(Tiếng Việt: Kết quả về county-level observation làm rõ rằng giá trị của $Y_D$ không chỉ phụ thuộc vào độ chi tiết của các khoảng cách mà còn phụ thuộc vào độ phân giải không gian tại đó phân phối được quan sát. Tuy nhiên, độ phân giải cao hơn không tự động dẫn đến cải thiện lớn hơn. Lợi ích chỉ xuất hiện khi việc chia nhỏ không gian thực sự cung cấp thông tin khác với phân phối cấp city. Trong cấu hình city-level, một vector $\mathbf{Y}_{D,c}$ duy nhất điều chỉnh khối lượng luồng giữa các khoảng cách cho toàn thành phố. Cách hiệu chỉnh này có thể sửa sai lệch khoảng cách trung bình của baseline, nhưng áp dụng cùng một tập trọng số cho tất cả origin tract. Vì vậy, nó không thể biểu diễn trường hợp các khu vực xuất phát khác nhau trong cùng thành phố có các phân phối khoảng cách khác nhau. County-level observation bổ sung khả năng này bằng cách sử dụng một phân phối riêng cho từng nhóm origin-county. Nếu baseline dự báo quá nhiều chuyến đi xa ở một county nhưng quá ít chuyến đi xa ở một county khác, một phân phối city-level có thể làm hai sai lệch này triệt tiêu khi tổng hợp. Ngược lại, county-level calibration có thể điều chỉnh từng nhóm riêng trước khi ghép chúng thành dự báo toàn thành phố. Đây là cơ chế hợp lý giải thích mức cải thiện quan sát được tại Kansas City, New York và Dallas. Tuy nhiên, mức tăng trung bình trên toàn bộ benchmark rất nhỏ. Điều này trước hết xuất phát từ cấu trúc dữ liệu: đối với 39 single-county cities, county-level calibration hoàn toàn tương đương city-level calibration và không thể tạo ra thay đổi ($\Delta\operatorname{CPC}_{\mathrm{res}}\equiv 0$). Ngay cả trong 11 multi-county cities, mức cải thiện trung bình cũng chỉ khoảng $+0.00063$ CPC ($81.8\%$ win rate). Do đó, kết quả không hỗ trợ nhận định rằng tăng độ phân giải từ city lên county luôn mang lại lợi ích đáng kể. Thay vào đó, nó cho thấy giá trị biên của độ phân giải cao hơn có tính điều kiện và tập trung ở một số thành phố có cấu trúc di chuyển không đồng nhất giữa các khu vực xuất phát. Kết quả này cũng giúp giải thích vì sao $Y_D$ chỉ tạo ra mức cải thiện nhỏ so với baseline. Phép hiệu chỉnh chỉ thay đổi tổng khối lượng giữa các khoảng cách hoặc giữa các nhóm county–distance; nó không thay đổi thứ tự tương đối của các cặp OD trong cùng một nhóm. Vì vậy, hiệu quả cuối cùng vẫn phụ thuộc vào việc baseline đã học được cấu trúc nội khoảng hữu ích đến mức nào. Nếu baseline xếp hạng sai các cặp trong cùng một khoảng, việc cung cấp phân phối ở cấp county không thể trực tiếp sửa sai lệch đó. Có ba giới hạn cần lưu ý: (1) County là đơn vị hành chính và không nhất thiết tương ứng với ranh giới chức năng của hành vi di chuyển; (2) Các nhóm county chỉ bao gồm những tract nằm trong bộ dữ liệu city do Lab cung cấp, không đại diện cho toàn bộ hoạt động bên ngoài phạm vi nghiên cứu; (3) Các phân phối county-level hiện được trích xuất dưới dạng oracle aggregate observations, kết quả do đó đánh giá giá trị thông tin tiềm năng chứ chưa chứng minh với dữ liệu quan sát thực tế có nhiễu. Nhìn chung, thí nghiệm này bổ sung cho RQ2 bằng một kết luận có giới hạn: tăng độ phân giải không gian của $Y_D$ có thể mang lại lợi ích bổ sung khi thành phố chứa nhiều vùng xuất phát với các mẫu khoảng cách khác nhau, nhưng lợi ích trung bình nhỏ và không tồn tại trong trường hợp việc phân nhóm không tạo ra thông tin mới.)*

---

## 5.8 Methodological and practical implications
*(Tiếng Việt: **5.8. Ý nghĩa phương pháp luận và thực tiễn**)*

From a **methodological** perspective, our findings illustrate that pre-trained cross-city models can benefit from target-domain observations without requiring parameter fine-tuning or end-to-end retraining. Decoupling spatial representation learning (performed globally across training cities) from post-hoc macro calibration (performed analytically at test time) offers a modular approach to cross-city flow prediction.

From a **practical** standpoint, the framework demonstrates that low-dimensional aggregate observations can be used to adjust cross-city predictions without transmitting individual OD flow values or updating model parameters in the target city. However, the current framework assumes that the target city's positive support $\Omega_c^+$ is already established. Consequently, it does not replace comprehensive travel surveys and does not address the full zero-filling problem in cities with entirely unobserved link topologies.

*(Tiếng Việt: Về mặt thực tiễn, phương pháp cho thấy một quan sát tổng hợp có số chiều thấp có thể được dùng để hiệu chỉnh một mô hình cross-city mà không cần cập nhật tham số tại thành phố mục tiêu. Tuy nhiên, framework hiện tại vẫn giả định rằng tập positive support $\Omega_c^+$ đã được biết. Vì vậy, nó chưa thay thế được khảo sát OD và chưa giải quyết bài toán tái tạo ma trận đầy đủ ở những thành phố hoàn toàn không có thông tin về liên kết OD.)*

---

## 5.9 Limitations
*(Tiếng Việt: **5.9. Các giới hạn của nghiên cứu**)*

Several key scope boundaries and methodological limitations must be acknowledged:
1. **Conditioning on Known Positive Support ($\Omega_c^+$)**: The evaluation is conducted on observed positive interzonal pairs ($T_{ij} \ge 1, D_{ij} > 0$). The framework does not address link prediction or the zero-flow identification problem.
2. **One-Dimensional Constraint**: $Y_D$ constrains only scalar distance allocations; it provides no information regarding directional orientation, polycentric attraction hubs, or trip purposes.
3. **Geographic Scope**: Experiments are evaluated on 50 U.S. metropolitan areas at the census tract level. Extrapolation to international regions with informal transit or different spatial zoning systems requires independent empirical validation.
4. **Synthetic Noise Assumptions**: Noise experiments utilize synthetic Dirichlet perturbations; real-world survey errors may exhibit structured demographic or geographic non-randomness.

*(Tiếng Việt: Nghiên cứu có một số giới hạn phạm vi cần được lưu ý: (1) Đánh giá thực hiện trên tập support dương đã biết ($\Omega_c^+$: $T_{ij} \ge 1, D_{ij} > 0$), chưa giải quyết bài toán phát hiện liên kết hay phân loại ô bằng 0; (2) $Y_D$ chỉ ràng buộc cự ly vô hướng, không cung cấp thông tin về hướng tuyến, các cực thu hút đa trung tâm hay mục đích chuyến đi; (3) Phạm vi đánh giá trên 50 vùng đô thị Hoa Kỳ ở cấp census tract, cần kiểm chứng độc lập khi ngoại suy sang các quốc gia có hệ thống giao thông và phân vùng khác biệt; (4) Thí nghiệm nhiễu sử dụng mô hình Dirichlet tổng hợp, trong khi sai số dữ liệu thực tế có thể mang tính thiên lệch nhân khẩu học hoặc địa lý.)*

---

## 5.10 Future research directions
*(Tiếng Việt: **5.10. Các định hướng nghiên cứu tương lai**)*

1. **Multi-Constraint Aggregate Calibration**: Integrating $Y_D$ with complementary low-dimensional observations, such as zone-level trip generations ($\mathcal{O}_i$) and attractions ($\mathcal{D}_j$), temporal flow distributions, or trip-purpose proportions via entropic optimal transport or iterative proportional fitting.
2. **Adaptive Target Diagnostics**: Developing pre-inference gating criteria to identify target cities with large $d_{\text{pre}}$, selectively triggering calibration only when expected utility is high.
3. **End-to-End Joint Link-Intensity Modeling**: Coupling cross-city zero-shot link classification with support-conditioned intensity calibration to achieve full-matrix OD reconstruction.
4. **Cross-National Generalization**: Validating the calibration framework on international mobility datasets with diverse transit infrastructures and spatial administrative definitions.
5. **Real-World Aggregate Telemetry Exploration**: Evaluating calibration on real-world public aggregate mobility observations (such as Meta Movement Range distributions or carrier-aggregated trip statistics) to assess domain transfer beyond synthetic noise models.

*(Tiếng Việt: Các hướng nghiên cứu tiếp theo bao gồm: (1) Hiệu chỉnh tổng hợp đa ràng buộc: kết hợp $Y_D$ với các thống kê biên như lượng phát sinh ($\mathcal{O}_i$), lượng thu hút ($\mathcal{D}_j$), phân phối thời gian hoặc mục đích chuyến đi thông qua tối ưu hóa vận chuyển hoặc IPF; (2) Bộ chẩn đoán thích ứng: xây dựng tiêu chí sàng lọc trước suy luận để nhận diện các thành phố có $d_{\text{pre}}$ lớn, chỉ kích hoạt hiệu chỉnh khi kỳ vọng lợi ích cao; (3) Mô hình hóa liên kết - cường độ đầu cuối: kết hợp phân loại liên kết zero-shot với hiệu chỉnh cường độ để tái tạo ma trận OD toàn phần; (4) Khái quát hóa quốc tế: kiểm chứng khung hiệu chỉnh trên các tập dữ liệu di chuyển quốc tế với hạ tầng giao thông và phân vùng hành chính đa dạng; (5) Khảo sát các nguồn viễn thông tổng hợp thực tế: đánh giá hiệu chỉnh trên các nguồn quan sát di chuyển tổng hợp công khai thực tế (chẳng hạn như phân phối Meta Movement Range hoặc thống kê tổng hợp từ nhà mạng viễn thông) để kiểm chứng khả năng chuyển giao thực tế ngoài mô hình nhiễu mô phỏng.)*

---

## 5.11 Conclusion of discussion
*(Tiếng Việt: **5.11. Kết luận phần thảo luận**)*

In summary, the target city's distance-binned mobility distribution provides a small, statistically significant, and consistent source of complementary information for zero-shot OD intensity reconstruction on known positive support. The observed benefit is consistent with an inter-bin mass reallocation mechanism, strictly requires target-specific spatial distance ordering, and degrades gracefully under synthetic observation noise. These results establish empirical evidence for combining aggregate observations with a frozen cross-city neural model, while not extending to link discovery, full-matrix OD reconstruction, or deployment with noisy real-world telemetry streams.

*(Tiếng Việt: Tóm lại, phân phối khoảng cách của thành phố mục tiêu cung cấp một nguồn thông tin bổ sung nhỏ nhưng có ý nghĩa và tương đối nhất quán cho tái tạo cường độ OD zero-shot trên positive support đã biết. Lợi ích quan sát được phù hợp với cơ chế sửa sai lệch phân bổ khối lượng giữa các khoảng, phụ thuộc vào việc sử dụng đúng phân phối của thành phố mục tiêu và suy giảm khi quan sát bị nhiễu. Kết quả thiết lập một bằng chứng thực nghiệm cho việc kết hợp quan sát tổng hợp với một mô hình cross-city đóng băng, đồng thời chưa mở rộng sang phát hiện liên kết, tái tạo ma trận OD đầy đủ hoặc triển khai với nguồn dữ liệu tổng hợp thực tế.)*
