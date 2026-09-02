# Section 5: Discussion

*(Tiếng Việt: **Mục 5: Thảo luận chuyên sâu**)*

In this section, we contextualize our findings within the broader literature on urban mobility modeling and transfer learning. We examine the theoretical mechanisms underlying the information value of aggregate distance distributions, evaluate the operational boundaries and robustness of target-conditioned inference, discuss methodological and practical implications for data-scarce urban analytics, and outline key limitations and future research directions.

*(Tiếng Việt: Trong phần này, chúng tôi đặt các phát hiện của nghiên cứu vào bức tranh tổng thể của các nghiên cứu về mô hình hóa di chuyển đô thị và học chuyển giao. Chúng tôi phân tích các cơ chế lý thuyết giải thích giá trị thông tin của phân phối cự ly tổng hợp, đánh giá các ranh giới vận hành và độ bền vững của suy luận có điều kiện mục tiêu, thảo luận ý nghĩa phương pháp luận và thực tiễn cho phân tích đô thị khan hiếm dữ liệu, đồng thời nêu rõ các hạn chế chính và định hướng nghiên cứu tiếp theo.)*

---

## 5.1 Main findings and information value
*(Tiếng Việt: **5.1. Các phát hiện chính và giá trị thông tin**)*

This study investigated whether a low-dimensional aggregate observation—the target-city trip distance distribution ($Y_D$)—can improve zero-shot origin–destination (OD) flow intensity reconstruction from a frozen cross-city neural model. Our empirical benchmark demonstrates that target-distance calibration yields a statistically overwhelming, systematic improvement in Common Part of Commuters across 50 held-out U.S. metropolitan areas ($\Delta\mathrm{CPC} = +0.00354$ [95% CI: $+0.0026, +0.0045]$, median $+0.00195$, two-sided Wilcoxon $W = 83.0, p = 1.93 \times 10^{-9}$). Crucially, positive gains are observed in 45 of 50 evaluated cities (a 90.0% win rate), establishing that the benefit of $Y_D$ is pervasive across highly diverse urban morphologies.

*(Tiếng Việt: Nghiên cứu này xem xét liệu một quan sát tổng hợp có số chiều thấp—phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu, ký hiệu là $Y_D$—có thể cải thiện kết quả tái tạo cường độ luồng OD từ một mô hình zero-shot đã được huấn luyện trên các thành phố khác hay không. Kết quả thực nghiệm chuẩn cho thấy việc sử dụng $Y_D$ tạo ra mức cải thiện CPC có ý nghĩa thống kê vượt trội trên 50 thành phố Hoa Kỳ được đánh giá ($\Delta\mathrm{CPC} = +0.00354$ [khoảng tin cậy 95%: $+0.0026, +0.0045$], trung vị $+0.00195$, kiểm định Wilcoxon hai phía $W = 83.0, p = 1.93 \times 10^{-9}$). Quan trọng hơn, mức cải thiện dương xuất hiện ở 45 trong tổng số 50 thành phố (tỷ lệ thắng 90.0%), khẳng định lợi ích của $Y_D$ có tính bao quát trên nhiều dạng hình thái đô thị đa dạng.)*

These findings positively answer our primary research question (RQ1): target-specific aggregate distance observations provide substantial incremental information beyond what can be inferred purely from static urban features and inter-zone geometric distances. However, the scientific interpretation of this result warrants careful calibration. The absolute magnitude of the CPC gain is modest ($\approx +0.0035$), meaning that $Y_D$ does not replace granular OD survey data or fundamentally transform baseline fidelity on its own. The primary value of $Y_D$ lies in its remarkable consistency and zero-training parameter footprint, offering a lightweight, cost-effective inference-time correction.

*(Tiếng Việt: Các kết quả này trả lời tích cực cho câu hỏi nghiên cứu chính (RQ1): phân phối cự ly tổng hợp của thành phố mục tiêu cung cấp thông tin bổ sung có giá trị vượt ra ngoài những gì có thể suy luận thuần túy từ đặc trưng đô thị tĩnh và khoảng cách hình học giữa các vùng. Tuy nhiên, ý nghĩa khoa học của phát hiện này cần được diễn giải thận trọng. Mức tăng CPC tuyệt đối ở quy mô khiêm tốn ($\approx +0.0035$), cho thấy $Y_D$ không thể thay thế dữ liệu khảo sát OD chi tiết hay làm thay đổi hoàn toàn chất lượng dự báo ban đầu. Giá trị cốt lõi của $Y_D$ nằm ở tính nhất quán cao trên diện rộng cùng việc không đòi hỏi huấn luyện thêm tham số nào tại thời điểm suy luận.)*

---

## 5.2 Mechanistic explanation: Correcting macro distance misalignment ($d_{\text{pre}}$)
*(Tiếng Việt: **5.2. Cơ chế giải thích: Nắn chỉnh sự lệch pha phân bổ cự ly vĩ mô ($d_{\text{pre}}$)**)*

While the cross-city zero-shot baseline ($M_0$) utilizes zone features and pairwise geometric distance to predict flows, it cannot directly observe how actual trip volumes in a specific target city are partitioned across short, medium, and long journeys. Two cities with comparable density and population may exhibit markedly different distance decay characteristics due to polycentric employment hubs, natural barriers, transit networks, or administrative boundaries. $Y_D$ introduces an explicit, target-specific constraint on the distance composition of mobility demand.

*(Tiếng Việt: Mô hình zero-shot $M_0$ đã sử dụng các đặc trưng phân vùng và khoảng cách hình học để dự báo luồng, nhưng không thể trực tiếp quan sát tổng lượng di chuyển thực tế của thành phố mục tiêu được phân bổ như thế nào giữa các dải cự ly ngắn, trung bình và dài. Hai thành phố có mật độ và dân số tương đồng vẫn có thể có đặc tính suy giảm cự ly rất khác biệt do cấu trúc việc làm đa trung tâm, rào cản tự nhiên, mạng lưới giao thông công cộng hoặc ranh giới hành chính. $Y_D$ bổ sung một ràng buộc ngoại sinh trực tiếp về thành phần khoảng cách của nhu cầu di chuyển.)*

Mechanistically, the calibration operator acts as an inter-bin mass reallocation instrument. If the zero-shot baseline systematically overpredicts long-distance flows and underpredicts local trips, the bin-specific scaling factor $s_k$ scales down the former and scales up the latter while strictly conserving total flow. Our empirical diagnostic confirms that the performance gain $\Delta\mathrm{CPC}_c$ is overwhelmingly explained by the baseline's initial distance-distribution mismatch:

$$d_{\text{pre}} = \mathrm{TV}\left(\hat{Y}_D^{M0}, Y_D\right) = \frac{1}{2}\sum_{k=1}^K \left|\hat{Y}_k^{(0)} - Y_k\right|$$

Controlling for baseline CPC, network size, urban land area, and mean trip distance, the partial correlation between $d_{\text{pre}}$ and $\Delta\mathrm{CPC}_c$ reaches $r_{\text{partial}} = +0.7951$ ($p = 5.35 \times 10^{-12}$, multiple regression $R^2 = 73.7\%$). This establishes that **the calibration operator helps most where the zero-shot baseline's macro distance allocation is most severely distorted** (such as Los Angeles, Phoenix, and Houston).

*(Tiếng Việt: Về mặt cơ chế, toán tử hiệu chỉnh hoạt động như một công cụ tái phân bổ khối lượng liên khoảng. Khi mô hình baseline dự báo thừa luồng cự ly dài và thiếu luồng cự ly ngắn, hệ số tỷ lệ $s_k$ sẽ giảm tải khoảng thứ nhất và tăng cường khoảng thứ hai nhưng vẫn bảo toàn tuyệt đối tổng lưu lượng. Phân tích chẩn đoán xác nhận mức cải thiện $\Delta\mathrm{CPC}_c$ được giải thích chủ yếu bởi mức độ sai lệch cự ly ban đầu của baseline $d_{\text{pre}} = \mathrm{TV}(\hat{Y}_D^{M0}, Y_D)$. Khi kiểm soát độ chính xác ban đầu, quy mô mạng lưới, diện tích và cự ly trung bình, hệ số tương quan riêng phần đạt $r_{\text{partial}} = +0.7951$ ($p = 5.35 \times 10^{-12}$, $R^2 = 73.7\%$). Điều này chứng minh rằng toán tử hiệu chỉnh đem lại lợi ích lớn nhất tại các thành phố mà mô hình baseline bị méo mó nghiêm trọng nhất về phân bổ cự ly vĩ mô.)*

Crucially, because intensity scaling applies a positive scalar $s_k > 0$ uniformly within each bin, the relative ranking and ratio of OD pairs within the same distance bin remain mathematically invariant (Kendall's $\tau = 1.00000000$). Empirically, intra-bin ranking fidelity ($Q_c^{\text{intra}}$) exhibits virtually zero correlation with gain ($r = +0.046, p = 0.75$). This refutes the notion that calibration requires high baseline intra-bin ranking accuracy, confirming that the operator functions strictly as an inter-bin macro-correction mechanism.

*(Tiếng Việt: Đáng chú ý, do việc hiệu chỉnh áp dụng một hệ số dương $s_k > 0$ đồng nhất cho toàn bộ các cặp trong cùng một khoảng, thứ tự xếp hạng và tỷ lệ tương đối giữa các cặp OD bên trong cùng khoảng cự ly được bảo toàn tuyệt đối về mặt toán học (Kendall's $\tau = 1.0$). Thực nghiệm cũng chỉ ra rằng chất lượng xếp hạng nội khoảng ($Q_c^{\text{intra}}$) không có tương quan với mức cải thiện ($r = +0.046, p = 0.75$). Kết quả này bác bỏ giả thuyết rằng hiệu chỉnh đòi hỏi độ chính xác xếp hạng nội khoảng cao, và khẳng định toán tử hoạt động thuần túy như một cơ chế nắn chỉnh vĩ mô liên khoảng.)*

---

## 5.3 Information resolution and diminishing marginal returns
*(Tiếng Việt: **5.3. Độ phân giải thông tin và quy luật lợi suất giảm dần**)*

Experiments varying partition granularity ($K \in \{2, 4, \dots, 20\}$) demonstrate that the information utility of $Y_D$ scales monotonically with distance resolution. Even at the coarsest binary partition ($K=2$), calibration improves mean CPC by $+0.00098$ with positive gains in 39 of 50 cities. As granularity increases to $K=8$ (canonical) and $K=20$, the mean gain rises to $+0.00354$ and $+0.00639$, respectively.

*(Tiếng Việt: Các thí nghiệm thay đổi độ phân giải phân vùng $K \in \{2, 4, \dots, 20\}$ cho thấy giá trị thông tin của $Y_D$ tăng đơn điệu theo độ chi tiết cự ly. Ngay tại phân vùng nhị phân thô nhất ($K=2$), hiệu chỉnh đã mang lại mức tăng $+0.00098$ với 39/50 thành phố cải thiện. Khi độ phân giải tăng lên $K=8$ (chuẩn) và $K=20$, mức tăng trung bình đạt lần lượt $+0.00354$ và $+0.00639$.)*

However, the marginal gain profile reveals a clear pattern of diminishing returns. Marginal gain per additional distance bin peaks at $K=4$ ($4.94 \times 10^{-4}/\text{bin}$) and steadily declines to $3.19 \times 10^{-4}/\text{bin}$ at $K=20$. Coarse initial bins separate fundamentally distinct travel regimes (intra-neighborhood vs inter-district trips). Finer partitions introduce increasingly localized constraints where bin volumes become sparser and more vulnerable to empirical sampling variance.

*(Tiếng Việt: Tuy nhiên, phân tích lợi ích biên cho thấy quy luật lợi suất giảm dần rõ nét. Mức tăng CPC trên mỗi khoảng bổ sung đạt đỉnh ở $K=4$ ($4.94 \times 10^{-4}/\text{bin}$) và giảm dần xuống $3.19 \times 10^{-4}/\text{bin}$ ở $K=20$. Các khoảng thô ban đầu giúp tách biệt các hình thái di chuyển cơ bản (nội khu vs liên quận), trong khi việc chia nhỏ hơn bổ sung các ràng buộc cục bộ với dung lượng mẫu thưa hơn và nhạy cảm hơn với phương sai lấy mẫu.)*

Importantly, even at $K=20$, the spatial aggregation ratio remains exceptionally small ($K / |\Omega_c^+| < 0.1\%$, averaging $\approx 1,757$ positive OD pairs per bin). This proves that significant macro-structural guidance can be extracted from highly compact, privacy-preserving summary statistics without requiring dense interaction observations.

*(Tiếng Việt: Quan trọng là ngay cả tại $K=20$, tỷ lệ tổng hợp không gian vẫn ở mức cực kỳ thấp ($K / |\Omega_c^+| < 0.1\%$, trung bình khoảng 1.757 cặp OD dương trên mỗi bin). Điều này chứng minh rằng các ràng buộc vĩ mô có giá trị cao có thể được khai thác hiệu quả từ các thống kê tóm tắt gọn nhẹ, bảo vệ quyền riêng tư mà không cần dữ liệu tương tác dày đặc.)*

---

## 5.4 Semantic ordering and noise breakdown threshold
*(Tiếng Việt: **5.4. Tính đúng thứ tự không gian và ngưỡng phá vỡ do nhiễu**)*

The utility of $Y_D$ is strictly bound to its spatial semantics. In our permutation placebo test, retaining the exact values of $Y_D$ but randomly shuffling their assignment across distance bins degrades performance dramatically ($\Delta\mathrm{CPC} = -0.00696$, a net deficit of $0.01050$ relative to correct calibration, $p < 10^{-14}$). This rules out any hypothesis that calibration benefits from generic variance reduction or output smoothing; the gain depends fundamentally on binding specific mobility masses to their true physical distance intervals.

*(Tiếng Việt: Giá trị sử dụng của $Y_D$ gắn chặt với nội dung ngữ nghĩa không gian của nó. Trong thí nghiệm hoán vị (permutation placebo), việc giữ nguyên các giá trị tỷ lệ của $Y_D$ nhưng xáo trộn thứ tự gán vào các khoảng khoảng cách làm CPC sụt giảm nghiêm trọng ($\Delta\mathrm{CPC} = -0.00696$, thấp hơn $0.01050$ so với hiệu chỉnh đúng, $p < 10^{-14}$). Kết quả này bác bỏ hoàn toàn giả thuyết cho rằng hiệu chỉnh chỉ là hệ quả của việc làm trơn đầu ra; lợi ích bắt nguồn từ sự liên kết chính xác giữa khối lượng di chuyển và dải cự ly vật lý tương ứng.)*

Under synthetic Total Variation noise ($\epsilon \in [0\%, 5\%]$), calibration gain degrades monotonically. The empirical crossover breakdown threshold is estimated at:

$$\epsilon_{\text{cross}} \approx 4.45\% \quad [95\%\text{ CI: } 4.16\%, 4.77\%]$$

Beyond approximately 4.4% TV estimation error, the distortion introduced by inaccurate scaling factors outweighs the benefit of macro distance alignment. This defines a concrete operational tolerance threshold: aggregate observations derived from mobile signaling or travel surveys must achieve an accuracy within $\approx 4.4\%$ TV error to guarantee positive calibration utility.

*(Tiếng Việt: Dưới tác động của nhiễu Total Variation tổng hợp ($\epsilon \in [0\%, 5\%]$), hiệu quả hiệu chỉnh suy giảm đơn điệu. Ngưỡng phá vỡ tín hiệu thực nghiệm đạt $\epsilon_{\text{cross}} \approx 4.45\%$ [khoảng tin cậy 95%: $4.16\%, 4.77\%$]. Khi sai số ước lượng của $Y_D$ vượt quá khoảng 4.4% sai số TV, độ lệch do trọng số hiệu chỉnh sai sẽ lấn át lợi ích nắn chỉnh cự ly. Điều này xác lập một giới hạn dung sai thực tiễn: dữ liệu viễn thông hay khảo sát cần đạt độ chính xác trong ngưỡng 4.4% TV error để bảo đảm mang lại hiệu quả cải thiện dương.)*

---

## 5.5 Target-specific information vs generic spatial priors
*(Tiếng Việt: **5.5. Thông tin đặc thù mục tiêu và các prior không gian phổ quát**)*

A central question in transfer learning is whether target observations convey city-specific idiosyncrasies or merely restate universal physical laws (e.g., gravity decay). Our dose-matched placebo controls provide unambiguous evidence:
1. **Dose-Matched Wrong Donors**: Applying donor distributions from other cities scaled to the exact same perturbation dose ($D_T$) yields zero systematic improvement ($\Delta\mathrm{CPC} = -0.000091, p = 0.4097$). The target distribution significantly outperforms wrong-city donors in 46 of 50 cities ($+0.003630, p = 2.19 \times 10^{-11}$).
2. **Dose-Matched Training-Mean**: Applying the mean distance decay profile averaged across training cities yields a marginal gain of $+0.000914$ ($p = 0.4319$), but target-specific $Y_D$ significantly outperforms it in 47 of 50 cities ($+0.002626, p = 4.03 \times 10^{-11}$).

*(Tiếng Việt: Một câu hỏi trọng tâm trong học chuyển giao là liệu quan sát mục tiêu mang thông tin đặc thù của thành phố hay chỉ lặp lại các quy luật vật lý phổ quát (như suy giảm trọng lực). Các kiểm tra placebo chuẩn hóa liều lượng cung cấp bằng chứng rõ ràng: (1) Phân phối sai thành phố khi khớp cùng liều lượng can thiệp ($D_T$) không mang lại cải thiện ($\Delta\mathrm{CPC} = -0.000091, p = 0.4097$), trong khi phân phối mục tiêu vượt trội tại 46/50 thành phố ($+0.003630, p = 2.19 \times 10^{-11}$); (2) Phân phối trung bình tập huấn luyện chỉ đạt mức tăng nhẹ $+0.000914$ ($p = 0.4319$), và phân phối mục tiêu vượt trội tại 47/50 thành phố ($+0.002626, p = 4.03 \times 10^{-11}$).)*

Because the zero-shot baseline already encodes pairwise Euclidean distances and regional spatial context, these results prove that geometric distance alone is insufficient to reconstruct target mobility patterns. $Y_D$ encapsulates unique structural properties—such as the balance between suburban commuting and central urban concentration—that cannot be deduced from cross-city priors alone.

*(Tiếng Việt: Vì mô hình zero-shot đã tích hợp khoảng cách Euclid và bối cảnh không gian khu vực, các kết quả trên chứng minh rằng khoảng cách hình học đơn thuần không đủ để tái tạo hoàn chỉnh cấu trúc di chuyển mục tiêu. $Y_D$ chứa đựng các đặc tính cấu trúc độc thù—chẳng hạn như sự cân bằng giữa di chuyển ngoại ô và tập trung trung tâm—vốn không thể suy diễn thuần túy từ các prior liên thành phố.)*

---

## 5.6 Inter-city performance heterogeneity
*(Tiếng Việt: **5.6. Sự không đồng nhất về hiệu quả giữa các thành phố**)*

While 45 of 50 cities exhibit positive gains, the magnitude of improvement varies across metropolitan areas, with 5 cities exhibiting negative changes (El Paso, Oklahoma City, Jacksonville, Louisville, Long Beach). This heterogeneity is entirely coherent with our mechanistic framework:
- **Low Initial Misalignment**: Cities with minimal baseline distance mismatch ($d_{\text{pre}} \approx 0$) offer little room for macro correction; small empirical perturbations in $Y_D$ can slightly depress CPC.
- **Intra-Bin vs Inter-Bin Error Dominance**: In cities where baseline errors reside primarily in misallocating flows across specific zone pairs within the same distance band rather than between distance bands, one-dimensional distance calibration cannot correct the underlying distortion.

*(Tiếng Việt: Mặc dù 45/50 thành phố có $\Delta\mathrm{CPC} > 0$, mức độ cải thiện có sự phân hóa và có 5 thành phố ghi nhận $\Delta\text{CPC} < 0$ (El Paso, Oklahoma City, Jacksonville, Louisville, Long Beach). Sự không đồng nhất này hoàn toàn nhất quán với khung cơ chế $d_{\text{pre}}$: (1) Ở các thành phố có baseline đã rất khớp phân phối cự ly ($d_{\text{pre}} \approx 0$), dư địa cải thiện vĩ mô rất ít; (2) Ở các đô thị mà sai số baseline chủ yếu nằm ở việc phân bổ sai giữa các cặp vùng trong cùng một khoảng thay vì giữa các khoảng, hiệu chỉnh khoảng cách một chiều không thể khắc phục được sai số này.)*

Consequently, $Y_D$ calibration should not be viewed as an unconditional guarantee of improvement, but as a conditioned inference tool whose efficacy is governed by the baseline's macro distance mismatch and the stability of target observations.

*(Tiếng Việt: Do đó, hiệu chỉnh $Y_D$ không nên được xem như một bảo đảm cải thiện trong mọi trường hợp, mà là một công cụ suy luận có điều kiện với hiệu quả phụ thuộc vào độ lệch cự ly vĩ mô của baseline và độ ổn định của quan sát mục tiêu.)*

---

## 5.7 The value of spatial resolution depends on intra-urban spatial structure
*(Tiếng Việt: **5.7. Giá trị của độ phân giải không gian phụ thuộc vào cấu trúc nội tại của thành phố**)*

The findings regarding county-level observations clarify that the utility of $Y_D$ depends not only on distance bin granularity, but also on the spatial scale at which distributions are aggregated. However, higher spatial resolution does not automatically guarantee substantial gains; benefits emerge only when spatial partitioning provides distinct distributional information compared to the city-wide signal.

In the city-level configuration (`M1_city`), a single vector $\mathbf{Y}_{D,c}$ modulates flow mass across distance intervals for the entire metropolis. This operator effectively rectifies average distance decay biases in the baseline, but applies an identical set of scaling multipliers to all origin tracts. Consequently, it cannot accommodate settings where distinct subregions within the same urban area exhibit markedly different distance distributions.

County-level observations (`M1_county`) introduce this localized capacity by supplying a separate distribution for each origin-county group. If the baseline overpredicts long-distance journeys in one county but underpredicts them in another, a single city-wide distribution may allow these localized errors to cancel out in the aggregate sum. In contrast, county-level calibration scales each origin group independently before assembling the unified city-wide prediction. This provides a coherent mechanistic explanation for the localized improvements observed in Kansas City ($+0.0027$), New York ($+0.0021$), and Dallas ($+0.0011$).

Nevertheless, the average gain across the full benchmark remains modest ($+0.00014$). This primarily stems from dataset structure: for the 39 single-county cities, county-level calibration is mathematically identical to city-level calibration and produces zero structural difference ($\Delta\operatorname{CPC}_{\mathrm{res}} \equiv 0$). Even within the 11 multi-county cities, the average incremental gain is $+0.00063$ CPC. Thus, the evidence does not support a sweeping claim that refining resolution from city to county always delivers large benefits. Instead, it indicates that the marginal value of higher spatial resolution is conditional and concentrated in cities characterized by pronounced intra-urban spatial heterogeneity across origin zones.

This finding also illuminates why $Y_D$ yields focused rather than transformative improvements over the baseline. The calibration operator only reallocates flow mass between distance intervals or origin-county slices; it leaves the relative ordering of OD pairs within each slice strictly invariant ($\tau = 1.00000000$). Consequently, overall accuracy remains bounded by the baseline's capacity to rank zone pairs internally. If the baseline misranks pairs within the same distance band, providing finer county-level distributions cannot directly correct that intra-bin distortion.

Three explicit limitations warrant consideration:
1. **Administrative vs Functional Zoning**: County boundaries represent political-administrative jurisdictions rather than functional commuting basins or travel communities. Partitioning based on functional urban zones or mobility communities might achieve greater within-zone behavioral homogeneity, a hypothesis reserved for future inquiry.
2. **Dataset Footprint Boundary**: County groups comprise only those tracts included within the study city dataset provided by the laboratory, and do not represent total county-wide travel demand extending beyond the study area.
3. **Oracle Aggregate Setting**: County distributions in our benchmark are derived as oracle aggregate observations from reference OD matrices. These results demonstrate the theoretical information ceiling of county-level granularity, but do not prove that equivalent gains would materialize under noisy or incomplete real-world telemetry.

In summary, this experiment enriches RQ2 with a nuanced conclusion: refining the spatial resolution of $Y_D$ provides supplementary predictive information when a metropolis encompasses multiple origin zones with heterogeneous distance decay patterns, but average gains are modest and vanish when spatial partitioning yields no new distributional variance.

*(Tiếng Việt: Kết quả về county-level observation làm rõ rằng giá trị của $Y_D$ không chỉ phụ thuộc vào độ chi tiết của các khoảng cách mà còn phụ thuộc vào độ phân giải không gian tại đó phân phối được quan sát. Tuy nhiên, độ phân giải cao hơn không tự động dẫn đến cải thiện lớn hơn. Lợi ích chỉ xuất hiện khi việc chia nhỏ không gian thực sự cung cấp thông tin khác với phân phối cấp city. Trong cấu hình city-level, một vector $\mathbf{Y}_{D,c}$ duy nhất điều chỉnh khối lượng luồng giữa các khoảng cách cho toàn thành phố. Cách hiệu chỉnh này có thể sửa sai lệch khoảng cách trung bình của baseline, nhưng áp dụng cùng một tập trọng số cho tất cả origin tract. Vì vậy, nó không thể biểu diễn trường hợp các khu vực xuất phát khác nhau trong cùng thành phố có các phân phối khoảng cách khác nhau. County-level observation bổ sung khả năng này bằng cách sử dụng một phân phối riêng cho từng nhóm origin-county. Nếu baseline dự báo quá nhiều chuyến đi xa ở một county nhưng quá ít chuyến đi xa ở một county khác, một phân phối city-level có thể làm hai sai lệch này triệt tiêu khi tổng hợp. Ngược lại, county-level calibration có thể điều chỉnh từng nhóm riêng trước khi ghép chúng thành dự báo toàn thành phố. Đây là cơ chế hợp lý giải thích mức cải thiện quan sát được tại Kansas City, New York và Dallas. Tuy nhiên, mức tăng trung bình trên toàn bộ benchmark rất nhỏ. Điều này trước hết xuất phát từ cấu trúc dữ liệu: đối với 39 single-county cities, county-level calibration hoàn toàn tương đương city-level calibration và không thể tạo ra thay đổi ($\Delta\operatorname{CPC}_{\mathrm{res}}\equiv 0$). Ngay cả trong 11 multi-county cities, mức cải thiện trung bình cũng chỉ khoảng $+0.00063$ CPC ($81.8\%$ win rate). Do đó, kết quả không hỗ trợ nhận định rằng tăng độ phân giải từ city lên county luôn mang lại lợi ích đáng kể. Thay vào đó, nó cho thấy giá trị biên của độ phân giải cao hơn có tính điều kiện và tập trung ở một số thành phố có cấu trúc di chuyển không đồng nhất giữa các khu vực xuất phát. Kết quả này cũng giúp giải thích vì sao $Y_D$ chỉ tạo ra mức cải thiện nhỏ so với baseline. Phép hiệu chỉnh chỉ thay đổi tổng khối lượng giữa các khoảng cách hoặc giữa các nhóm county–distance; nó không thay đổi thứ tự tương đối của các cặp OD trong cùng một nhóm ($\tau=1.0$). Vì vậy, hiệu quả cuối cùng vẫn phụ thuộc vào việc baseline đã học được cấu trúc nội khoảng hữu ích đến mức nào. Nếu baseline xếp hạng sai các cặp trong cùng một khoảng, việc cung cấp phân phối ở cấp county không thể trực tiếp sửa sai lệch đó. Có ba giới hạn cần lưu ý: (1) County là đơn vị hành chính và không nhất thiết tương ứng với ranh giới chức năng của hành vi di chuyển; (2) Các nhóm county chỉ bao gồm những tract nằm trong bộ dữ liệu city do Lab cung cấp, không đại diện cho toàn bộ hoạt động bên ngoài phạm vi nghiên cứu; (3) Các phân phối county-level hiện được trích xuất dưới dạng oracle aggregate observations, kết quả do đó đánh giá giá trị thông tin tiềm năng chứ chưa chứng minh với dữ liệu quan sát thực tế có nhiễu. Nhìn chung, thí nghiệm này bổ sung cho RQ2 bằng một kết luận có giới hạn: tăng độ phân giải không gian của $Y_D$ có thể mang lại lợi ích bổ sung khi thành phố chứa nhiều vùng xuất phát với các mẫu khoảng cách khác nhau, nhưng lợi ích trung bình nhỏ và không tồn tại trong trường hợp việc phân nhóm không tạo ra thông tin mới.)*

---

## 5.8 Methodological and practical implications
*(Tiếng Việt: **5.8. Ý nghĩa phương pháp luận và thực tiễn**)*

From a **methodological** standpoint, our findings illustrate that cross-city deep learning models do not necessarily require end-to-end retraining or parameter fine-tuning to benefit from target-domain observations. Decoupling spatial representation learning (performed globally on training cities) from macro-distribution calibration (performed analytically at test time) offers a modular, highly scalable paradigm for cross-city transfer.

*(Tiếng Việt: Về mặt phương pháp luận, kết quả cho thấy các mô hình deep learning xuyên thành phố không nhất thiết phải huấn luyện lại toàn bộ hay tinh chỉnh tham số để khai thác thông tin từ miền mục tiêu. Việc tách biệt giữa học biểu diễn không gian (thực hiện toàn cục trên tập huấn luyện) và hiệu chỉnh phân phối vĩ mô (thực hiện giải tích tại thời điểm kiểm tra) tạo ra một mô hình chuyển giao có tính mô-đun hóa cao và mở rộng dễ dàng.)*

From a **practical** perspective, this approach is well suited for resource-constrained cities that lack full OD survey matrices but can access aggregate statistics from telecom data, census commuting reports, or travel surveys. A 1D distance distribution ($K=8$) requires minimal storage, incurs negligible computational overhead, and dramatically mitigates privacy risks compared to releasing full $N \times N$ origin–destination tables.

*(Tiếng Việt: Về mặt thực tiễn, phương pháp này đặc biệt phù hợp cho các đô thị hạn chế nguồn lực, nơi không có ma trận khảo sát OD đầy đủ nhưng có thể tiếp cận thống kê tổng hợp từ dữ liệu viễn thông, báo cáo điều tra dân số hoặc khảo sát giao thông. Phân phối cự ly 1 chiều ($K=8$) đòi hỏi dung lượng lưu trữ tối thiểu, chi phí tính toán không đáng kể và giảm thiểu rủi ro vi phạm quyền riêng tư so với việc công bố toàn bộ bảng OD kích thước $N \times N$.)*

---

## 5.9 Limitations
*(Tiếng Việt: **5.9. Các giới hạn của nghiên cứu**)*

Several key scope boundaries and methodological limitations must be acknowledged:
1. **Conditioning on Known Positive Support ($\Omega_c^+$)**: The evaluation is conducted on observed positive interzonal pairs ($T_{ij} \ge 1, D_{ij} > 0$). The framework does not address link prediction or the zero-flow identification problem.
2. **One-Dimensional Constraint**: $Y_D$ constrains only scalar distance allocations; it provides no information regarding directional orientation, polycentric attraction hubs, or trip purposes.
3. **Geographic and Spatial Resolution Scope**: Experiments are evaluated on 50 U.S. metropolitan areas at the census tract level. Extrapolation to developing countries with informal transit or different spatial zoning systems requires independent empirical validation.
4. **Synthetic Noise Model**: Noise experiments utilize synthetic Dirichlet perturbations; real-world survey biases may exhibit structured demographic or geographic non-randomness.

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

In conclusion, target-city trip distance distributions ($Y_D$) provide a quantifiable, statistically robust, and structurally grounded source of incremental information for zero-shot OD intensity reconstruction. The observed performance gain is fundamentally driven by correcting macro-level distance allocation mismatch ($d_{\text{pre}}$), strictly requires target-specific spatial ordering, exhibits high tolerance up to $\approx 4.4\%$ TV noise, and operates consistently across 90.0% of evaluated urban regions. These findings establish the feasibility of leveraging lightweight aggregate observations to guide frozen foundation models in data-scarce urban environments.

*(Tiếng Việt: Tóm lại, phân phối di chuyển theo cự ly của thành phố mục tiêu ($Y_D$) mang lại nguồn thông tin gia tăng định lượng được, vững chắc về mặt thống kê và có cơ sở cấu trúc rõ ràng cho quá trình tái tạo cường độ luồng OD zero-shot. Mức cải thiện quan sát được bắt nguồn từ việc nắn chỉnh sự lệch pha phân bổ cự ly vĩ mô ($d_{\text{pre}}$), đòi hỏi tính đúng thứ tự không gian đặc thù của thành phố mục tiêu, có khả năng chịu nhiễu lên tới $\approx 4.4\%$ TV error, và vận hành nhất quán trên 90.0% các đô thị được đánh giá. Những phát hiện này khẳng định tính khả thi của việc sử dụng các quan sát tổng hợp gọn nhẹ để hỗ trợ các mô hình nền tảng đóng băng trong các môi trường đô thị khan hiếm dữ liệu.)*
