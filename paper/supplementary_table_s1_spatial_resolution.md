# Supplementary Note S7. Exploratory Analysis of County-Level Spatial Resolution

*(Tiếng Việt: **Phụ lục S7. Phân tích thăm dò về độ phân giải không gian cấp county**)*

---

## S7.1 Setup

*(Tiếng Việt: **S7.1. Thiết lập**)*

This exploratory analysis investigates whether conditioning on aggregate distance observations at a finer sub-metropolitan spatial resolution—specifically grouped by administrative county units—provides supplemental information beyond city-level aggregation.

County boundaries are obtained from the Database of Global Administrative Areas, version 4.1 (GADM 4.1) [@gadm41]. Each tract is mapped to its encompassing county via a spatial point-in-polygon join between the tract centroid and the county polygon. If a centroid does not receive a valid within match—for example, because it lies on a polygon boundary or near a coastline—the implementation falls back to a nearest-polygon join in EPSG:5070 and accepts the assignment only when the centroid-to-polygon distance is at most 5 km; otherwise, execution stops with an error. Duplicate matches are resolved deterministically so that each tract receives exactly one county label. GADM is strictly utilized for this spatial grouping step; GADM is not the source of tract centroid coordinates, urban features, or OD flows.

Letting (i)$ denote the county assigned to tract $, OD pairs are grouped strictly by the **origin tract's county**:


\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c : g(i) = \ell\right\}.


Destination tract $ may belong to the same county or a different county within the metropolitan area. The distance-binned flow mass of county group $\ell$ is:


Y_{c,\ell,b} = \frac{\sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij}}, \qquad \sum_{b=1}^K Y_{c,\ell,b} = 1.


Because the input data are strictly bounded within the tracts of the city dataset provided by the laboratory, $\mathbf{Y}_{c,\ell}$ describes the outflow distance distribution of trips originating from the tracts of city $ assigned to county $\ell$. It does not represent total county-wide mobility outside the study city's spatial footprint.

Each distribution $\mathbf{Y}_{c,\ell}$ is used to calibrate OD pairs whose origin tract belongs to county $\ell$. The calibrated predictions from all county groups are then assembled into a complete OD prediction for the city:


\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \hat{t}_{c,ij}^{\mathrm{county}} : (i,j) \in \Omega_{c,\ell}^+ \right\},


where $\mathcal{G}_c$ denotes the set of counties present in the dataset for city $.

Crucially, increasing observational resolution from city to county does not alter the evaluation scope. The model still reconstructs and is evaluated against the complete set of positive flows $\Omega_c$ for the target city; only the aggregate supervisory signal supplied during calibration becomes spatially more granular (M1_county).

Among the 50 metropolitan datasets in the benchmark, exactly 39 are single-county areas (where all tracts belong to a single county, $|\mathcal{G}_c| = 1$). For these 39 areas, county partitioning is mathematically identical to city-level partitioning, yielding {\mathrm{county}} \equiv M1_{\mathrm{city}}$ and $\Delta\mathrm{CPC}_{\mathrm{res},c} = 0$ by construction. Only the 11 metropolitan areas spanning between 2 and 7 counties create genuine sub-metropolitan partitions.

*(Tiếng Việt: Phân tích thăm dò này kiểm tra xem việc cung cấp quan sát khoảng cách tổng hợp ở độ phân giải không gian chi tiết hơn cấp thành phố—cụ thể là nhóm theo đơn vị hành chính cấp hạt (county)—có mang lại thông tin bổ sung hay không.

Ranh giới county được lấy từ Database of Global Administrative Areas, phiên bản 4.1 (GADM 4.1) [@gadm41]. Mỗi tract được gán vào county bao quanh tương ứng thông qua phép nối điểm trong đa giác (point-in-polygon) giữa tọa độ tâm tract và polygon của county. Trường hợp tâm tract nằm trên ranh giới polygon hoặc gần bờ biển, quy trình sử dụng phép gán polygon gần nhất trong hệ tọa độ EPSG:5070 với ngưỡng khoảng cách tối đa 5 km. Mỗi tract được gán duy nhất vào một county. GADM chỉ được sử dụng nghiêm ngặt cho bước phân nhóm không gian này, không phải nguồn của tọa độ tâm, đặc trưng đô thị hay luồng OD.

Gọi (i)$ là county được gán cho tract $. Các cặp OD được nhóm theo **county của điểm xuất phát (origin tract)**:

\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c : g(i) = \ell\right\}.

Tract đích $ có thể thuộc cùng county hoặc county khác trong vùng đô thị. Phân phối khoảng cách của nhóm origin-county $\ell$ được định nghĩa:

Y_{c,\ell,b} = \frac{\sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij}}, \qquad \sum_{b=1}^K Y_{c,\ell,b} = 1.

Vì dữ liệu đầu vào giới hạn trong tập tract của vùng đô thị do phòng thí nghiệm cung cấp, $\mathbf{Y}_{c,\ell}$ mô tả phân phối khoảng cách xuất phát từ các tract thuộc county $\ell$ trong vùng đô thị đó, không đại diện cho toàn bộ di chuyển trên toàn địa bàn county ngoài phạm vi nghiên cứu.

Mỗi phân phối $\mathbf{Y}_{c,\ell}$ được sử dụng để hiệu chỉnh các cặp OD có origin tract thuộc county $\ell$. Sau đó, các dự báo đã hiệu chỉnh từ toàn bộ các nhóm county được tập hợp lại thành dự báo hoàn chỉnh cho vùng đô thị:

\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \hat{t}_{c,ij}^{\mathrm{county}} : (i,j) \in \Omega_{c,\ell}^+ \right\},

trong đó $\mathcal{G}_c$ là tập hợp các county xuất hiện trong tập dữ liệu của vùng đô thị $.

Quan trọng là việc chuyển độ phân giải quan sát từ cấp thành phố sang cấp county không làm thay đổi phạm vi đánh giá: mô hình vẫn tái tạo và được đánh giá trên toàn bộ tập hỗ trợ luồng dương $\Omega_c$ của vùng đô thị mục tiêu; chỉ có tín hiệu giám sát tổng hợp trong bước hiệu chỉnh trở nên chi tiết hơn theo không gian (M1_county).

Trong số 50 vùng đô thị của benchmark, có đúng 39 vùng single-county (nơi toàn bộ các tract thuộc cùng một county duy nhất, do đó $|\mathcal{G}_c| = 1$). Với 39 vùng này, phân hoạch theo county hoàn toàn trùng khớp với phân hoạch cấp thành phố, dẫn đến {\mathrm{county}} \equiv M1_{\mathrm{city}}$ và $\Delta\mathrm{CPC}_{\mathrm{res},c} = 0$ về mặt toán học. Chỉ có 11 vùng đô thị trải rộng qua từ 2 đến 7 county tạo ra phân hoạch mới thực sự.)*

---

## S7.2 Results

*(Tiếng Việt: **S7.2. Kết quả**)*

Across all 50 metropolitan datasets, the pooled incremental gain from county-level calibration over city-level calibration is very small:


\Delta\mathrm{CPC}_{\mathrm{res}} = +0.00014, \quad \text{95% CI } [+0.00002, +0.00028], \quad \text{Wilcoxon } p = 0.0064.


This modest pooled gain is heavily dominated by the 39 single-county areas where the incremental gain is identically zero by construction.

For the subset of 11 multi-county metropolitan areas (22% of the benchmark), county-level calibration achieved gains in 9 of 11 areas, with a mean incremental gain of $+0.00063$ (Table S1 and Figure S1).

![Figure S1](figures/fig_s1_spatial_resolution.png)
**Figure S1. Comparison of city-level and county-level calibration CPC gains across 11 multi-county metropolitan areas.** Analysis is exploratory; the 39 single-county metropolitan areas are omitted because the two partitions are mathematically equivalent.

*(Tiếng Việt: Trên toàn bộ 50 vùng đô thị, mức tăng bổ sung pooled từ hiệu chỉnh cấp county so với hiệu chỉnh cấp thành phố là rất nhỏ:

\Delta\mathrm{CPC}_{\mathrm{res}} = +0.00014, \quad \text{95% CI } [+0.00002, +0.00028], \quad \text{Wilcoxon } p = 0.0064.

Mức tăng pooled khiêm tốn này chịu chi phối bởi 39 vùng single-county có mức tăng bằng 0 tuyệt đối theo cấu trúc.

Đối với nhóm 11 vùng đô thị multi-county (chiếm 22% tập benchmark), hiệu chỉnh cấp county đạt mức cải thiện tại 9/11 vùng, với mức tăng bổ sung trung bình là $+0.00063$ (Bảng S1 và Hình S1).

![Hình S1](figures/fig_s1_spatial_resolution.png)
**Hình S1. So sánh mức tăng CPC của hiệu chỉnh cấp thành phố và cấp county trên 11 vùng đô thị multi-county. Phân tích mang tính thăm dò; 39 vùng single-county không được hiển thị vì hai cách phân nhóm tương đương về mặt toán học.**)*

### Supplementary Table S1. Descriptive city-level results for the multi-county spatial-resolution subset

City-level comparison of the zero-shot baseline ($), city-level oracle calibration ({\mathrm{city}}$), and origin-county-conditioned oracle calibration ({\mathrm{county}}$) for the 11 metropolitan datasets containing tracts assigned to more than one county. The resolution increment is defined as $\Delta\mathrm{CPC}_{\mathrm{res},c}=\mathrm{CPC}(M1_{\mathrm{county}})-\mathrm{CPC}(M1_{\mathrm{city}})$. Values are descriptive city-level estimates. No subgroup confidence interval or hypothesis test is reported unless supported by a separately verified uncertainty artifact.

*(Tiếng Việt: **Bảng S1. Kết quả mô tả theo thành phố cho nhóm phân tích độ phân giải không gian đa county.** Bảng so sánh zero-shot baseline ($), hiệu chỉnh oracle cấp city ({\mathrm{city}}$) và hiệu chỉnh oracle có điều kiện theo origin-county ({\mathrm{county}}$) cho 11 bộ dữ liệu đô thị có các tract được gán vào nhiều hơn một county. Mức tăng do độ phân giải được định nghĩa là $\Delta\mathrm{CPC}_{\mathrm{res},c}=\mathrm{CPC}(M1_{\mathrm{county}})-\mathrm{CPC}(M1_{\mathrm{city}})$. Các giá trị là ước lượng mô tả ở cấp city. Không báo cáo khoảng tin cậy hoặc kiểm định giả thuyết cho subgroup nếu không có artifact bất định riêng đã được xác minh.)*

| City / Thành phố | Origin counties | $ CPC | {\mathrm{city}}$ CPC | {\mathrm{county}}$ CPC | $\Delta\mathrm{CPC}_{\mathrm{city}}$ | $\Delta\mathrm{CPC}_{\mathrm{county}}$ | $\Delta\mathrm{CPC}_{\mathrm{res}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kansas City | 3 | 0.721071 | 0.726877 | 0.729612 | +0.005807 | +0.008542 | +0.002735 |
| New York | 7 | 0.524464 | 0.525775 | 0.527870 | +0.001311 | +0.003407 | +0.002096 |
| Dallas | 3 | 0.685251 | 0.695768 | 0.696916 | +0.010517 | +0.011665 | +0.001148 |
| Denver | 3 | 0.715551 | 0.715713 | 0.716053 | +0.000162 | +0.000501 | +0.000339 |
| Omaha | 2 | 0.747005 | 0.752621 | 0.752828 | +0.005616 | +0.005822 | +0.000207 |
| Tulsa | 2 | 0.779746 | 0.781563 | 0.781750 | +0.001817 | +0.002005 | +0.000187 |
| Detroit | 2 | 0.684499 | 0.685059 | 0.685239 | +0.000560 | +0.000740 | +0.000180 |
| Chicago | 2 | 0.672433 | 0.674337 | 0.674358 | +0.001905 | +0.001925 | +0.000021 |
| Boston | 3 | 0.687180 | 0.687561 | 0.687578 | +0.000381 | +0.000398 | +0.000017 |
| Milwaukee | 2 | 0.741276 | 0.742868 | 0.742854 | +0.001591 | +0.001578 | -0.000014 |
| Atlanta | 2 | 0.710814 | 0.719676 | 0.719645 | +0.008862 | +0.008831 | -0.000031 |
| **Multi-county mean / Trung bình đa county** | — | — | — | — | — | — | **+0.000626** |
| **Positive resolution gains / Số thành phố tăng dương** | — | — | — | — | — | — | **9 / 11** |

*Note: Rows are sorted by $\Delta\mathrm{CPC}_{\mathrm{res}}$ in descending order. County labels are assigned from tract centroids using GADM 4.1 and group OD pairs by the county of the origin tract. Destination tracts may belong to the same or another county represented within the city dataset. Prediction and evaluation remain city-wide on the same known positive support. The 39 single-county cities are omitted from this table because {\mathrm{county}}\equiv M1_{\mathrm{city}}$ by construction. Results are seed-averaged across model seeds $\{1, 10, 100\}$.*

*(Tiếng Việt: Ghi chú: Các dòng được sắp xếp theo $\Delta\mathrm{CPC}_{\mathrm{res}}$ giảm dần. Nhãn county được gán từ tâm tract bằng GADM 4.1 và nhóm các cặp OD theo county của tract gốc. Tract đích có thể thuộc cùng county hoặc county khác trong vùng đô thị. Dự báo và đánh giá thực hiện trên toàn thành phố trên cùng tập hỗ trợ dương đã biết. 39 thành phố đơn county được bỏ qua trong bảng này vì {\mathrm{county}}\equiv M1_{\mathrm{city}}$ theo cấu trúc. Kết quả trung bình qua các seed $\{1, 10, 100\}$.)*

---

## S7.3 Interpretive Boundaries

*(Tiếng Việt: **S7.3. Giới hạn diễn giải**)*

Results of the county-level analysis must be interpreted under the following strict boundaries:

1. **Small Sample Size and Descriptive Evidence**: The analysis is based on only 11 multi-county metropolitan areas. In the absence of a separately verified stratified uncertainty estimation for this subset, the 9/11 improvement remains purely descriptive empirical evidence and does not establish a generalized statistical law.
2. **Administrative vs. Functional Boundaries**: Counties are historical administrative units not delineated based on travel commuting sheds, transport corridors, or functional urban zoning. Grouping by county therefore does not necessarily capture the true behavioral heterogeneity of spatial travel.
3. **Incomplete Spatial Coverage**: County groups only encompass tracts located within the laboratory-provided metropolitan area boundary, and do not represent total travel flows across the full territorial area of those counties.
4. **No Proof of Causality or Practical Operational Guarantee**: Assigning tract centroids geometrically and utilizing oracle distributions do not account for real-world linkage errors. The experiment does not prove that increasing spatial resolution in general will always improve OD matrix reconstruction in real-world applications.

*(Tiếng Việt: Kết quả phân tích cấp county cần được diễn giải với các giới hạn nghiêm ngặt sau:

1. **Quy mô mẫu nhỏ và bằng chứng mô tả**: Phân tích chỉ dựa trên 11 vùng đô thị multi-county. Do không có ước lượng bất định phân tầng riêng cho tập con này, kết quả 9/11 vùng cải thiện chỉ mang tính chất mô tả thực nghiệm, không đủ cơ sở để khẳng định tính quy luật thống kê tổng quát.
2. **Ranh giới hành chính so với ranh giới chức năng**: County (đơn vị hành chính cấp hạt) là ranh giới quản lý hành chính lịch sử, không được thiết kế dựa trên lưu vực đi lại, hành lang giao thông hay cấu trúc phân vùng chức năng đô thị. Vì vậy, việc phân nhóm theo county không nhất thiết phản ánh đúng tính không đồng nhất của hành vi di chuyển.
3. **Phạm vi không gian không đầy đủ**: Các nhóm county chỉ bao gồm các tract nằm trong ranh giới vùng đô thị do phòng thí nghiệm cung cấp, không đại diện cho toàn bộ luồng di chuyển trên toàn diện tích địa giới của các county đó.
4. **Không chứng minh quan hệ nhân quả hay bảo đảm thực tế**: Việc gán tâm tract bằng phương pháp hình học và sử dụng phân phối oracle không phản ánh các sai số ghép nối thực tế. Thí nghiệm không chứng minh rằng tăng độ phân giải không gian nói chung sẽ luôn cải thiện việc tái tạo ma trận OD trong các ứng dụng thực tế.)*
