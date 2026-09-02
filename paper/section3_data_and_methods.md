# Section 3: Data Sources, Spatial Units, and Methodology

*(Tiếng Việt: **Mục 3: Nguồn dữ liệu, đơn vị không gian và phương pháp luận**)*

---

## 3.1 Data Sources and Spatial Representation
*(Tiếng Việt: **3.1. Nguồn dữ liệu và biểu diễn không gian**)*

The empirical evaluation is conducted across 50 metropolitan areas in the United States. Each city is represented as a spatial network composed of census tract units. Each tract is characterized by its geographic centroid coordinates $\mathbf{s}_i = (\operatorname{lon}_i, \operatorname{lat}_i)$ and a 26-dimensional feature vector describing local urban context:
- **13 Census demographic features** (e.g., population density, employment, household income),
- **8 Point-of-Interest (POI) features** (e.g., commercial, education, recreational amenities), and
- **5 Road network features** (e.g., road density, intersection topology).

These features are sourced from an aggregated benchmark dataset compiled by the laboratory. The primary data origins, vintage collection years, exact versions, and preprocessing workflows for each feature group are currently under verification and will be fully reported prior to formal publication.

Tract-level polygon geometries are not utilized in the spatial model. Instead, each tract is represented spatially exclusively by its centroid coordinates. All pairwise distances and spatial neighborhood graph structures are constructed directly from these coordinates.

The continuous distance domain between tract centroids is partitioned into $K$ intervals:

$$I_b = [a_{b-1}, a_b), \qquad b = 1, \dots, K$$

where $d_{ij}$ is computed using the spherical Haversine formula with Earth radius $R = 6371\text{ km}$:

$$d_{ij} = 2R \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos(\varphi_i)\cos(\varphi_j) \sin^2\left(\frac{\Delta\lambda}{2}\right) } \right)$$

For each cross-validation fold $f$, distance bin edges are determined independently from the interzonal pairs of the training cities:

$$\mathcal{D}_{\mathrm{train}}^{(f)} = \left\{ d_{ij} : c \in \mathcal{C}_{\mathrm{train}}^{(f)}, (i,j) \in \Omega_c^+, i \ne j, d_{ij} > 0 \right\}$$

The internal bin edges are defined at the $b/K$ quantiles:

$$a_b = Q_{b/K}\left(\mathcal{D}_{\mathrm{train}}^{(f)}\right), \qquad b = 1, \dots, K-1$$

with $a_0 = 0$ and $a_K = \infty$. Each OD pair contributes exactly one distance observation to quantile estimation, independent of trip count (**pair-weighted quantiles**). Because edges are derived strictly from training cities, test-city data are never used to define distance bins. Duplicate quantile values are removed, so the active number of bins may be strictly less than $K$.

*(Tiếng Việt: Nghiên cứu được thực hiện trên 50 thành phố của Hoa Kỳ. Mỗi thành phố được biểu diễn dưới dạng một tập hợp các đơn vị không gian cấp tract. Mỗi tract có tọa độ tâm $\mathbf{s}_i = (\operatorname{lon}_i, \operatorname{lat}_i)$ và 26 đặc trưng mô tả bối cảnh đô thị, bao gồm 13 đặc trưng Census, 8 đặc trưng điểm quan tâm (POI) và 5 đặc trưng mạng lưới đường. Các đặc trưng này được lấy từ bộ dữ liệu do Lab tổng hợp. Nguồn ban đầu, năm dữ liệu, phiên bản và quy trình tiền xử lý của từng nhóm đặc trưng đang được xác minh và sẽ được bổ sung đầy đủ trước khi công bố nghiên cứu. Nghiên cứu không sử dụng hình học polygon của tract. Thay vào đó, mỗi tract được biểu diễn về mặt không gian bằng tọa độ tâm. Khoảng cách giữa các cặp tract $d_{ij}$ được tính bằng công thức Haversine với bán kính Trái Đất $R=6371$ km. Với mỗi fold $f$, các biên khoảng cách $a_b = Q_{b/K}(\mathcal{D}_{\mathrm{train}}^{(f)})$ ($b=1,\dots,K-1$) được xác định độc lập theo phân vị cặp luồng (pair-weighted) từ tập các thành phố huấn luyện $\mathcal{D}_{\mathrm{train}}^{(f)}$, với $a_0=0$ và $a_K=\infty$. Không sử dụng thông tin thành phố kiểm tra để thiết lập khoảng cách.)*

---

## 3.2 Spatial Units and Observational Resolution: Primary City-Level Benchmark (`M1_city`)
*(Tiếng Việt: **3.2. Đơn vị không gian và độ phân giải của quan sát: Cấu hình chuẩn cấp thành phố (`M1_city`)**)*

The dataset provided by the laboratory is organized on a per-city basis. Each city $c$ comprises a discrete set of census tracts and the observed positive interzonal OD pairs between them. **Tract** is the elementary spatial node unit of the neural network, while **city** is the unit for cross-validation data partitioning, zero-shot transfer learning, and performance evaluation.

For each target city $c$, the model predicts flow intensities across the entire observed positive support:

$$\Omega_c^+ = \left\{(i,j) : t_{ij} \ge 1\right\}$$

The primary benchmark employs a single distance-binned mobility distribution defined at the **city level**. The reference flow volume of city $c$ within distance bin $b$ is:

$$F_{c,b} = \sum_{(i,j) \in \Omega_c^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1} \le d_{ij} < a_b)$$

The normalized city-level distance distribution is:

$$Y_{D,c,b} = \frac{F_{c,b}}{\sum_{r=1}^K F_{c,r}}, \qquad \sum_{b=1}^K Y_{D,c,b} = 1$$

The resulting vector $\mathbf{Y}_{D,c} = [Y_{D,c,1}, \dots, Y_{D,c,K}]^T$ is used to calibrate the entire OD flow prediction of target city $c$. This constitutes the primary configuration of the study (`M1_city`).

*(Tiếng Việt: Bộ dữ liệu do Lab cung cấp được tổ chức theo từng thành phố. Mỗi thành phố $c$ bao gồm một tập các tract và các cặp OD dương giữa những tract đó. Tract là đơn vị không gian cơ sở của mô hình, trong khi city là đơn vị chia dữ liệu, thực hiện zero-shot transfer và đánh giá kết quả. Đối với mỗi thành phố mục tiêu, mô hình dự báo cường độ cho toàn bộ tập cặp OD được quan sát $\Omega_c^+ = \{(i,j):t_{ij}\geq1\}$. Các thử nghiệm chính sử dụng một phân phối di chuyển theo khoảng cách duy nhất ở cấp city. Tổng luồng tham chiếu của city $c$ trong khoảng cách $b$ là $F_{c,b} = \sum_{(i,j)\in\Omega_c^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1}\leq d_{ij}<a_b)$. Phân phối khoảng cách ở cấp city là $Y_{D,c,b} = F_{c,b} / \sum_{r=1}^{K}F_{c,r}$ với $\sum_{b=1}^{K}Y_{D,c,b}=1$. Vector $\mathbf{Y}_{D,c}$ được sử dụng để hiệu chỉnh toàn bộ dự báo OD của thành phố mục tiêu. Đây là cấu hình chính của nghiên cứu (`M1_city`).)*

---

## 3.3 Fine-Grained Spatial Resolution Variant: County-Level Observations (`M1_county`)
*(Tiếng Việt: **3.3. Biến thể quan sát chi tiết ở cấp county (`M1_county`)**)*

A supplementary experiment examines whether providing aggregate distance observations at a finer sub-metropolitan spatial resolution provides incremental predictive information. In this analysis, the tracts of each city are grouped by county.

County boundaries are obtained from the Database of Global Administrative Areas, version 4.1 [@gadm41]. Each tract is mapped to its encompassing county via a spatial point-in-polygon join between the tract centroid and the county polygon. If a centroid does not receive a valid `within` match—for example, because it lies on a polygon boundary or near a coastline—the implementation falls back to a nearest-polygon join in EPSG:5070 and accepts the assignment only when the centroid-to-polygon distance is at most 5 km; otherwise, execution stops with an error. Duplicate matches are resolved deterministically so that each tract receives exactly one county label. GADM is strictly utilized for this spatial grouping step; GADM is not the source of tract centroid coordinates, urban features, or OD flows.

Letting $g(i)$ denote the county assigned to tract $i$, OD pairs are grouped strictly by the **origin tract's county**:

$$\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c^+ : g(i) = \ell\right\}$$

Destination tract $j$ may belong to the same county or a different county within the metropolitan area. The distance-binned flow mass of county group $\ell$ is:

$$F_{c,\ell,b} = \sum_{(i,j) \in \Omega_{c,\ell}^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1} \le d_{ij} < a_b)$$

and its normalized distance distribution vector is:

$$Y_{D,c,\ell,b} = \frac{F_{c,\ell,b}}{\sum_{r=1}^K F_{c,\ell,r}}, \qquad \sum_{b=1}^K Y_{D,c,\ell,b} = 1$$

Because the input data are strictly bounded within the tracts of the city dataset provided by the laboratory, $\mathbf{Y}_{D,c,\ell}$ describes the outflow distance distribution of trips originating from the tracts of city $c$ assigned to county $\ell$. It does not represent total county-wide mobility outside the study city's spatial footprint.

Each distribution $\mathbf{Y}_{D,c,\ell}$ is used to calibrate OD pairs whose origin tract belongs to county $\ell$. The calibrated predictions from all county groups are then assembled into a complete OD prediction for the city:

$$\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \widehat{T}_{ij}^{\mathrm{CAL}} : (i,j) \in \Omega_{c,\ell}^+ \right\}$$

where $\mathcal{G}_c$ denotes the set of counties present in the dataset for city $c$.

Crucially, increasing observational resolution from city to county does not alter the evaluation scope. The model still reconstructs and is evaluated against the complete set of positive flows $\Omega_c^+$ for the target city; only the aggregate supervisory signal supplied during calibration becomes spatially more granular (`M1_county`).

*(Tiếng Việt: Một thí nghiệm bổ sung kiểm tra liệu quan sát có độ phân giải không gian chi tiết hơn city có mang lại thêm thông tin hay không. Trong thí nghiệm này, các tract của mỗi city được phân nhóm theo county. Ranh giới county được lấy từ GADM phiên bản 4.1 [@gadm41]. Mỗi tract được gán vào county tương ứng dựa trên vị trí tọa độ tâm trong polygon county. Nếu phép ghép `within` không cho kết quả hợp lệ—chẳng hạn khi tâm tract nằm trên biên polygon hoặc gần đường bờ—mã nguồn chuyển sang polygon gần nhất trong EPSG:5070 và chỉ chấp nhận kết quả khi khoảng cách không quá 5 km; nếu không, chương trình dừng và báo lỗi. Các kết quả trùng được xử lý xác định để mỗi tract chỉ có một nhãn county. GADM chỉ được sử dụng cho bước phân nhóm này; GADM không phải nguồn của tọa độ tract, đặc trưng đô thị hoặc luồng OD. Gọi $g(i)$ là county được gán cho tract $i$. Theo quy tắc được xác nhận từ mã nguồn, các cặp OD được phân nhóm theo county của origin: $\Omega_{c,\ell}^+ = \{(i,j)\in\Omega_c^+:g(i)=\ell\}$. Destination $j$ có thể thuộc cùng county hoặc một county khác. Phân phối khoảng cách của nhóm county $\ell$ được xác định bởi $F_{c,\ell,b} = \sum_{(i,j)\in\Omega_{c,\ell}^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1}\leq d_{ij}<a_b)$ và $Y_{D,c,\ell,b} = F_{c,\ell,b} / \sum_{r=1}^{K}F_{c,\ell,r}$. Do dữ liệu đầu vào vẫn được giới hạn trong các tract thuộc city do Lab cung cấp, $\mathbf{Y}_{D,c,\ell}$ mô tả phân phối khoảng cách của các chuyến đi xuất phát từ những tract của city được gán vào county $\ell$. Đại lượng này không nhất thiết đại diện cho toàn bộ hoạt động di chuyển của county bên ngoài phạm vi dữ liệu thành phố. Mỗi phân phối $\mathbf{Y}_{D,c,\ell}$ được dùng để hiệu chỉnh các cặp có origin thuộc county $\ell$. Sau đó, dự báo của tất cả nhóm county được ghép lại thành một dự báo OD hoàn chỉnh cho city: $\widehat{\mathbf{T}}_{c}^{\mathrm{county}} = \bigcup_{\ell\in\mathcal{G}_c} \{\widehat{T}_{ij}^{\mathrm{CAL}} : (i,j)\in\Omega_{c,\ell}^+\}$, trong đó $\mathcal{G}_c$ là tập county xuất hiện trong dữ liệu của city $c$. Như vậy, việc tăng độ phân giải quan sát từ city lên county không làm thay đổi phạm vi dự báo. Mô hình vẫn tái tạo và đánh giá toàn bộ OD của thành phố trên $\Omega_c^+$; chỉ thông tin tổng hợp được cung cấp cho bước hiệu chỉnh trở nên chi tiết hơn về mặt không gian (`M1_county`).)*

---

## 3.4 Zero-Shot Flow Intensity Calibration via Distance Distribution
*(Tiếng Việt: **3.4. Hiệu chỉnh dự báo zero-shot bằng phân phối khoảng cách**)*

The neural backbone is trained on source cities and kept strictly frozen prior to evaluation on the target city. For each positive pair $(i,j) \in \Omega_c^+$, the ZTNB model generates an initial zero-shot flow intensity prediction:

$$\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij} \mid T_{ij} \ge 1]$$

These predictions constitute baseline $M_0$. Baseline $M_0$ utilizes target-city urban context features and inter-tract distances, but has no access to $Y_D$ or target reference OD intensities.

### 3.4.1 Primary Calibration at the City Level (`M1_city`)

The total flow mass predicted by the baseline in distance bin $b$ is:

$$\widehat{F}_{c,b}^{\mathrm{ZS}} = \sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}} \mathbb{I}(a_{b-1} \le d_{ij} < a_b)$$

Letting:

$$\widehat{S}_{c}^{\mathrm{ZS}} = \sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}}$$

denote total predicted flow intensity across target city $c$, the implied distance distribution predicted by the baseline is:

$$\widehat{Y}_{D,c,b}^{\mathrm{ZS}} = \frac{\widehat{F}_{c,b}^{\mathrm{ZS}}}{\widehat{S}_{c}^{\mathrm{ZS}}}$$

Because compact cities may contain zero candidate OD pairs in outer distance bins within $\Omega_c^+$, the set of active distance bins is defined as:

$$\mathcal{A}_c = \left\{ b : \widehat{Y}_{D,c,b}^{\mathrm{ZS}} > 0 \right\}$$

The target distance observation is conditioned on active distance bins:

$$p_{c,b}^{\mathrm{cond}} = \frac{Y_{D,c,b} \mathbb{I}(b \in \mathcal{A}_c)}{\sum_{r \in \mathcal{A}_c} Y_{D,c,r}}$$

This conditioning ensures that calibration reallocates mass strictly among distance bins containing at least one positive OD pair in the target city's support.

For $b \in \mathcal{A}_c$, the raw calibration ratio and soft weight are:

$$r_{c,b} = \frac{p_{c,b}^{\mathrm{cond}}}{\widehat{Y}_{D,c,b}^{\mathrm{ZS}}}, \qquad w_{c,b}(q) = r_{c,b}^q, \quad q \in [0, 1]$$

To strictly conserve total predicted flow mass, the weight is normalized by:

$$Z_c(q) = \sum_{r \in \mathcal{A}_c} \widehat{Y}_{D,c,r}^{\mathrm{ZS}} w_{c,r}(q), \qquad s_{c,b}(q) = \frac{w_{c,b}(q)}{Z_c(q)}$$

The calibrated prediction for pair $(i,j)$ under `M1_city` is:

$$\widehat{T}_{ij}^{M1_{\mathrm{city}}} = s_{c, b(i,j)}(q) \cdot \widehat{T}_{ij}^{\mathrm{ZS}}$$

where $b(i,j)$ indexes the distance bin containing $d_{ij}$.

In the primary benchmark, $q=1$ is pre-specified and locked prior to evaluation. $q=0$ reverts identically to baseline $M_0$ because all scaling factors equal 1.

The normalization mechanism strictly preserves total predicted flow intensity:

$$\sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{M1_{\mathrm{city}}} = \sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}}$$

At $q=1$, the calibrated model's implied distance distribution matches the conditioned target distribution:

$$\widehat{Y}_{D,c,b}^{M1_{\mathrm{city}}} = p_{c,b}^{\mathrm{cond}}$$

When all distance bins in $\mathbf{Y}_{D,c}$ are active ($\mathcal{A}_c = \{1, \dots, K\}$), $p_{c,b}^{\mathrm{cond}} = Y_{D,c,b}$, and the calibrated distribution matches raw $\mathbf{Y}_{D,c}$ directly.

### 3.4.2 Spatial Resolution Variant at the County Level (`M1_county`)

In the spatial resolution experiment, the above procedure is applied independently to each origin-county group:

$$\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c^+ : g(i) = \ell\right\}$$

For each county $\ell$, the algorithm identifies the active bin set $\mathcal{A}_{c,\ell}$, conditions the target observation to $p_{c,\ell,b}^{\mathrm{cond}}$, and computes:

$$w_{c,\ell,b}(q) = \left(\frac{p_{c,\ell,b}^{\mathrm{cond}}}{\widehat{Y}_{D,c,\ell,b}^{\mathrm{ZS}}}\right)^q, \qquad s_{c,\ell,b}(q) = \frac{w_{c,\ell,b}(q)}{\sum_{r \in \mathcal{A}_{c,\ell}} \widehat{Y}_{D,c,\ell,r}^{\mathrm{ZS}} w_{c,\ell,r}(q)}$$

Predictions are scaled by the factor corresponding to the origin tract's county:

$$\widehat{T}_{ij}^{M1_{\mathrm{county}}} = s_{c, g(i), b(i,j)}(q) \cdot \widehat{T}_{ij}^{\mathrm{ZS}}$$

Because normalization is executed separately per origin county, total predicted flow originating from each county is conserved ($\sum_{(i,j)\in\Omega_c^+:g(i)=\ell} \widehat{T}_{ij}^{M1_{\mathrm{county}}} = \sum_{(i,j)\in\Omega_c^+:g(i)=\ell} \widehat{T}_{ij}^{\mathrm{ZS}}$). When aggregated, total city-wide flow is also preserved.

### 3.4.3 Invariant Mathematical Properties

The calibration operator possesses three foundational properties:
1. **Analytic Post-Processing**: No neural parameters of the GNN or ZTNB heads are updated on the target city.
2. **Support Invariance**: $\Omega_c^+(M_0) = \Omega_c^+(M1_{\mathrm{city}}) = \Omega_c^+(M1_{\mathrm{county}})$. The method neither discovers unobserved links nor assigns zero flows to missing pairs.
3. **Intra-Bin Rank Invariance**: All predictions within the same distance bin are multiplied by an identical positive scalar $s_{c,b}(q) > 0$. Consequently, the relative ranking among pairs within any distance bin is strictly invariant (for non-degenerate groups with sufficient pairs, Kendall's rank correlation before and after calibration is identically $\tau = 1.00000000$).

*(Tiếng Việt: Mô hình được huấn luyện trên các thành phố nguồn và được đóng băng trước khi đánh giá trên thành phố mục tiêu. Với mỗi cặp $(i,j)\in\Omega_c^+$, mô hình ZTNB tạo ra dự báo cường độ zero-shot $\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij}\mid T_{ij}\geq1]$, tạo thành baseline $M_0$. Tổng luồng dự báo trong khoảng $b$ là $\widehat{F}_{c,b}^{\mathrm{ZS}} = \sum_{(i,j)\in\Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}} \mathbb{I}(a_{b-1}\leq d_{ij}<a_b)$ và tổng cường độ dự báo của thành phố là $\widehat{S}_{c}^{\mathrm{ZS}} = \sum_{(i,j)\in\Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}}$. Phân phối khoảng cách ngầm định bởi baseline là $\widehat{Y}_{D,c,b}^{\mathrm{ZS}} = \widehat{F}_{c,b}^{\mathrm{ZS}} / \widehat{S}_{c}^{\mathrm{ZS}}$. Tập khoảng hoạt động là $\mathcal{A}_c = \{b : \widehat{Y}_{D,c,b}^{\mathrm{ZS}} > 0\}$ và phân phối mục tiêu điều kiện hóa là $p_{c,b}^{\mathrm{cond}} = Y_{D,c,b}\mathbb{I}(b\in\mathcal{A}_c) / \sum_{r\in\mathcal{A}_c}Y_{D,c,r}$. Với $b\in\mathcal{A}_c$, tỷ lệ $r_{c,b} = p_{c,b}^{\mathrm{cond}} / \widehat{Y}_{D,c,b}^{\mathrm{ZS}}$ và trọng số $w_{c,b}(q) = r_{c,b}^q$ ($q\in[0,1]$, $q=1.0$ chuẩn). Hệ số chuẩn hóa $s_{c,b}(q) = w_{c,b}(q) / Z_c(q)$ với $Z_c(q) = \sum_{r\in\mathcal{A}_c}\widehat{Y}_{D,c,r}^{\mathrm{ZS}}w_{c,r}(q)$. Dự báo sau hiệu chỉnh là $\widehat{T}_{ij}^{M1_{\mathrm{city}}} = s_{c,b(i,j)}(q)\widehat{T}_{ij}^{\mathrm{ZS}}$. Chuẩn hóa bảo toàn chính xác tổng cường độ dự báo $\sum_{\Omega_c^+}\widehat{T}_{ij}^{M1_{\mathrm{city}}} = \sum_{\Omega_c^+}\widehat{T}_{ij}^{\mathrm{ZS}}$. Khi $q=1$, phân phối sau hiệu chỉnh khớp với $p_{c,b}^{\mathrm{cond}}$ (khớp raw $\mathbf{Y}_{D,c}$ khi mọi bin đều hoạt động). Đối với biến thể $M1_{\mathrm{county}}$, việc hiệu chỉnh áp dụng độc lập cho từng origin county $\Omega_{c,\ell}^+ = \{(i,j)\in\Omega_c^+:g(i)=\ell\}$, bảo toàn tổng lưu lượng xuất phát của từng county. Cả hai cấu hình đều là phép hậu xử lý giải tích, giữ nguyên tập hỗ trợ $\Omega_c^+$ và bảo toàn thứ hạng nội khoảng ($\tau = 1.0$ đối với nhóm không suy biến).)*

---

## 3.5 OD Flow Intensity Modeling via Zero-Truncated Negative Binomial (ZTNB)
*(Tiếng Việt: **3.5. Mô hình hóa cường độ OD bằng ZTNB**)*

### 3.5.1 Frozen neural backbone and training configuration

Within each fold, the 26 tract features are standardized using statistics fitted exclusively on the 35 training cities and then applied unchanged to the validation and test cities. The spatial graph connects tract centroids within a 5 km Haversine radius, includes self-loops, and represents neighborhood relations in both directions. Any tract without a neighbor inside the radius is connected to its nearest tract to avoid isolated nodes.

The frozen backbone contains two graph neural-network layers with hidden dimension 64 and dropout 0.1. Its pairwise decoder receives the origin and destination embeddings together with $\log(1+d_{ij})$ and a log gravity-prior term. Models are trained for at most 200 epochs with AdamW (learning rate $2\times10^{-3}$, weight decay $10^{-4}$) [@loshchilov2019adamw], gradient clipping at 5.0, a `ReduceLROnPlateau` scheduler (factor 0.5, patience 4), and early stopping with patience 15 based on validation CPC. After model selection, all backbone and output-head parameters remain fixed during target-city calibration.

*(Tiếng Việt: Trong mỗi fold, 26 đặc trưng tract được chuẩn hóa bằng các thống kê chỉ fit trên 35 thành phố huấn luyện, sau đó áp dụng nguyên trạng cho tập validation và test. Đồ thị không gian nối các tâm tract trong bán kính Haversine 5 km, có self-loop và biểu diễn quan hệ láng giềng theo hai chiều. Tract không có láng giềng trong bán kính được nối với tract gần nhất để tránh nút cô lập. Backbone gồm hai lớp GNN, chiều ẩn 64 và dropout 0.1. Pairwise decoder nhận embedding của origin và destination cùng với $\log(1+d_{ij})$ và log gravity prior. Mô hình được huấn luyện tối đa 200 epoch bằng AdamW (learning rate $2\times10^{-3}$, weight decay $10^{-4}$) [@loshchilov2019adamw], gradient clipping 5.0, scheduler `ReduceLROnPlateau` (factor 0.5, patience 4) và early stopping patience 15 theo validation CPC. Sau bước chọn mô hình, toàn bộ tham số backbone và output head được giữ cố định khi hiệu chỉnh trên thành phố mục tiêu.)*

### 3.5.2 Zero-truncated negative binomial likelihood and inference

Because the evaluation and training samples consist exclusively of OD pairs with positive flows ($t_{ij} \ge 1$), flow intensities are modeled using the Zero-Truncated Negative Binomial distribution [@grogger1991truncated; @hilbe2011negative]. For $t_{ij} \ge 1$, the conditional likelihood is:

$$P(T_{ij} = t_{ij} \mid T_{ij} \ge 1) = \frac{P_{\mathrm{NB}}(T_{ij} = t_{ij}; \mu_{ij}, \phi)}{1 - P_{\mathrm{NB}}(T_{ij} = 0; \mu_{ij}, \phi)}$$

where the neural network predicts the unconstrained base Negative Binomial mean $\mu_{ij} > 0$, and $\phi > 0$ is the dispersion parameter. The zero-probability of the base Negative Binomial is:

$$p_{0,ij} = \left( \frac{\phi}{\mu_{ij} + \phi} \right)^\phi$$

The training loss is the negative log-likelihood of the ZTNB distribution:

$$\mathcal{L}_{\mathrm{ZTNB}} = -\frac{1}{|\Omega_c^+|} \sum_{(i,j) \in \Omega_c^+} \left[ \log P_{\mathrm{NB}}(t_{ij}; \mu_{ij}, \phi) - \log(1 - p_{0,ij}) \right]$$

At inference time, zero-shot flow predictions do not use $\mu_{ij}$ directly. Instead, the model outputs the **conditional expected mean**:

$$\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij} \mid T_{ij} \ge 1] = \frac{\mu_{ij}}{1 - p_{0,ij}}$$

As the expectation of a count distribution, $\widehat{T}_{ij}^{\mathrm{ZS}}$ is a strictly positive real value and is not required to be an integer. ZTNB strictly models flow volume conditioned on positive links $\Omega_c^+$; it does not predict link existence or treat unobserved pairs as zero flows [@grogger1991truncated; @hilbe2011negative].

*(Tiếng Việt: Do tập dữ liệu chỉ bao gồm những cặp OD có luồng dương, cường độ luồng được mô hình hóa bằng phân phối negative binomial cắt tại 0 [@grogger1991truncated; @hilbe2011negative]. Với $t_{ij}\geq1$, likelihood là: $P(T_{ij}=t_{ij}\mid T_{ij}\geq1) = P_{\mathrm{NB}}(T_{ij}=t_{ij};\mu_{ij},\phi) / (1-P_{\mathrm{NB}}(T_{ij}=0;\mu_{ij},\phi))$. Trong đó, mạng nơ-ron dự báo trung bình chưa cắt $\mu_{ij}>0$, còn $\phi>0$ là tham số phân tán. Xác suất bằng 0 của phân phối negative binomial cơ sở là $p_{0,ij} = (\phi/(\mu_{ij}+\phi))^\phi$. Hàm mất mát huấn luyện là negative log-likelihood của phân phối ZTNB: $\mathcal{L}_{\mathrm{ZTNB}} = -\frac{1}{|\Omega_c^+|} \sum_{(i,j)\in\Omega_c^+} [\log P_{\mathrm{NB}}(t_{ij};\mu_{ij},\phi) - \log(1-p_{0,ij})]$. Tại thời điểm suy luận, dự báo zero-shot không sử dụng trực tiếp $\mu_{ij}$. Thay vào đó, mô hình sử dụng kỳ vọng có điều kiện: $\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij}\mid T_{ij}\geq1] = \frac{\mu_{ij}}{1-p_{0,ij}}$. Do là kỳ vọng của phân phối, $\widehat{T}_{ij}^{\mathrm{ZS}}$ là một giá trị thực dương và không bắt buộc phải là số nguyên. ZTNB chỉ mô hình hóa cường độ của những cặp thuộc $\Omega_c^+$; mô hình không dự báo sự tồn tại của các cặp OD chưa quan sát và không xem chúng là các luồng bằng 0.)*

Figure 1 summarizes the support-conditioned oracle calibration framework, separating cross-city model training, frozen target-city inference, and the oracle aggregate intervention.

*(Tiếng Việt: Hình 1 tóm tắt framework hiệu chỉnh oracle có điều kiện theo support, đồng thời phân tách rõ quá trình huấn luyện cross-city, suy luận trên thành phố mục tiêu bằng mô hình đóng băng và can thiệp thông tin tổng hợp oracle.)*

![Figure 1](figures/fig1_oracle_calibration_framework.svg)
**Figure 1. Support-conditioned oracle calibration framework.** The cross-city model $M_0$ is trained on source cities and frozen before target-city inference. For a held-out target city, $M_0$ first produces baseline intensities $\widehat{\mathbf{T}}^{(0)}$ on the known positive support $\Omega_c^+$. The oracle distance-binned distribution $\mathbf{Y}_{D,c}$ is deterministically derived from the same target-city positive ground-truth OD flows used for evaluation and is introduced only at inference time. Bin-specific scaling factors reallocate predicted mass across distance intervals to obtain $\widehat{\mathbf{T}}^{(1)}$ without updating model parameters or creating new OD links. The schematic represents an oracle information intervention, not an independently collected external telemetry pipeline.

*(Tiếng Việt: **Hình 1. Framework hiệu chỉnh oracle có điều kiện theo support.** Mô hình cross-city $M_0$ được huấn luyện trên các thành phố nguồn và đóng băng trước khi suy luận trên thành phố mục tiêu. Đối với một thành phố mục tiêu chưa từng xuất hiện trong huấn luyện, $M_0$ trước hết tạo ra dự báo cường độ baseline $\widehat{\mathbf{T}}^{(0)}$ trên tập hỗ trợ dương đã biết $\Omega_c^+$. Phân phối theo nhóm khoảng cách oracle $\mathbf{Y}_{D,c}$ được xác định trực tiếp từ chính các luồng OD ground-truth dương của thành phố mục tiêu đang được sử dụng để đánh giá và chỉ được đưa vào tại thời điểm suy luận. Các hệ số theo bin tái phân bổ khối lượng dự báo giữa các khoảng cự ly để tạo $\widehat{\mathbf{T}}^{(1)}$ mà không cập nhật tham số mô hình hoặc tạo liên kết OD mới. Sơ đồ biểu diễn một can thiệp thông tin oracle, không phải pipeline telemetry bên ngoài được thu thập độc lập.)*

---

## 3.6 Cross-City Evaluation Protocol and Statistical Inference
*(Tiếng Việt: **3.6. Giao thức đánh giá cross-city và suy luận thống kê**)*

### 3.6.1 5-Fold Cross-City Validation Scheme
The empirical benchmark is structured around a strict 5-fold cross-validation protocol over $N=50$ U.S. metropolitan areas. In each fold, 35 cities are assigned to the training set, 5 cities to the validation set, and the remaining 10 held-out cities to the test set. Every city appears in the test partition exactly once, ensuring comprehensive 50-city out-of-sample evaluation coverage.

Data partitioning is conducted strictly at the **city level** rather than at the tract or OD pair level. Consequently, all tracts and OD pairs belonging to a given city reside exclusively in one of the three splits (train, validation, or test) within any fold. This design prevents spatial data leakage and ensures that the neural model never observes target-city representations during training.

Distance bin edges are calculated independently for each fold using only interzonal OD pairs from the training cities. Following training completion, backbone model parameters are permanently frozen prior to target-city inference.

For each target city, three primary model conditions are evaluated:
- $M_0$: Zero-shot predicted flows without access to $Y_D$;
- $M1_{\mathrm{city}}$: Analytically calibrated flows using a single oracle $Y_D$ at the city level (Primary Benchmark);
- $M1_{\mathrm{county}}$: Analytically calibrated flows using multiple oracle $Y_D$ grouped by origin county (Spatial Resolution Variant).

The comparison between $M_0$ and $M1_{\mathrm{city}}$ represents the primary experiment designed to evaluate whether target distance distributions provide incremental information for zero-shot reconstruction (RQ1). The comparison between $M1_{\mathrm{city}}$ and $M1_{\mathrm{county}}$ provides empirical evidence for the spatial observational resolution aspect of RQ2.

Across all configurations, predictions are evaluated on the exact same observed positive interzonal support $\Omega_c^+$ for the entire city.

*(Tiếng Việt: Nghiên cứu sử dụng dữ liệu của 50 thành phố và áp dụng giao thức cross-validation theo thành phố gồm 5 folds. Trong mỗi fold, 35 thành phố được sử dụng để huấn luyện, 5 thành phố để validation và 10 thành phố còn lại để kiểm tra. Mỗi thành phố xuất hiện trong tập kiểm tra đúng một lần, vì vậy kết quả tổng hợp cuối cùng bao phủ toàn bộ 50 thành phố. Việc chia dữ liệu được thực hiện ở cấp city thay vì cấp tract hoặc cặp OD. Do đó, tất cả tract và cặp OD của một thành phố chỉ thuộc một trong ba tập training, validation hoặc test trong mỗi fold. Thiết kế này ngăn việc mô hình nhìn thấy một phần dữ liệu của thành phố mục tiêu trong quá trình huấn luyện. Các biên khoảng cách được tính riêng cho từng fold và chỉ sử dụng khoảng cách của các cặp OD thuộc tập thành phố huấn luyện. Sau khi huấn luyện hoàn tất, tham số của mô hình được đóng băng trước khi dự báo trên các thành phố kiểm tra. Đối với mỗi thành phố mục tiêu, ba cấu hình được phân biệt: $M_0$ (dự báo zero-shot không sử dụng $Y_D$), $M1_{\mathrm{city}}$ (hiệu chỉnh bằng một $Y_D$ oracle ở cấp city), và $M1_{\mathrm{county}}$ (hiệu chỉnh bằng nhiều $Y_D$ oracle được phân nhóm theo county). So sánh giữa $M_0$ và $M1_{\mathrm{city}}$ là thí nghiệm chính nhằm trả lời liệu phân phối khoảng cách của thành phố mục tiêu có bổ sung thông tin cho dự báo zero-shot hay không (RQ1). So sánh giữa $M1_{\mathrm{city}}$ và $M1_{\mathrm{county}}$ cung cấp bằng chứng cho khía cạnh độ phân giải không gian của quan sát trong RQ2. Trong tất cả cấu hình, mô hình dự báo và được đánh giá trên cùng tập hỗ trợ dương $\Omega_c^+$ của toàn thành phố.)*

---

### 3.6.2 Primary Evaluation Metric: Common Part of Commuters (CPC)

The primary accuracy metric is the Common Part of Commuters (CPC), computed on positive interzonal pairs:

$$\operatorname{CPC}_c = \frac{2 \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \min\left(t_{ij}^{\mathrm{GT}}, \widehat{T}_{ij}\right)}{\sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{ij}^{\mathrm{GT}} + \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{ij}}$$

where $\Omega_{c,\mathrm{inter}}^+ = \left\{(i,j) \in \Omega_c^+ : i \ne j\right\}$. CPC is bounded in $[0, 1]$, where values closer to 1 denote greater agreement between predicted and ground-truth flows. CPC is standard in spatial mobility modeling and OD reconstruction benchmarks [@lenormand2016comparison].

The incremental information value of $Y_D$ for city $c$ is measured by the paired gain:

$$\Delta\operatorname{CPC}_c = \operatorname{CPC}_c(M1_{\mathrm{city}}) - \operatorname{CPC}_c(M_0)$$

A positive value indicates that conditioning on $Y_D$ improves reconstruction accuracy over the zero-shot baseline on the same city, same support, and identical pre-trained network.

*(Tiếng Việt: Chỉ số chính là Common Part of Commuters (CPC), được tính trên các cặp OD liên vùng thuộc tập hỗ trợ dương: $\operatorname{CPC}_c = 2\sum_{(i,j)\in\Omega_{c,\mathrm{inter}}^+} \min(t_{ij}, \widehat{T}_{ij}) / (\sum_{(i,j)} t_{ij} + \sum_{(i,j)} \widehat{T}_{ij})$, trong đó $\Omega_{c,\mathrm{inter}}^+ = \{(i,j)\in\Omega_c^+ : i \ne j\}$. CPC nằm trong khoảng từ 0 đến 1; giá trị lớn hơn biểu thị mức độ trùng khớp cao hơn giữa luồng dự báo và luồng tham chiếu [@lenormand2016comparison]. Hiệu quả bổ sung của $Y_D$ tại thành phố $c$ được xác định bằng chênh lệch ghép cặp: $\Delta\operatorname{CPC}_c = \operatorname{CPC}_c(M1_{\mathrm{city}}) - \operatorname{CPC}_c(M_0)$. Giá trị dương cho thấy việc sử dụng $Y_D$ cải thiện kết quả so với dự báo zero-shot trên cùng thành phố, cùng tập hỗ trợ và cùng mô hình nền.)*

---

### 3.6.3 Aggregation Across Model Seeds and Cities

To account for stochasticity in neural initialization and training optimization, each configuration is trained across three independent model seeds:

$$\mathcal{S} = \{1, 10, 100\}$$

For each city, paired performance differences are computed per seed and then averaged:

$$\overline{\Delta\operatorname{CPC}}_c = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left[ \operatorname{CPC}_{c,s}(M1_{\mathrm{city}}) - \operatorname{CPC}_{c,s}(M_0) \right]$$

The population-level headline estimand is defined as the macro-average across all 50 cities:

$$\overline{\Delta\operatorname{CPC}} = \frac{1}{50} \sum_{c=1}^{50} \overline{\Delta\operatorname{CPC}}_c$$

Macro-averaging assigns equal weight to each metropolitan area regardless of network size, number of tracts, or total travel demand. The primary estimand represents the average expected gain across diverse cities, rather than an unweighted average pooled across millions of OD pairs.

*(Tiếng Việt: Mỗi cấu hình được chạy với ba model seeds: $\mathcal{S}=\{1,10,100\}$. Đối với mỗi thành phố, chênh lệch CPC trước hết được tính riêng cho từng seed và sau đó lấy trung bình: $\overline{\Delta\operatorname{CPC}}_c = \frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}} [\operatorname{CPC}_{c,s}(M1_{\mathrm{city}}) - \operatorname{CPC}_{c,s}(M_0)]$. Hiệu quả tổng thể được tính bằng macro-average trên 50 thành phố: $\overline{\Delta\operatorname{CPC}} = \frac{1}{50}\sum_{c=1}^{50}\overline{\Delta\operatorname{CPC}}_c$. Cách tổng hợp này trao trọng số như nhau cho mỗi thành phố, bất kể số tract, số cặp OD hoặc tổng số chuyến đi của thành phố đó. Vì vậy, estimand chính là mức cải thiện trung bình giữa các thành phố, không phải mức cải thiện trung bình giữa tất cả cặp OD gộp chung.)*

---

### 3.6.4 Uncertainty Quantification and Statistical Hypothesis Testing

The 95% confidence interval for the population mean improvement is estimated via fold-stratified city-level bootstrap ($B=10,000$ resamples) [@efron1993bootstrap]. In each resample, cities are sampled with replacement within their fold strata from the set of city deltas $\left\{\overline{\Delta\operatorname{CPC}}_c\right\}_{c=1}^{50}$, and the macro-average is recomputed. Sampling at the city level maintains the city as the fundamental unit of statistical inference and avoids treating non-independent OD pairs within the same city as independent observations.

A two-sided paired Wilcoxon signed-rank test [@wilcoxon1945ranking] is conducted across the 50 city-level deltas:

$$\left\{ \overline{\Delta\operatorname{CPC}}_c \right\}_{c=1}^{50}$$

The null hypothesis tests whether the median paired difference between $M1_{\mathrm{city}}$ and $M_0$ equals zero. This non-parametric test evaluates whether the observed directional improvement represents a systematic shift rather than random fluctuation around zero.

*(Tiếng Việt: Khoảng tin cậy 95% của mức cải thiện trung bình được ước lượng bằng bootstrap phân tầng theo fold ở cấp city ($B=10,000$) [@efron1993bootstrap]. Trong mỗi lần bootstrap, các thành phố được lấy mẫu có hoàn lại trong từng fold từ tập các giá trị $\overline{\Delta\operatorname{CPC}}_c$, sau đó tính lại macro-average. Việc lấy mẫu ở cấp city giữ city là đơn vị suy luận thống kê và tránh xem hàng triệu cặp OD trong cùng thành phố như các quan sát độc lập. Kiểm định Wilcoxon signed-rank ghép cặp [@wilcoxon1945ranking] được áp dụng trên 50 giá trị $\{\overline{\Delta\operatorname{CPC}}_c\}_{c=1}^{50}$. Giả thuyết không cho rằng phân phối chênh lệch giữa $M1_{\mathrm{city}}$ và $M_0$ có trung vị bằng 0. Kiểm định này bổ sung cho khoảng tin cậy bootstrap bằng cách đánh giá liệu hướng cải thiện quan sát được có phù hợp với biến động ngẫu nhiên quanh 0 hay không.)*

---

### 3.6.5 Robustness and Diagnostic Stress Tests

Supplementary experiments investigate the operational boundaries and mechanisms governing the primary result:
1. **Distance Resolution ($K$-Sensitivity)**: Varying distance partitions across $K \in \{2,4,6,8,10,12,14,16,18,20\}$. The nine secondary configurations are compared with the locked $K=8$ anchor using Holm's step-down family-wise error correction [@holm1979sequential].
2. **Spatial Observational Granularity**: Comparing city-level ($M1_{\mathrm{city}}$) against origin-county grouped observations ($M1_{\mathrm{county}}$).
3. **Observational Noise Tolerance**: Adding controlled synthetic Total Variation noise $\epsilon \in [0\%, 5\%]$ to $Y_D$ to identify breakdown thresholds.
4. **Spatial Semantic Ordering**: Permuting the bin order of $Y_D$ to test whether distance alignment is mandatory.
5. **Target Specificity Placebos**: Applying dose-matched donor distributions from incorrect cities and fold training-mean profiles.
6. **Initialization Stability**: Replicating across independent model initializations (Seeds 1, 10, 100).
7. **Architectural Generality**: Evaluating the Urban GNN and Node MLP neural backbones together with a classical gravity baseline.

*(Tiếng Việt: Các phân tích bổ sung được thiết kế để kiểm tra phạm vi và cơ chế của kết quả chính: (1) Độ phân giải khoảng cách: thay đổi $K \in \{2,4,6,8,10,12,14,16,18,20\}$ và so sánh chín cấu hình phụ với mốc khóa $K=8$ bằng hiệu chỉnh Holm step-down [@holm1979sequential]; (2) Độ phân giải không gian: so sánh $M1_{\mathrm{city}}$ với $M1_{\mathrm{county}}$; (3) Chất lượng quan sát: thêm nhiễu Total Variation có kiểm soát vào $Y_D$; (4) Thứ tự khoảng cách: hoán vị các khoảng của $Y_D$; (5) Tính đặc thù theo thành phố: sử dụng phân phối của thành phố khác trong matched-placebo; (6) Độ bền theo khởi tạo: lặp lại với các model seeds 1, 10, 100; và (7) Độ bền theo kiến trúc: đánh giá Urban GNN và Node MLP cùng với gravity baseline cổ điển. Các phân tích này không thay đổi estimand chính; chúng xác định ranh giới vận hành và cơ chế khoa học của phương pháp.)*

---

### 3.6.6 County-Level Spatial Observational Resolution Protocol
*(Tiếng Việt: **3.6.6. Phân tích độ phân giải không gian theo county**)*

Across the 50 urban benchmark datasets, 39 metropolitan areas contain tracts that map to a single county, whereas 11 metropolitan areas contain tracts distributed across two to seven counties (the multi-county group comprises Kansas City, New York, Dallas, Denver, Omaha, Tulsa, Detroit, Chicago, Boston, Milwaukee, and Atlanta).

For the 39 single-county cities, all origin tracts belong to the exact same county group. Consequently, the county-level distance observation and the city-wide observation are mathematically equivalent:

$$\mathbf{Y}_{D,c,\ell} = \mathbf{Y}_{D,c},$$

yielding an exact mathematical identity:

$$M1_{\mathrm{county}} \equiv M1_{\mathrm{city}}, \qquad \Delta\operatorname{CPC}_{\mathrm{res},c} = 0$$

where the incremental gain from spatial resolution refinement is defined as:

$$\Delta\operatorname{CPC}_{\mathrm{res},c} = \operatorname{CPC}_c(M1_{\mathrm{county}}) - \operatorname{CPC}_c(M1_{\mathrm{city}})$$

Thus, the 39 single-county cities serve as an invariant algorithmic sanity check: partitioning a city into a single trivial group cannot alter prediction outputs.

Empirical evidence regarding the informational benefit of county-level resolution arises from the 11 multi-county cities. For these metropolitan areas, each observation $\mathbf{Y}_{D,c,\ell}$ is constructed from trips originating in county $\ell$, while the final prediction is assembled and evaluated over the full positive support of the target city.

Results are reported across two evaluation tiers:
1. **Pooled Benchmark Tier ($N=50$ cities)**: Reflecting the expected average effect of providing county-level observations across the entire heterogeneous benchmark;
2. **Multi-County Focus Tier ($n=11$ cities)**: Reflecting the empirical effect specifically where county-level grouping supplies genuine spatial granularity.

Because 39 cities produce structural zeros by construction, scientific interpretation of the incremental value of county-resolved observations is grounded primarily in the 11 multi-county cities.

*(Tiếng Việt: Trong 50 bộ dữ liệu đô thị, 39 thành phố chỉ chứa các tract được gán vào một county, trong khi 11 thành phố chứa tract thuộc từ hai đến bảy counties. Nhóm multi-county gồm Kansas City, New York, Dallas, Denver, Omaha, Tulsa, Detroit, Chicago, Boston, Milwaukee và Atlanta. Đối với 39 single-county cities, tất cả origin tract thuộc cùng một nhóm county. Vì vậy, phân phối quan sát theo county và phân phối quan sát theo city là tương đương: $\mathbf{Y}_{D,c,\ell} = \mathbf{Y}_{D,c}$, dẫn đến $M1_{\mathrm{county}} \equiv M1_{\mathrm{city}}$ và $\Delta\operatorname{CPC}_{\mathrm{res},c}=0$. Trong đó, hiệu quả của việc tăng độ phân giải được định nghĩa là $\Delta\operatorname{CPC}_{\mathrm{res},c} = \operatorname{CPC}_c(M1_{\mathrm{county}}) - \operatorname{CPC}_c(M1_{\mathrm{city}})$. Do đó, 39 single-county cities đóng vai trò như một kiểm tra bất biến của thuật toán: việc chia một city thành đúng một nhóm không được làm thay đổi kết quả. Thông tin thực nghiệm về lợi ích của độ phân giải county đến từ 11 multi-county cities. Đối với các thành phố này, mỗi phân phối $\mathbf{Y}_{D,c,\ell}$ được xây dựng từ những chuyến đi có origin thuộc county $\ell$, còn dự báo cuối cùng vẫn được ghép và đánh giá trên toàn bộ positive support của city. Kết quả được báo cáo theo hai phạm vi: (1) kết quả pooled trên toàn bộ 50 thành phố, phản ánh hiệu quả trung bình của việc cung cấp quan sát county-level trên toàn benchmark; và (2) kết quả riêng trên 11 multi-county cities, phản ánh hiệu quả tại những thành phố mà county-level thực sự cung cấp độ phân giải bổ sung. Do 39 thành phố tạo ra chênh lệch bằng 0 theo cấu trúc, diễn giải về giá trị của county-level observation chủ yếu dựa trên nhóm 11 multi-county cities.)*
