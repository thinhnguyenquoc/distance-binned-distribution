# Section 3: Data Sources, Spatial Units, and Methodology

*(Tiếng Việt: **Mục 3: Nguồn dữ liệu, đơn vị không gian và phương pháp luận**)*

---

## 3.1 Notation and Data Inputs
*(Tiếng Việt: **3.1. Ký hiệu và dữ liệu đầu vào**)*

Let $c$ denote a metropolitan area (city) and let $\mathcal{V}_c$ denote the discrete set of spatial units (census tracts) partitioning that city. Each ordered pair $(i,j)$ with $i,j \in \mathcal{V}_c$ represents a potential origin–destination (OD) pair, where the same spatial partition serves as both origins and destinations. Table 1 summarizes the core mathematical notation, underlying data sources, and the role and availability of each quantity during model training, zero-shot inference, calibration, and evaluation. Specialized notation used exclusively for secondary stress tests (such as synthetic noise perturbations, donor placebos, or alternate neural backbones) is defined locally within the corresponding experimental sections.

### Table 1: Core notation, data sources, and information availability

| Symbol | Description | Source / Role |
| :--- | :--- | :--- |
| $c$ | Metropolitan area (city) index ($c \in \mathcal{C}$) | City identifier |
| $\mathcal{C}$ | Benchmark set of 50 U.S. metropolitan areas ($|\mathcal{C}| = 50$) | Experimental setting |
| $\mathcal{V}_c$ | Discrete set of census tract spatial units in city $c$ | Static target-city input |
| $N_c$ | Number of spatial units in city $c$, $N_c = |\mathcal{V}_c|$ | Static target-city input |
| $\mathcal{G}_c$ | Spatial radius graph $(\mathcal{V}_c, \mathcal{E}_c)$ with 5 km radius threshold | Computed from geography |
| $i, j$ | Origin and destination tract indices, both belonging to $\mathcal{V}_c$ | Spatial unit index |
| $\mathbf{s}_{c,i}$ | Geographic centroid coordinates $(\operatorname{lon}_{c,i}, \operatorname{lat}_{c,i})$ of tract $i$ | Static target-city input |
| $\mathbf{x}_{c,i}$ | 26-dimensional urban context feature vector for tract $i$ (13 census, 8 POI, 5 road) | Static target-city input |
| $d_{c,ij}$ | Spherical Haversine distance between tract centroids of $i$ and $j$ (km) | Computed from geography |
| $t_{c,ij}$ | Observed reference trip flow volume from tract $i$ to tract $j$ ($t_{c,ij} \ge 1$) | Reference only |
| $\Omega_c^+$ | Set of observed positive OD pairs, $\{(i,j) : t_{c,ij} \ge 1\}$ | Known-support assumption |
| $\Omega_{c,\mathrm{inter}}^+$ | Positive interzonal support, $\{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij} \ge 1, i \ne j, d_{c,ij} > 0\}$ | Known-support assumption |
| $K$ | Number of distance intervals ($K = 8$ canonical) | Experimental setting |
| $I_b$ | Distance interval $b$, $[a_{b-1}, a_b)$ with pair-weighted quantiles $a_b$ from training cities | Experimental setting |
| $\mathcal{B}_{c,b}$ | Set of positive interzonal pairs in city $c$ within distance interval $b$ | Computed from geography |
| $F_{c,b}$ | Target reference trip volume in distance interval $b$, $\sum_{(i,j) \in \mathcal{B}_{c,b}} t_{c,ij}$ | Reference only |
| $\mathbf{Y}_{D,c}$ | Target distance-binned mobility distribution $[Y_{D,c,1}, \dots, Y_{D,c,K}]^T \in \Delta^{K-1}$ | Oracle calibration input |
| $\mathcal{A}_c$ | Set of active distance intervals with non-zero candidate pairs, $\{b : \widehat{Y}_{D,c,b}^{(0)} > 0\}$ | Model output |
| $p_{c,b}^{\mathrm{cond}}$ | Target distance distribution re-normalized over active intervals $\mathcal{A}_c$ | Oracle calibration input |
| $\theta$ | Trainable parameters of GNN backbone and ZTNB intensity head (frozen at target evaluation) | Source-city learned |
| $\widehat{T}_{c,ij}^{(0)}$ | Zero-shot baseline predicted flow intensity for pair $(i,j)$ (condition $M_0$) | Model output |
| $\widehat{F}_{c,b}^{(0)}$ | Baseline predicted flow mass falling into distance interval $b$ | Model output |
| $\widehat{Y}_{D,c,b}^{(0)}$ | Baseline implied distance distribution proportion in interval $b$ | Model output |
| $q$ | Soft calibration response parameter ($q \in [0, 1]$, canonical $q = 1.0$) | Experimental setting |
| $w_{c,b}(q)$ | Raw calibration ratio $(p_{c,b}^{\mathrm{cond}} / \widehat{Y}_{D,c,b}^{(0)})^q$ for active interval $b$ | Model output |
| $s_{c,b}(q)$ | Mass-preserving scaling multiplier $w_{c,b}(q) / \sum_{r \in \mathcal{A}_c} \widehat{Y}_{D,c,r}^{(0)} w_{c,r}(q)$ | Model output |
| $\widehat{T}_{c,ij}^{(1)}$ | Calibrated predicted flow intensity, $s_{c,b(i,j)}(q) \cdot \widehat{T}_{c,ij}^{(0)}$ (condition $M_1$) | Model output |
| $\widehat{\mathbf{T}}_c^{\mathrm{county}}$ | Calibrated predicted flows using origin-county grouped observations (condition $M1_{\mathrm{county}}$) | Model output |
| $\operatorname{CPC}_c$ | Common Part of Commuters on $\Omega_{c,\mathrm{inter}}^+$ between reference $t_{c,ij}$ and predicted flows | Metric |
| $\Delta\operatorname{CPC}_c$ | Paired performance gain $\operatorname{CPC}_c(\widehat{\mathbf{T}}_c^{(1)}) - \operatorname{CPC}_c(\widehat{\mathbf{T}}_c^{(0)})$ on city $c$ | Metric |
| $\overline{\Delta\operatorname{CPC}}$ | Benchmark macro-average improvement across all 50 metropolitan areas | Metric |

*Note: Table columns summarize the mathematical symbol, formal description, and source or functional role during inference on target city $c$. In the experimental conditions, $M_0$ denotes the frozen zero-shot baseline, $M_1$ (or $M1_{\mathrm{city}}$) denotes the primary city-level calibrated model, and $M1_{\mathrm{county}}$ denotes the spatial resolution variant.*

To explicitly clarify information availability across stages of the pipeline:

| Information Element | Baseline ($M_0$) Input? | Calibration ($M_1$) Input? | Reference / Evaluation Only? |
| :--- | :---: | :---: | :---: |
| Urban context features ($\mathbf{x}_{c,i}$) | Yes | Via baseline predictions $\widehat{T}_{c,ij}^{(0)}$ | No |
| Centroid Haversine distance ($d_{c,ij}$) | Yes | Yes (bin assignment) | No |
| Known positive support ($\Omega_{c,\mathrm{inter}}^+$) | Yes (estimand scope) | Yes (calibration domain) | Yes (evaluation mask) |
| Oracle aggregate distribution ($\mathbf{Y}_{D,c}$) | No | Yes | No |
| Individual reference flows ($t_{c,ij}$) | No | No | Yes |

*(Tiếng Việt: Gọi $c$ là một vùng đô thị (thành phố) và $\mathcal{V}_c$ là tập rời rạc các đơn vị không gian (census tract) phân chia thành phố đó. Mỗi cặp có thứ tự $(i,j)$ với $i,j \in \mathcal{V}_c$ biểu diễn một cặp nguồn–đích (OD), trong đó cùng một tập đơn vị không gian được sử dụng làm cả origin và destination. Bảng 1 tổng hợp các ký hiệu toán học cốt lõi, nguồn dữ liệu gốc, cũng như vai trò và trạng thái sẵn có của từng đại lượng trong các giai đoạn huấn luyện mô hình, suy luận zero-shot, hiệu chỉnh và đánh giá. Các ký hiệu chuyên biệt chỉ sử dụng trong các phân tích chẩn đoán hoặc stress test phụ (như nhiễu nhân tạo, donor placebo, hoặc kiến trúc backbone thay thế) được định nghĩa cục bộ tại các tiểu mục thực nghiệm tương ứng.)*

---

## 3.2 Data Sources and Spatial Representation
*(Tiếng Việt: **3.2. Nguồn dữ liệu và biểu diễn không gian**)*

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

## 3.3 Spatial Units and Observational Resolution: Primary City-Level Benchmark (`M1_city`)
*(Tiếng Việt: **3.3. Đơn vị không gian và độ phân giải của quan sát: Cấu hình chuẩn cấp thành phố (`M1_city`)**)*

The dataset provided by the laboratory is organized on a per-city basis. Each city $c$ comprises a discrete set of census tracts $\mathcal{V}_c$ and the observed positive interzonal OD pairs between them. **Tract** is the elementary spatial node unit of the neural network, while **city** is the unit for cross-validation data partitioning, zero-shot transfer learning, and performance evaluation.

For each target city $c$, the model predicts flow intensities across the entire observed positive support:

$$\Omega_c^+ = \left\{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij} \ge 1\right\}$$

The primary benchmark employs a single distance-binned mobility distribution defined at the **city level**. The reference flow volume of city $c$ within distance bin $b$ is:

$$F_{c,b} = \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{c,ij} \mathbb{I}(a_{b-1} \le d_{c,ij} < a_b)$$

The normalized city-level distance distribution is:

$$Y_{D,c,b} = \frac{F_{c,b}}{\sum_{r=1}^K F_{c,r}}, \qquad \sum_{b=1}^K Y_{D,c,b} = 1$$

The resulting vector $\mathbf{Y}_{D,c} = [Y_{D,c,1}, \dots, Y_{D,c,K}]^T$ is used to calibrate the entire OD flow prediction of target city $c$. This constitutes the primary configuration of the study (`M1_city`).

*(Tiếng Việt: Bộ dữ liệu do Lab cung cấp được tổ chức theo từng thành phố. Mỗi thành phố $c$ bao gồm một tập các tract $\mathcal{V}_c$ và các cặp OD dương giữa những tract đó. Tract là đơn vị không gian cơ sở của mô hình, trong khi city là đơn vị chia dữ liệu, thực hiện zero-shot transfer và đánh giá kết quả. Đối với mỗi thành phố mục tiêu, mô hình dự báo cường độ cho toàn bộ tập cặp OD được quan sát $\Omega_c^+ = \{(i,j):t_{c,ij}\geq1\}$. Các thử nghiệm chính sử dụng một phân phối di chuyển theo khoảng cách duy nhất ở cấp city. Tổng luồng tham chiếu của city $c$ trong khoảng cách $b$ là $F_{c,b} = \sum_{(i,j)\in\Omega_{c,\mathrm{inter}}^+} t_{c,ij} \mathbb{I}(a_{b-1}\leq d_{c,ij}<a_b)$. Phân phối khoảng cách ở cấp city là $Y_{D,c,b} = F_{c,b} / \sum_{r=1}^{K}F_{c,r}$ với $\sum_{b=1}^{K}Y_{D,c,b}=1$. Vector $\mathbf{Y}_{D,c}$ được sử dụng để hiệu chỉnh toàn bộ dự báo OD của thành phố mục tiêu. Đây là cấu hình chính của nghiên cứu (`M1_city`).)*

---

## 3.4 Fine-Grained Spatial Resolution Variant: County-Level Observations (`M1_county`)
*(Tiếng Việt: **3.4. Biến thể quan sát chi tiết ở cấp county (`M1_county`)**)*

A supplementary experiment examines whether providing aggregate distance observations at a finer sub-metropolitan spatial resolution provides incremental predictive information. In this analysis, the tracts of each city are grouped by county.

County boundaries are obtained from the Database of Global Administrative Areas, version 4.1 [@gadm41]. Each tract is mapped to its encompassing county via a spatial point-in-polygon join between the tract centroid and the county polygon. If a centroid does not receive a valid `within` match—for example, because it lies on a polygon boundary or near a coastline—the implementation falls back to a nearest-polygon join in EPSG:5070 and accepts the assignment only when the centroid-to-polygon distance is at most 5 km; otherwise, execution stops with an error. Duplicate matches are resolved deterministically so that each tract receives exactly one county label. GADM is strictly utilized for this spatial grouping step; GADM is not the source of tract centroid coordinates, urban features, or OD flows.

Letting $g(i)$ denote the county assigned to tract $i$, OD pairs are grouped strictly by the **origin tract's county**:

$$\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_{c,\mathrm{inter}}^+ : g(i) = \ell\right\}$$

Destination tract $j$ may belong to the same county or a different county within the metropolitan area. The distance-binned flow mass of county group $\ell$ is:

$$F_{c,\ell,b} = \sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij} \mathbb{I}(a_{b-1} \le d_{c,ij} < a_b)$$

and its normalized distance distribution vector is:

$$Y_{D,c,\ell,b} = \frac{F_{c,\ell,b}}{\sum_{r=1}^K F_{c,\ell,r}}, \qquad \sum_{b=1}^K Y_{D,c,\ell,b} = 1$$

Because the input data are strictly bounded within the tracts of the city dataset provided by the laboratory, $\mathbf{Y}_{D,c,\ell}$ describes the outflow distance distribution of trips originating from the tracts of city $c$ assigned to county $\ell$. It does not represent total county-wide mobility outside the study city's spatial footprint.

Each distribution $\mathbf{Y}_{D,c,\ell}$ is used to calibrate OD pairs whose origin tract belongs to county $\ell$. The calibrated predictions from all county groups are then assembled into a complete OD prediction for the city:

$$\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \widehat{T}_{c,ij}^{\mathrm{county}} : (i,j) \in \Omega_{c,\ell}^+ \right\}$$

where $\mathcal{G}_c$ denotes the set of counties present in the dataset for city $c$.

Crucially, increasing observational resolution from city to county does not alter the evaluation scope. The model still reconstructs and is evaluated against the complete set of positive flows $\Omega_{c,\mathrm{inter}}^+$ for the target city; only the aggregate supervisory signal supplied during calibration becomes spatially more granular (`M1_county`).

*(Tiếng Việt: Một thí nghiệm bổ sung kiểm tra liệu quan sát có độ phân giải không gian chi tiết hơn city có mang lại thêm thông tin hay không. Trong thí nghiệm này, các tract của mỗi city được phân nhóm theo county. Ranh giới county được lấy từ GADM phiên bản 4.1 [@gadm41]. Mỗi tract được gán vào county tương ứng dựa trên vị trí tọa độ tâm trong polygon county. Nếu phép ghép `within` không cho kết quả hợp lệ—chẳng hạn khi tâm tract nằm trên biên polygon hoặc gần đường bờ—mã nguồn chuyển sang polygon gần nhất trong EPSG:5070 và chỉ chấp nhận kết quả khi khoảng cách không quá 5 km; nếu không, chương trình dừng và báo lỗi. Các kết quả trùng được xử lý xác định để mỗi tract chỉ có một nhãn county. GADM chỉ được sử dụng cho bước phân nhóm này; GADM không phải nguồn của tọa độ tract, đặc trưng đô thị hoặc luồng OD. Gọi $g(i)$ là county được gán cho tract $i$. Theo quy tắc được xác nhận từ mã nguồn, các cặp OD được phân nhóm theo county của origin: $\Omega_{c,\ell}^+ = \{(i,j)\in\Omega_{c,\mathrm{inter}}^+:g(i)=\ell\}$. Destination $j$ có thể thuộc cùng county hoặc một county khác. Phân phối khoảng cách của nhóm county $\ell$ được xác định bởi $F_{c,\ell,b} = \sum_{(i,j)\in\Omega_{c,\ell}^+} t_{c,ij} \mathbb{I}(a_{b-1}\leq d_{c,ij}<a_b)$ và $Y_{D,c,\ell,b} = F_{c,\ell,b} / \sum_{r=1}^{K}F_{c,\ell,r}$. Do dữ liệu đầu vào vẫn được giới hạn trong các tract thuộc city do Lab cung cấp, $\mathbf{Y}_{D,c,\ell}$ mô tả phân phối khoảng cách của các chuyến đi xuất phát từ những tract của city được gán vào county $\ell$. Đại lượng này không nhất thiết đại diện cho toàn bộ hoạt động di chuyển của county bên ngoài phạm vi dữ liệu thành phố. Mỗi phân phối $\mathbf{Y}_{D,c,\ell}$ được dùng để hiệu chỉnh các cặp có origin thuộc county $\ell$. Sau đó, dự báo của tất cả nhóm county được ghép lại thành một dự báo OD hoàn chỉnh cho city: $\widehat{\mathbf{T}}_{c}^{\mathrm{county}} = \bigcup_{\ell\in\mathcal{G}_c} \{\widehat{T}_{c,ij}^{\mathrm{county}} : (i,j)\in\Omega_{c,\ell}^+\}$, trong đó $\mathcal{G}_c$ là tập county xuất hiện trong dữ liệu của city $c$. Như vậy, việc tăng độ phân giải quan sát từ city lên county không làm thay đổi phạm vi dự báo. Mô hình vẫn tái tạo và đánh giá toàn bộ OD của thành phố trên $\Omega_{c,\mathrm{inter}}^+$; chỉ thông tin tổng hợp được cung cấp cho bước hiệu chỉnh trở nên chi tiết hơn về mặt không gian (`M1_county`).)*

---

## 3.5 Model Structure and Inference-Time Calibration
*(Tiếng Việt: **3.5. Cấu trúc mô hình và hiệu chỉnh tại thời điểm suy luận**)*

### 3.5.1 Common Baseline Prediction Interface
*(Tiếng Việt: **3.5.1. Giao diện dự báo baseline chung**)*

All three candidate predictor families—the primary Gravity-Informed Urban GNN ($m = \text{GNN}$), the ablated Pairwise Node MLP ($m = \text{MLP}$), and the classical Two-Parameter Gravity model ($m = \text{Grav}$)—generate an initial zero-shot flow intensity prediction across the identical known positive interzonal support $\Omega_{c,\mathrm{inter}}^+$. This shared operational interface is formalized as:

$$\widehat{T}_{c,ij}^{(0,m)} = f_{\widehat{\theta}_m}^{(m)}(\text{target-city inputs}), \qquad (i,j) \in \Omega_{c,\mathrm{inter}}^+$$

where superscript $(0)$ designates uncalibrated baseline predictions and $m \in \{\text{GNN}, \text{MLP}, \text{Grav}\}$ indexes the model family. 

Each model is trained or fitted strictly on the source training cities $\mathcal{C}_{\mathrm{train}}^{(f)}$ of the active cross-validation fold, and all parameters $\widehat{\theta}_m$ are held strictly frozen prior to target-city inference. The target city's distance-binned mobility distribution $\mathbf{Y}_{D,c}$ is never supplied during baseline prediction generation. Downstream, the identical analytical calibration operator $\operatorname{Calibrate}(\cdot, \mathbf{Y}_{D,c})$ is applied to the output of all three models. Here, the Urban GNN serves as the primary predictive architecture, while the MLP and classical gravity models provide structured counterfactual baselines to verify whether calibration benefits depend on graph message passing or neural representations.

*(Tiếng Việt: Cả ba họ mô hình dự báo—mô hình neural chính Urban GNN kết hợp tiên nghiệm Gravity ($m = \text{GNN}$), mô hình bóc tách Pairwise Node MLP ($m = \text{MLP}$), và mô hình tham số cổ điển Gravity hai tham số ($m = \text{Grav}$)—đều tạo ra dự báo cường độ luồng zero-shot ban đầu trên cùng một tập hỗ trợ liên vùng dương đã biết $\Omega_{c,\mathrm{inter}}^+$. Giao diện vận hành chung này được hình thức hóa như sau:
$$
\widehat{T}_{c,ij}^{(0,m)} = f_{\widehat{\theta}_m}^{(m)}(\text{dữ liệu đầu vào thành phố mục tiêu}), \qquad (i,j) \in \Omega_{c,\mathrm{inter}}^+
$$
trong đó số mũ $(0)$ biểu thị dự báo baseline trước hiệu chỉnh và $m \in \{\text{GNN}, \text{MLP}, \text{Grav}\}$ chỉ định họ mô hình. Mỗi mô hình được huấn luyện hoặc khớp tham số hoàn toàn trên các thành phố huấn luyện nguồn $\mathcal{C}_{\mathrm{train}}^{(f)}$ của fold kiểm định chéo tương ứng, và toàn bộ tham số $\widehat{\theta}_m$ được đóng băng nghiêm ngặt trước khi suy luận trên thành phố mục tiêu. Phân phối di chuyển theo khoảng cách $\mathbf{Y}_{D,c}$ của thành phố mục tiêu tuyệt đối không được sử dụng ở bước tạo dự báo baseline. Sau đó, cùng một toán tử hiệu chỉnh giải tích $\operatorname{Calibrate}(\cdot, \mathbf{Y}_{D,c})$ được áp dụng thống nhất cho đầu ra của cả ba mô hình. Trong đó, Urban GNN là kiến trúc dự báo chính, còn MLP và mô hình gravity đóng vai trò như các đối chứng có cấu trúc nhằm kiểm tra xem lợi ích hiệu chỉnh có phụ thuộc vào cơ chế message passing trên đồ thị hay bản chất kiến trúc neural hay không.)*

---

### 3.5.2 Primary Neural Predictor: Gravity-Informed Urban GNN
*(Tiếng Việt: **3.5.2. Mô hình neural chính: Urban GNN kết hợp tiên nghiệm Gravity**)*

The primary predictive model is a support-conditioned zero-shot architecture combining spatial graph convolutions with a physics-inspired gravity prior and a Zero-Truncated Negative Binomial (ZTNB) intensity head.

#### Spatial Graph Construction and Node Features
For each target city $c$, the discrete set of spatial units $\mathcal{V}_c$ comprises $N_c = |\mathcal{V}_c|$ census tracts. Each tract $i \in \mathcal{V}_c$ is georeferenced by its centroid coordinates $\mathbf{s}_{c,i} = (\operatorname{lon}_{c,i}, \operatorname{lat}_{c,i})$. Pairwise geographic distances $d_{c,ij}$ are computed via the spherical Haversine formula with Earth radius $R = 6371\text{ km}$. 

An undirected spatial radius graph $\mathcal{G}_c = (\mathcal{V}_c, \mathcal{E}_c)$ is constructed exclusively from geographic centroids by connecting any pair of tracts whose centroid distance satisfies $d_{c,ij} \le r$ with a fixed radius threshold of $r = 5.0\text{ km}$. The graph includes self-loops $(i,i) \in \mathcal{E}_c$ with $d_{c,ii} = 0$. To prevent disconnected nodes, any tract possessing zero neighbors within the 5 km radius is connected to its single geographically nearest tract. Crucially, $\mathcal{G}_c$ is constructed purely from observable spatial geography; no trip or OD flow data are ever utilized in graph generation.

Each tract is described by a 26-dimensional urban context vector $\mathbf{x}_{c,i} \in \mathbb{R}^{26}$ (13 Census demographic attributes, 8 POI amenity counts, and 5 road network density metrics). Within each fold $f$, all node features are standardized using `StandardScaler` statistics fitted exclusively on the 35 training cities $\mathcal{C}_{\mathrm{train}}^{(f)}$ and applied frozen to target cities.

#### GNN Encoder Architecture
The node encoder (`UrbanGNN`) maps normalized features $\mathbf{x}_{c,i}$ and graph topology $\mathcal{G}_c$ into $d$-dimensional node embeddings $\mathbf{h}_{c,i} \in \mathbb{R}^{64}$. First, raw node features are projected through a linear layer with LayerNorm, ReLU activation, and dropout ($p = 0.1$):

$$\mathbf{h}_{c,i}^{(0)} = \operatorname{Dropout}\left(\operatorname{ReLU}\left(\operatorname{LayerNorm}\left(\mathbf{W}_{\mathrm{in}} \mathbf{x}_{c,i} + \mathbf{b}_{\mathrm{in}}\right)\right)\right), \quad \mathbf{W}_{\mathrm{in}} \in \mathbb{R}^{64 \times 26}$$

The encoder then stacks $L = 2$ distance-modulated message-passing layers (`GraphConvLayer`). For layer $l \in \{1, \dots, L\}$, incoming messages from neighboring nodes $j \in \mathcal{N}(i)$ are conditioned on edge Haversine distance via logarithmic transformation $\log(1 + d_{c,ji})$:

$$\mathbf{m}_{ji} = \mathbf{W}_{\mathrm{msg}} \left[ \mathbf{h}_{c,j}^{(l-1)} \,\Vert\, \log(1 + d_{c,ji}) \right] + \mathbf{b}_{\mathrm{msg}}, \quad \mathbf{W}_{\mathrm{msg}} \in \mathbb{R}^{64 \times (64 + 1)}$$

Messages are aggregated using degree-normalized mean aggregation:

$$\mathbf{a}_{c,i}^{(l)} = \frac{1}{\max(\operatorname{deg}(i), 1)} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ji}$$

The aggregated context is combined with a linear self-transformation, activated through ReLU, normalized via LayerNorm, and passed through a residual skip connection with dropout:

$$\widetilde{\mathbf{h}}_{c,i}^{(l)} = \operatorname{LayerNorm}\left(\operatorname{ReLU}\left(\mathbf{a}_{c,i}^{(l)} + \mathbf{W}_{\mathrm{self}} \mathbf{h}_{c,i}^{(l-1)} + \mathbf{b}_{\mathrm{self}}\right)\right)$$

$$\mathbf{h}_{c,i}^{(l)} = \mathbf{h}_{c,i}^{(l-1)} + \operatorname{Dropout}\left(\widetilde{\mathbf{h}}_{c,i}^{(l)}\right)$$

A final linear projection produces the tract representations: $\mathbf{h}_{c,i} = \mathbf{W}_{\mathrm{out}} \mathbf{h}_{c,i}^{(L)} + \mathbf{b}_{\mathrm{out}} \in \mathbb{R}^{64}$.

#### Pairwise OD Representation and Residual Gravity Decoding
To predict flow intensity between origin tract $i$ and destination tract $j$, candidate pairs are indexed via integer arrays `pair_o_idx` and `pair_d_idx`. These arrays serve strictly as index retrieval pointers to gather embeddings $\mathbf{h}_{c,i}$ and $\mathbf{h}_{c,j}$; they contain no learned node identity embeddings.

A classical two-parameter physics gravity prior is computed for each pair using tract populations $P_{c,i} = \max(\operatorname{pop}_{c,i}, 1.0)$ and $P_{c,j} = \max(\operatorname{pop}_{c,j}, 1.0)$, and inter-tract Haversine distance clamped at a floor of 0.1 km:

$$\log T_{c,ij}^{\mathrm{grav}} = G + \log P_{c,i} + \log P_{c,j} - \alpha \log(\max(d_{c,ij}, 0.1))$$

where global scale $G \in \mathbb{R}$ and distance decay $\alpha = \exp(\log \alpha) > 0$ are trainable parameters initialized at $G = 0.0$ and $\alpha = 1.0$.

The pairwise OD edge representation $\mathbf{e}_{c,ij} \in \mathbb{R}^{130}$ concatenates origin and destination embeddings, the log-transformed inter-tract distance, and the log gravity prior:

$$\mathbf{e}_{c,ij} = \left[ \mathbf{h}_{c,i} \,\Vert\, \mathbf{h}_{c,j} \,\Vert\, \log(1 + d_{c,ij}) \,\Vert\, \log T_{c,ij}^{\mathrm{grav}} \right]$$

The pairwise decoder (`PairwiseODDecoder`) consists of a multi-layer perceptron with layers $\operatorname{Linear}(130, 64) \to \operatorname{LayerNorm} \to \operatorname{ReLU} \to \operatorname{Dropout}(0.1) \to \operatorname{Linear}(64, 32) \to \operatorname{ReLU} \to \operatorname{Dropout}(0.1) \to \operatorname{Linear}(32, 1)$. The weights and bias of the final linear projection are explicitly initialized to zero. Consequently, the network predicts a residual adjustment to the log gravity prior:

$$\operatorname{residual}_{c,ij} = \operatorname{MLP}_{\mathrm{dec}}(\mathbf{e}_{c,ij})$$

$$\log \mu_{c,ij} = \log T_{c,ij}^{\mathrm{grav}} + \operatorname{residual}_{c,ij}$$

$$\mu_{c,ij} = \operatorname{softplus}(\log \mu_{c,ij}) + 10^{-4}$$

Because $\operatorname{residual}_{c,ij} \approx 0$ at initialization, the base parameter initially tracks $\mu_{c,ij} \approx \operatorname{softplus}(\log T_{c,ij}^{\mathrm{grav}})$, anchoring neural optimization to physical spatial interaction.

#### Conditional Mean Conversion for Flow Intensity
Because the training and evaluation support consists exclusively of observed positive OD links ($t_{c,ij} \ge 1$), flow volume is modeled using the Zero-Truncated Negative Binomial distribution [@grogger1991truncated; @hilbe2011negative]. The base count distribution is parameterized by base mean $\mu_{c,ij} > 0$ and global dispersion $\phi = \exp(\log \phi) > 0$. The zero-probability of the base Negative Binomial is:

$$p_{0,c,ij} = P_{\mathrm{NB}}(T_{c,ij} = 0; \mu_{c,ij}, \phi) = \left( \frac{\phi}{\mu_{c,ij} + \phi} \right)^\phi$$

At inference time on target cities, the model outputs the exact **conditional expectation**:

$$\widehat{T}_{c,ij}^{(0,\mathrm{GNN})} = \mathbb{E}[T_{c,ij} \mid T_{c,ij} \ge 1] = \frac{\mu_{c,ij}}{1 - p_{0,c,ij}}$$

Because $\mu_{c,ij} > 0$ and $p_{0,c,ij} \in (0, 1)$, $\widehat{T}_{c,ij}^{(0,\mathrm{GNN})}$ is a strictly positive real value ($\widehat{T}_{c,ij}^{(0,\mathrm{GNN})} > \mu_{c,ij} > 0$). The formal ZTNB training likelihood, gradient normalization, and optimization setup are established in Section 3.5.6.

*(Tiếng Việt: Do tập dữ liệu huấn luyện và đánh giá chỉ bao gồm các liên kết OD dương quan sát được ($t_{c,ij} \ge 1$), cường độ luồng được mô hình hóa bằng phân phối Negative Binomial cắt tại 0 (ZTNB). Phân phối đếm cơ sở được tham số hóa bởi mean $\mu_{c,ij} > 0$ và tham số phân tán toàn cục $\phi = \exp(\log \phi) > 0$. Xác suất tại 0 của phân phối Negative Binomial cơ sở là $p_{0,c,ij} = (\phi/(\mu_{c,ij}+\phi))^\phi$. Tại thời điểm suy luận, dự báo baseline zero-shot được tính chính xác bằng kỳ vọng có điều kiện $\widehat{T}_{c,ij}^{(0,\mathrm{GNN})} = \mathbb{E}[T_{c,ij}\mid T_{c,ij}\ge 1] = \mu_{c,ij} / (1 - p_{0,c,ij}) > 0$. Quy trình huấn luyện likelihood ZTNB và tối ưu hóa chi tiết được trình bày tại Mục 3.5.6.)*

---

### 3.5.3 Alternative Neural Predictor: Pairwise Node MLP
*(Tiếng Việt: **3.5.3. Mô hình neural đối chứng: Pairwise Node MLP**)*

To test whether the incremental information gain from distance distribution calibration is contingent on spatial graph convolutions, we evaluate an ablated neural architecture: the Pairwise Node MLP (`NodeMLP`). 

The Pairwise Node MLP operates on the exact same 26 normalized urban features $\mathbf{x}_{c,i}$ as the Urban GNN, but completely eliminates spatial graph convolutions and message passing. To maintain architectural and parameter-count parity with `UrbanGNN`, each layer of `NodeMLP` replicates the linear transformations of `GraphConvLayer` using a dense linear mapping with dummy zero-padded distance features, LayerNorm, ReLU, and residual dropout connections:

$$\mathbf{h}_{c,i}^{\mathrm{MLP}} = \operatorname{NodeMLP}_{\theta_M}(\mathbf{x}_{c,i}) \in \mathbb{R}^{64}$$

Tract embeddings are computed strictly from local tract features without aggregating information from geographic neighbors. 

The pairwise edge representation $\mathbf{e}_{c,ij}^{\mathrm{MLP}}$ is formed identically by vector concatenation:

$$\mathbf{e}_{c,ij}^{\mathrm{MLP}} = \left[ \mathbf{h}_{c,i}^{\mathrm{MLP}} \,\Vert\, \mathbf{h}_{c,j}^{\mathrm{MLP}} \,\Vert\, \log(1 + d_{c,ij}) \,\Vert\, \log T_{c,ij}^{\mathrm{grav}} \right]$$

The MLP uses the exact same `PairwiseODDecoder`, trainable `GravityPrior`, and ZTNB loss function $\mathcal{L}_{\mathrm{ZTNB}}$ with global dispersion $\phi$. Zero-shot baseline predictions are computed via the conditional expectation:

$$\widehat{T}_{c,ij}^{(0,\mathrm{MLP})} = \mathbb{E}[T_{c,ij} \mid T_{c,ij} \ge 1] = \frac{\mu_{c,ij}^{\mathrm{MLP}}}{1 - p_{0,c,ij}^{\mathrm{MLP}}}$$

This model isolates the contribution of local node features and pairwise gravity priors in the absence of spatial relational message passing.

*(Tiếng Việt: Nhằm kiểm tra xem giá trị bổ sung của phân phối khoảng cách có phụ thuộc vào cơ chế tích chập đồ thị hay không, chúng tôi xây dựng mô hình bóc tách Pairwise Node MLP (`NodeMLP`). Mô hình sử dụng cùng 26 đặc trưng đô thị đã chuẩn hóa $\mathbf{x}_{c,i}$, duy trì cấu hình 2 lớp ẩn chiều 64 và số lượng tham số tương đương với Urban GNN, nhưng loại bỏ hoàn toàn các liên kết đồ thị và quá trình message passing qua láng giềng. Embedding của mỗi tract được tính độc lập từ chính đặc trưng nội tại của tract đó. Biểu diễn cặp OD tiếp tục được đưa qua cùng một cấu trúc `PairwiseODDecoder`, cùng tiên nghiệm `GravityPrior` và cùng hàm mất mát ZTNB NLL. Dự báo baseline zero-shot của mô hình là $\widehat{T}_{c,ij}^{(0,\mathrm{MLP})} = \mu_{c,ij}^{\mathrm{MLP}} / (1 - p_{0,c,ij}^{\mathrm{MLP}})$.)*

---

### 3.5.4 Explicit Low-Complexity Baseline: Two-Parameter Power-Law Gravity
*(Tiếng Việt: **3.5.4. Mô hình đối chứng tham số: Gravity hai tham số**)*

To establish whether the calibration operator delivers benefits outside of deep neural architectures, we incorporate a classical two-parameter power-law gravity model as an explicit, low-complexity parametric benchmark:

$$T_{c,ij}^{\mathrm{grav}} = \exp(G) \cdot \frac{P_{c,i} \cdot P_{c,j}}{d_{c,ij}^\alpha}$$

where $P_{c,i} = \max(\operatorname{pop}_{c,i}, 1.0)$ and $P_{c,j} = \max(\operatorname{pop}_{c,j}, 1.0)$ are tract population totals, and $d_{c,ij}$ is Haversine centroid distance clamped at a 0.1 km minimum. 

The model contains exactly two global parameters:
1. $G \in \mathbb{R}$: the global log-scale intercept factor;
2. $\alpha > 0$: the power-law distance decay exponent.

The parameters are estimated via log-linear Ordinary Least Squares (OLS) regression over the pooled positive interzonal pairs of the training cities in fold $f$:

$$\log(t_{c,ij}) - \log(P_{c,i} P_{c,j}) = G - \alpha \log(d_{c,ij})$$

Crucially, $G$ and $\alpha$ are fitted once per fold strictly on $\mathcal{C}_{\mathrm{train}}^{(f)}$ and held fixed when evaluating target cities. The model receives no target-city OD flows, no origin production balancing factors ($A_i$), no destination attraction balancing factors ($B_j$), and no observed marginal totals ($O_i, D_j$). The zero-shot baseline prediction on target city $c$ is:

$$\widehat{T}_{c,ij}^{(0,\mathrm{Grav})} = \exp(\widehat{G}^{(f)}) \cdot \frac{P_{c,i} \cdot P_{c,j}}{d_{c,ij}^{\widehat{\alpha}^{(f)}}}, \qquad (i,j) \in \Omega_{c,\mathrm{inter}}^+$$

This baseline provides a highly constrained, non-neural control whose distance-decay behavior is governed entirely by a single power-law exponent.

*(Tiếng Việt: Để xác định liệu toán tử hiệu chỉnh có mang lại lợi ích ngoài các mô hình học sâu hay không, chúng tôi đưa vào một mô hình gravity hàm lũy thừa hai tham số cổ điển làm đối chứng tường minh có độ phức tạp thấp: $T_{c,ij}^{\mathrm{grav}} = \exp(G) \cdot \frac{P_{c,i} P_{c,j}}{d_{c,ij}^\alpha}$. Hai tham số gồm hệ số quy mô toàn cục $G \in \mathbb{R}$ và số mũ phân rã khoảng cách $\alpha > 0$. Hai tham số này được ước lượng bằng hồi quy OLS log-tuyến tính trên toàn bộ các cặp liên vùng dương của các thành phố huấn luyện thuộc fold $f$. Mô hình hoàn toàn không được fit lại trên thành phố mục tiêu, không sử dụng hệ số cân bằng sinh/hút chuyến (balancing factors), và không sử dụng tổng phát sinh hay thu hút quan sát được ($O_i, D_j$), đảm bảo tính chất zero-shot tuyệt đối.)*

---

### 3.5.5 Comparative Summary of Baseline Predictors
*(Tiếng Việt: **3.5.5. Bảng so sánh các mô hình dự báo baseline**)*

Table 2 contrasts the input specifications, spatial mechanisms, output modeling assumptions, and scientific roles of the three baseline predictors.

#### Table 2: Architectural and operational comparison across baseline zero-shot predictors
*(Tiếng Việt: **Bảng 2: So sánh kiến trúc và đặc tính vận hành giữa các mô hình baseline**)*

| Predictor | Urban Context Features | Pairwise Distance | Graph Message Passing | Output Model / Objective | Role in Study |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Gravity-Informed Urban GNN** | 26 features ($\mathbf{x}_{c,i}$) | In message passing & pairwise decoder | Yes (5 km Haversine radius graph) | ZTNB conditional mean $\mathbb{E}[T \mid T \ge 1]$ | Primary proposed transfer architecture |
| **Pairwise Node MLP** | 26 features ($\mathbf{x}_{c,i}$) | In pairwise decoder | No (isolated local node features) | ZTNB conditional mean $\mathbb{E}[T \mid T \ge 1]$ | Neural ablation baseline (tests graph convolutions) |
| **Classical Two-Parameter Gravity** | None (Tract population only) | Power-law impedance $d_{c,ij}^{-\alpha}$ | No | Closed-form log-linear OLS | Explicit low-complexity parametric control |

*Note: All three models generate baseline zero-shot predictions on the identical positive interzonal support $\Omega_{c,\mathrm{inter}}^+$, are trained or fitted strictly on source cities, and remain completely frozen during target-city calibration. Performance metrics are reported in Section 4.*

---

### 3.5.6 Model Fitting under Partial OD Observations
*(Tiếng Việt: **3.5.6. Hàm mục tiêu dưới thiết lập quan sát partial OD**)*

#### Partial positive-flow observation setting
In empirical urban mobility modeling, the complete origin-destination flow matrix $\mathcal{V}_c \times \mathcal{V}_c$ is never assumed to be fully observable. Instead, empirical records capture only a subset of cell pairs exhibiting positive, verifiable travel movements. In our formulation, unobserved pairs are treated strictly as missing or unknown rather than zero-flow observations. The absence of an OD pair from the dataset is not treated as evidence of zero travel flow; unobserved pairs are therefore never incorporated into the loss function as structural zeros. Consequently, our predictive framework does not train binary classifiers to separate links from non-links, nor does it penalize models for unobserved pairs. The spatial link formation or observation process governing network sparsity is considered exogenous and falls outside the scope of our intensity models.

Formally, we distinguish three operational roles across the cross-validation partitions for fold $f$:
1. **Training cities ($c \in \mathcal{C}_{\mathrm{train}}^{(f)}$)**: Model parameters are learned exclusively by fitting observed positive interzonal travel intensities on the training support $\Omega_{c,\mathrm{inter}}^+ = \{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij} \ge 1, i \ne j, d_{c,ij} > 0\}$. Unobserved pairs and intrazonal pairs are excluded from parameter estimation.
2. **Validation cities ($c \in \mathcal{C}_{\mathrm{val}}^{(f)}$)**: Predictions are evaluated on their positive interzonal support strictly to guide learning rate schedules, trigger early stopping, and select the optimal model checkpoint $\theta^*$. No gradients are backpropagated from validation cities.
3. **Test target cities ($c \in \mathcal{C}_{\mathrm{test}}^{(f)}$)**: Baseline parameters remain completely frozen. Ground-truth flow intensities $t_{c,ij}$ are never accessed during baseline forward passes; they are accessed solely to construct the oracle aggregate distance observation $\mathbf{Y}_{D,c}$ for the controlled calibration experiment and to evaluate downstream accuracy. Under the evaluated estimand, the positive interzonal support $\Omega_{c,\mathrm{inter}}^+$ of the target city is assumed known.

*(Tiếng Việt: Trong các mạng lưới di chuyển đô thị thực tế, ma trận xuất phát–đích đến đầy đủ $\mathcal{V}_c \times \mathcal{V}_c$ không bao giờ được giả định là có thể quan sát toàn phần. Thay vào đó, dữ liệu ghi nhận thực nghiệm chỉ bao gồm một tập con các cặp ô có phát sinh lưu lượng dương và có thể xác minh được. Trong cách tiếp cận của chúng tôi, các cặp không xuất hiện trong dữ liệu được xem là chưa quan sát (missing hoặc unknown), chứ không phải là luồng bằng 0. Sự vắng mặt của một cặp OD trong dữ liệu không được xem là bằng chứng cho thấy cường độ thực của cặp đó bằng 0; do đó, các cặp chưa quan sát không được đưa vào loss như những mẫu zero. Do đó, khung mô hình không huấn luyện bộ phân loại nhị phân để phân biệt liên kết tồn tại hay không tồn tại, và không phạt mô hình trên các cặp không quan sát. Quá trình hình thành liên kết hoặc cơ chế quan sát tập hỗ trợ được xem là ngoại sinh và nằm ngoài phạm vi của mô hình cường độ. Ba vai trò quan sát được phân định rõ trong mỗi fold: (1) Thành phố huấn luyện ($c \in \mathcal{C}_{\mathrm{train}}^{(f)}$): tham số mô hình được học độc quyền từ cường độ luồng của các cặp liên vùng dương quan sát được trên $\Omega_{c,\mathrm{inter}}^+$; (2) Thành phố validation ($c \in \mathcal{C}_{\mathrm{val}}^{(f)}$): dùng để điều chỉnh learning rate, kích hoạt early stopping và chọn checkpoint tốt nhất; và (3) Thành phố kiểm tra ($c \in \mathcal{C}_{\mathrm{test}}^{(f)}$): tham số mô hình được đóng băng tuyệt đối, cường độ luồng chỉ dùng để tạo tín hiệu oracle $\mathbf{Y}_{D,c}$ tại thời điểm suy luận và tính toán độ chính xác đánh giá trên tập hỗ trợ dương $\Omega_{c,\mathrm{inter}}^+$ được giả định đã biết.)*

#### Neural ZTNB objective
Because training observations are restricted to strictly positive counts ($t_{c,ij} \ge 1$), both neural architectures—the Urban GNN ($m = \text{GNN}$) and the Pairwise Node MLP ($m = \text{MLP}$)—are optimized under the exact same Zero-Truncated Negative Binomial (ZTNB) likelihood. Let $\theta_m$ denote the trainable parameters of neural backbone $m$ (including node encoder, pairwise decoder, and gravity prior parameters), and let $\phi = \exp(\log \phi) > 0$ denote the global dispersion parameter, where $\log \phi \in \mathbb{R}$ is a shared learnable scalar. 

For any city $c \in \mathcal{C}_{\mathrm{train}}^{(f)}$, the network outputs unconstrained base Negative Binomial mean values $\mu_{c,ij}^{(m)} > 0$ for all $(i,j) \in \Omega_{c,\mathrm{inter}}^+$. The base Negative Binomial probability at integer count $t$ is:

$$\log p_{\mathrm{NB}}\left(t \mid \mu_{c,ij}^{(m)}, \phi\right) = \log \Gamma(t + \phi) - \log \Gamma(\phi) - \log \Gamma(t + 1) + \phi \log \left( \frac{\phi}{\mu_{c,ij}^{(m)} + \phi} \right) + t \log \left( \frac{\mu_{c,ij}^{(m)}}{\mu_{c,ij}^{(m)} + \phi} \right)$$

The probability of zero under the base distribution is $p_{0,c,ij} = P_{\mathrm{NB}}(0 \mid \mu_{c,ij}^{(m)}, \phi) = \left(\frac{\phi}{\mu_{c,ij}^{(m)} + \phi}\right)^\phi$. Conditioning strictly on non-zero observations ($t \ge 1$), the zero-truncated likelihood is:

$$p_{\mathrm{ZTNB}}\left(t_{c,ij} \mid \mu_{c,ij}^{(m)}, \phi\right) = \frac{p_{\mathrm{NB}}\left(t_{c,ij} \mid \mu_{c,ij}^{(m)}, \phi\right)}{1 - p_{0,c,ij}}$$

The denominator term $1 - p_{0,c,ij}$ acts as an exact normalization factor that re-scales probability mass onto the truncated support $\{1, 2, \dots\}$. To prevent cities with disproportionately large numbers of observed pairs from dominating parameter updates solely through pair count, the neural loss is computed as the mean negative log-likelihood per city:

$$\mathcal{L}_{\mathrm{neural}}^{(c)}\left(\theta_m, \phi\right) = -\frac{1}{|\Omega_{c,\mathrm{inter}}^+|} \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \left[ \log p_{\mathrm{NB}}\left(t_{c,ij} \mid \mu_{c,ij}^{(m)}, \phi\right) - \log\left(1 - p_{0,c,ij}\right) \right]$$

Gradients with respect to network weights $\theta_m$ and dispersion $\log \phi$ are computed via PyTorch automatic differentiation (`torch.autograd`). Optimization proceeds sequentially city-by-city across all $c \in \mathcal{C}_{\mathrm{train}}^{(f)}$ in each training epoch. Numerical stability is enforced throughout:
1. $\log \phi$ is clamped to $[-10.0, 10.0]$;
2. $\log(1 - p_0)$ is computed via the numerically stable identity $\log(1 - p_0) = \operatorname{log1p}(-\exp(\log p_0))$ clamped at $1.0 - 10^{-7}$;
3. Small epsilons ($\epsilon = 10^{-8}$) are added to $\mu$ and $\phi$ to avoid vanishing arguments;
4. Gradients are clipped to a maximum Euclidean norm of $5.0$ (`torch.nn.utils.clip_grad_norm_`).

*(Tiếng Việt: Do các quan sát huấn luyện được giới hạn chặt chẽ ở các số đếm dương ($t_{c,ij} \ge 1$), cả hai mô hình neural Urban GNN và Node MLP đều được tối ưu hóa theo cùng một hàm hợp lý ZTNB. Gọi $\theta_m$ là tập tham số của mô hình $m$ và $\phi = \exp(\log \phi) > 0$ là tham số phân tán toàn cục học được. Với mỗi cặp thuộc $\Omega_{c,\mathrm{inter}}^+$, mô hình dự báo giá trị trung bình cơ sở $\mu_{c,ij}^{(m)} > 0$. Xác suất ZTNB có điều kiện là $p_{\mathrm{ZTNB}}(t_{c,ij} \mid \mu_{c,ij}^{(m)}, \phi) = p_{\mathrm{NB}}(t_{c,ij} \mid \mu_{c,ij}^{(m)}, \phi) / (1 - p_{0,c,ij})$, trong đó mẫu số $1 - p_{0,c,ij} = 1 - (\phi/(\mu_{c,ij}^{(m)}+\phi))^\phi$ đóng vai trò là hệ số chuẩn hóa bắt buộc do likelihood được điều kiện hóa trên tập quan sát dương. Gradient được tính tự động thông qua PyTorch autograd. Nhằm tránh tình trạng các thành phố có số lượng cặp OD quá lớn áp đảo các thành phố nhỏ hơn, loss được tính bằng trung bình negative log-likelihood theo từng thành phố và tối ưu hóa tuần tự city-by-city trong mỗi epoch, kết hợp gradient clipping 5.0 và các cơ chế ổn định số học.)*

#### Two-parameter gravity objective
In contrast to the neural architectures, the standalone classical gravity baseline possesses a closed-form parametric structure governed by exactly two global parameters: global log-scale factor $G \in \mathbb{R}$ and power-law distance decay exponent $\alpha > 0$:

$$T_{c,ij}^{\mathrm{grav}} = \exp(G) \cdot \frac{P_{c,i} \cdot P_{c,j}}{d_{c,ij}^\alpha}$$

Under the partial OD observation setting, parameters $(G, \alpha)$ are estimated by minimizing the sum of squared log-intensity residuals across the pooled positive interzonal training pairs of all source cities in fold $f$:

$$\mathcal{L}_{\mathrm{grav}}(G, \alpha) = \sum_{c \in \mathcal{C}_{\mathrm{train}}^{(f)}} \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \left[ \left(\log t_{c,ij} - \log(P_{c,i} P_{c,j})\right) - \left(G - \alpha \log d_{c,ij}\right) \right]^2$$

This formulation corresponds to a log-linear Ordinary Least Squares (OLS) objective:

$$\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2, \qquad \boldsymbol{\beta} = [G, \alpha]^T$$

where the response vector $\mathbf{y}$ contains elements $y_{c,ij} = \log t_{c,ij} - (\log P_{c,i} + \log P_{c,j})$, and the design matrix $\mathbf{X}$ has rows $[1, -\log d_{c,ij}]$ for all $(i,j) \in \Omega_{c,\mathrm{inter}}^+$ across all $c \in \mathcal{C}_{\mathrm{train}}^{(f)}$. The exact analytical solution is obtained directly via linear least squares:

$$\widehat{\boldsymbol{\beta}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

This parametric objective uses no Poisson likelihood, no iterative gradient descent, no balancing factors ($A_i, B_j$), no origin or destination trip marginals ($O_i, D_j$), and no unobserved pairs. It is fitted once per fold strictly on $\mathcal{C}_{\mathrm{train}}^{(f)}$ and held frozen during evaluation.

*(Tiếng Việt: Ngược lại với các mạng neural, mô hình gravity hai tham số cổ điển được ước lượng thông qua nghiệm đóng bằng hồi quy bình phương tối thiểu log-tuyến tính (OLS) trên các cặp liên vùng dương gộp chung của toàn bộ các thành phố huấn luyện thuộc fold $f$: $\mathcal{L}_{\mathrm{grav}}(G, \alpha) = \sum_{c\in\mathcal{C}_{\mathrm{train}}^{(f)}} \sum_{(i,j)\in\Omega_{c,\mathrm{inter}}^+} [(\log t_{c,ij} - \log(P_{c,i} P_{c,j})) - (G - \alpha \log d_{c,ij})]^2$. Mục tiêu này được giải chính xác qua công thức OLS đại số tuyến tính $\widehat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ với $\boldsymbol{\beta} = [G, \alpha]^T$. Mô hình không sử dụng phân phối Poisson, không sử dụng hệ số cân bằng $A_i, B_j$, không sử dụng tổng phát sinh hay thu hút $O_i, D_j$, và không đưa các cặp chưa quan sát vào hàm mục tiêu.)*

---

### 3.5.7 Training, Model Selection, and Freezing
*(Tiếng Việt: **3.5.7. Huấn luyện, lựa chọn mô hình và đóng băng tham số**)*

#### Neural optimizer and hyperparameter configuration
Both neural predictors (`UrbanGNN` and `NodeMLP`) share an identical training procedure. Parameter updates are performed using the AdamW optimizer [@loshchilov2019adamw] (`torch.optim.AdamW`) with an initial learning rate $\eta = 2 \times 10^{-3}$ and decoupled weight decay coefficient $\lambda_{\mathrm{wd}} = 10^{-4}$. Regularization is enforced strictly through the optimizer's decoupled weight decay mechanism; no explicit $\ell_2$ penalty term $\lambda \|\theta\|_2^2$ is added to the loss function. The weight decay value is fixed a priori without grid search. Dropout ($p = 0.1$) and LayerNorm modules embedded within each network layer provide internal architectural regularization during training passes. Optimization operates with full-city batching (passing all $|\Omega_{c,\mathrm{inter}}^+|$ positive interzonal pairs of a city simultaneously per forward/backward step), executing sequential gradient updates across the 35 training cities in each epoch for a maximum budget of 200 epochs. Gradients are clipped to a maximum Euclidean norm of $5.0$ before each parameter update. Network weights are initialized across three independent random seeds $\mathcal{S} = \{1, 10, 100\}$.

*(Tiếng Việt: Cả hai mô hình neural Urban GNN và Node MLP đều dùng chung một quy trình huấn luyện bằng AdamW với learning rate khởi tạo $\eta = 2 \times 10^{-3}$ và weight decay $\lambda_{\mathrm{wd}} = 10^{-4}$. Regularization được thực thi hoàn toàn thông qua cơ chế weight decay của bộ tối ưu; hàm mất mát không cộng thêm số hạng phạt $\ell_2$. Giá trị weight decay được cố định tiên nghiệm không qua grid search. Dropout ($p = 0.1$) và LayerNorm cung cấp cơ chế điều hòa kiến trúc nội tại. Quá trình tối ưu sử dụng batching theo từng thành phố (chuyển toàn bộ $|\Omega_{c,\mathrm{inter}}^+|$ cặp liên vùng dương của một thành phố trong mỗi bước), cập nhật gradient tuần tự qua 35 thành phố huấn luyện trong tối đa 200 epoch, cắt chuẩn gradient ở mức 5.0 và lặp lại trên ba seed ngẫu nhiên độc lập $\mathcal{S} = \{1, 10, 100\}$.)*

#### Gravity parameter estimation
The two parameters of the classical gravity baseline ($G \in \mathbb{R}$ and $\alpha > 0$) are learned model parameters estimated via Ordinary Least Squares (OLS). They are not regularization hyperparameters. Estimation minimizes the sum of squared log-intensity residuals across the pooled positive interzonal training pairs of $\mathcal{C}_{\mathrm{train}}^{(f)}$ using `np.linalg.lstsq(rcond=None)`. Because this normal-equation formulation admits an exact, closed-form linear algebra solution, it requires no iterative gradient optimizer, no initial values, no convergence threshold, and zero restarts. The decay exponent $\alpha$ is unconstrained during fitting and empirically yields $\alpha \approx 1.09 - 1.22 > 0$ across folds. Parameters are fitted once per fold strictly on source training cities $\mathcal{C}_{\mathrm{train}}^{(f)}$ and are held frozen during test evaluation.

*(Tiếng Việt: Hai tham số của mô hình gravity cổ điển ($G \in \mathbb{R}$ và $\alpha > 0$) là các learned model parameters được ước lượng bằng hồi quy OLS đại số tuyến tính nghiệm đóng, không phải là siêu tham số regularization. Mô hình được giải bằng `np.linalg.lstsq` trên tập gộp các cặp liên vùng dương của $\mathcal{C}_{\mathrm{train}}^{(f)}$, không cần optimizer lặp, không cần giá trị khởi tạo, không cần ngưỡng hội tụ và không có restart. Hai tham số được khớp độc lập một lần cho mỗi fold trên $\mathcal{C}_{\mathrm{train}}^{(f)}$ và giữ cố định khi đánh giá.)*

#### Checkpoint selection and validation protocol
Model selection adheres strictly to the cross-city validation protocol (35 training, 5 validation, and 10 test cities per fold; see Section 3.6.1 for detailed split definitions). No sub-city, tract-level, or origin-zone cross-validation folds are created within cities, preserving the city-level zero-shot transfer structure:

1. **Training Objective vs. Validation Criterion**: The training objective minimized by AdamW on $\mathcal{C}_{\mathrm{train}}^{(f)}$ is the ZTNB negative log-likelihood $\mathcal{L}_{\mathrm{neural}}$. In contrast, checkpoint tracking and early stopping on $\mathcal{C}_{\mathrm{val}}^{(f)}$ are governed strictly by macro-averaged interzonal **Validation CPC** ($\operatorname{CPC}_{\mathrm{val}}$):
   $$\operatorname{CPC}_{\mathrm{val}} = \frac{1}{|\mathcal{C}_{\mathrm{val}}^{(f)}|} \sum_{c \in \mathcal{C}_{\mathrm{val}}^{(f)}} \operatorname{CPC}_c\left(\widehat{\mathbf{T}}_c^{(0,m)}, \mathbf{t}_{c}\right)$$
   Validation loss is not used for checkpoint selection, preventing likelihood probability density from superseding spatial mass overlap.
2. **Learning Rate Scheduling**: A `ReduceLROnPlateau` scheduler monitors $\operatorname{CPC}_{\mathrm{val}}$ in `max` mode. When $\operatorname{CPC}_{\mathrm{val}}$ fails to improve by at least $\min\_delta = 10^{-4}$ for $4$ consecutive epochs, the learning rate is scaled down by a factor of $0.5$ (bounded below by $\min\_lr = 10^{-5}$).
3. **Early Stopping**: Training terminates early if validation CPC does not achieve a new best value for $15$ consecutive epochs (patience $= 15$). The model state dict corresponding to the epoch with the highest $\operatorname{CPC}_{\mathrm{val}}$ is restored as the final trained model $\theta^*$.
4. **Target City Exclusion**: Test cities $\mathcal{C}_{\mathrm{test}}^{(f)}$ play zero role in hyperparameter tuning, checkpoint selection, or early stopping decisions.

*(Tiếng Việt: Quá trình lựa chọn mô hình tuân thủ nghiêm ngặt giao thức kiểm định chéo cấp đô thị 35/5/10 (tham chiếu Mục 3.6.1), không chia nhỏ các vùng xuất phát hay cặp OD trong thành phố thành các fold phụ. Trong khi objective huấn luyện trên $\mathcal{C}_{\mathrm{train}}^{(f)}$ là ZTNB NLL, tiêu chí chọn checkpoint trên tập validation là chỉ số **Validation CPC** ($\operatorname{CPC}_{\mathrm{val}}$) trung bình trên 5 thành phố validation; validation loss không được dùng để chọn checkpoint. Scheduler `ReduceLROnPlateau` giảm một nửa learning rate nếu $\operatorname{CPC}_{\mathrm{val}}$ không cải thiện sau 4 epoch. Early stopping dừng huấn luyện nếu không có cải thiện tối thiểu $10^{-4}$ trong 15 epoch liên tiếp và phục hồi lại checkpoint có $\operatorname{CPC}_{\mathrm{val}}$ cao nhất. Các thành phố kiểm tra hoàn toàn không tham gia vào bước lựa chọn này.)*

#### Permanent parameter freezing
Following checkpoint selection on validation cities, all model parameters—including neural weights $\theta_{\mathrm{GNN}}^*$, $\theta_{\mathrm{MLP}}^*$, dispersion $\phi^*$, and gravity parameters $(G^*, \alpha^*)$—are permanently frozen (`requires_grad = False`). 

During zero-shot target-city inference, target city $c$ provides only permissible static spatial data: tract centroid coordinates $\mathbf{s}_{c,i}$, normalized urban context features $\mathbf{x}_{c,i}$, and tract populations $P_{c,i}$, evaluated over the known positive interzonal support $\Omega_{c,\mathrm{inter}}^+$. Target-city flow intensities $t_{c,ij}$ are never accessed during this forward pass. The output of this stage constitutes the uncalibrated zero-shot baseline prediction $\widehat{T}_{c,ij}^{(0,m)}$ ($M_0$ condition). Baseline predictions are generated strictly prior to the introduction of the target city's distance-binned observation in the calibration stage.

*(Tiếng Việt: Sau khi chọn xong checkpoint trên các thành phố validation, toàn bộ tham số mô hình—bao gồm trọng số neural $\theta_{\mathrm{GNN}}^*$, $\theta_{\mathrm{MLP}}^*$, độ phân tán $\phi^*$ và tham số gravity $(G^*, \alpha^*)$—đều được đóng băng vĩnh viễn (`requires_grad = False`). Khi suy luận trên thành phố mục tiêu $c$, mô hình chỉ tiếp nhận các dữ liệu không gian tĩnh được phép trên tập hỗ trợ $\Omega_{c,\mathrm{inter}}^+$. Cường độ luồng của thành phố mục tiêu không bao giờ được dùng để cập nhật hay tinh chỉnh trọng số. Dự báo baseline trước hiệu chỉnh $\widehat{T}_{c,ij}^{(0,m)}$ ($M_0$) được tạo ra hoàn toàn độc lập trước khi tín hiệu phân phối khoảng cách mục tiêu được đưa vào toán tử hiệu chỉnh.)*

#### Table 3: Hyperparameter and parameter status taxonomy
*(Tiếng Việt: **Bảng 3: Phân loại trạng thái siêu tham số và cấu hình tối ưu hóa**)*

| Component | Quantity | Value or candidate set | Status | Selection data |
| :--- | :--- | :---: | :---: | :--- |
| **Neural optimizer** | Algorithm | AdamW (`torch.optim.AdamW`) | Fixed | Standard deep learning configuration |
| **Learning rate** | Initial step size ($\eta$) | $2 \times 10^{-3}$ | Fixed | Standard deep learning configuration |
| **Weight decay** | Decoupled parameter shrinkage ($\lambda_{\mathrm{wd}}$) | $10^{-4}$ | Fixed | Standard deep learning configuration |
| **Dropout** | Layer dropout rate ($p$) | $0.1$ | Fixed | Standard deep learning configuration |
| **Batch size** | Optimization grouping | 1 city per step ($|\Omega_{c,\mathrm{inter}}^+|$ pairs) | Fixed | City-level batching |
| **Maximum epochs** | Training budget | 200 epochs | Fixed | Computational budget |
| **Early-stopping patience** | Convergence stopping criterion | 15 epochs without improvement $\ge 10^{-4}$ | Fixed | Monitored on validation cities ($\mathcal{C}_{\mathrm{val}}^{(f)}$) |
| **Checkpoint metric** | Model selection criterion | Macro-averaged interzonal $\operatorname{CPC}_{\mathrm{val}}$ | Selected on validation cities | Validation partition ($\mathcal{C}_{\mathrm{val}}^{(f)}$) |
| **Model seeds** | Random initializations | $\mathcal{S} = \{1, 10, 100\}$ | Fixed | Random initialization seeds |
| **Gravity solver** | Fitting algorithm | Ordinary Least Squares (`np.linalg.lstsq`) | Fixed | Closed-form analytical OLS |
| **Gravity parameters** | Global intercept $G$, decay $\alpha$ | $G \in \mathbb{R}$, $\alpha > 0$ (unconstrained OLS) | Learned on training cities | Training partition ($\mathcal{C}_{\mathrm{train}}^{(f)}$) |
| **Primary distance bins** | Number of moving-distance bins ($K$) | $K = 8$ | Fixed | Pre-specified quantile partition |
| **Calibration strength** | Modulation domain & canonical value | $\mathcal{Q} = [0, 1]$ (canonical $q = 1.0$) | Fixed | Pre-specified analytical default ($M_1$) |
| **Calibration selection metric**| Calibration parameter rule | Analytical matching ($q = 1.0$; no grid search) | Fixed | Fixed default without test-city tuning |
| **Alternative distance bins**| Bin granularity sweep | $K \in \{2, 4, 6, 8, 10, 12, 14, 16, 18, 20\}$ | Sensitivity setting | Robustness analysis (Section 4.3) |
| **Synthetic noise levels** | TV noise magnitude ($\epsilon$) | $\epsilon \in \{0.00, 0.01, 0.02, 0.03, 0.04, 0.05\}$ | Sensitivity setting | Robustness analysis (Section 4.4) |
| **Neural model parameters** | Network weights $\theta_m$, dispersion $\log \phi$| Continuous parameter tensors | Learned on training cities | Training partition ($\mathcal{C}_{\mathrm{train}}^{(f)}$) |

*Note: In accordance with the protocol taxonomy, "Fixed" denotes configurations pre-specified before experimentation; "Learned on training cities" denotes quantities optimized strictly on $\mathcal{C}_{\mathrm{train}}^{(f)}$; "Selected on validation cities" denotes model checkpoint choices governed by $\mathcal{C}_{\mathrm{val}}^{(f)}$; and "Sensitivity setting" denotes variations evaluated exclusively in secondary robustness analyses. Test-city flow observations are never used for parameter learning, validation selection, or hyperparameter tuning.*

---

### 3.5.8 Target-City Distance-Binned Observation
*(Tiếng Việt: **3.5.8. Quan sát theo khoảng khoảng cách của thành phố mục tiêu**)*

The distance continuum is partitioned into $K$ intervals $I_b = [a_{b-1}, a_b)$ ($b = 1, \dots, K$) using pair-weighted quantiles estimated strictly from training cities ($a_0 = 0, a_K = \infty$). The primary benchmark fixes the number of moving-distance intervals at $K = 8$ a priori (`K_MOVE = 8`). Alternative bin resolutions $K \in \{2, 4, 6, 10, 12, 14, 16, 18, 20\}$ are evaluated exclusively as secondary sensitivity settings in Section 4.3, rather than being selected via validation grid search.

For target city $c$, the set of candidate interzonal pairs falling into distance interval $b$ is:

$$\mathcal{B}_{c,b} = \left\{ (i,j) \in \Omega_{c,\mathrm{inter}}^+ : a_{b-1} \le d_{c,ij} < a_b \right\}$$

The total observed reference trip volume falling into interval $b$ is $F_{c,b} = \sum_{(i,j) \in \mathcal{B}_{c,b}} t_{c,ij}$. The normalized target distance distribution is:

$$\mathbf{Y}_{D,c} = [Y_{D,c,1}, \dots, Y_{D,c,K}]^T \in \Delta^{K-1}, \qquad Y_{D,c,b} = \frac{F_{c,b}}{\sum_{r=1}^K F_{c,r}}, \quad \sum_{b=1}^K Y_{D,c,b} = 1$$

Crucially, $\mathbf{Y}_{D,c}$ is a normalized probability distribution over coarse distance intervals, not a vector of absolute trip volumes. It supplies no individual OD-pair flow quantities. In this benchmark, $\mathbf{Y}_{D,c}$ is extracted directly from target ground-truth flows as an oracle aggregate signal, serving as a controlled probe into the incremental value of macro travel patterns. It is supplied strictly at inference time to the calibration operator.

*(Tiếng Việt: Miền khoảng cách được chia thành $K$ khoảng $I_b = [a_{b-1}, a_b)$ ($b=1,\dots,K$) dựa trên phân vị cặp luồng từ các thành phố huấn luyện. Cấu hình chính cố định $K=8$ tiên nghiệm; các giá trị $K \in \{2, 4, \dots, 20\}$ khác chỉ là các thiết lập phân tích độ nhạy (sensitivity settings) trong Mục 4.3, không phải siêu tham số chọn bằng validation. Vector phân phối chuẩn hóa mục tiêu là $\mathbf{Y}_{D,c} \in \Delta^{K-1}$ với $\sum_{b=1}^K Y_{D,c,b} = 1$, đóng vai trò tín hiệu oracle vĩ mô được cung cấp tại bước hiệu chỉnh.)*

---

### 3.5.9 Unified Analytical Inference-Time Calibration Operator
*(Tiếng Việt: **3.5.9. Toán tử hiệu chỉnh giải tích dùng chung tại thời điểm suy luận**)*

Given any frozen baseline predictor $m \in \{\text{GNN}, \text{MLP}, \text{Grav}\}$ generating initial predictions $\widehat{T}_{c,ij}^{(0,m)}$ on $\Omega_{c,\mathrm{inter}}^+$, the calibration operator executes the following deterministic reallocation:

1. **Baseline predicted bin mass**: The flow volume assigned by baseline $m$ to distance interval $b$ is:
   $$\widehat{F}_{c,b}^{(0,m)} = \sum_{(i,j) \in \mathcal{B}_{c,b}} \widehat{T}_{c,ij}^{(0,m)}$$

2. **Total baseline interzonal volume**:
   $$\widehat{S}_c^{(0,m)} = \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{c,ij}^{(0,m)}$$

3. **Implied baseline distance distribution**:
   $$\widehat{Y}_{D,c,b}^{(0,m)} = \frac{\widehat{F}_{c,b}^{(0,m)}}{\widehat{S}_c^{(0,m)}}$$

4. **Active interval identification**: Distance intervals containing at least one candidate pair with non-zero baseline prediction are identified as:
   $$\mathcal{A}_c^{(m)} = \left\{ b \in \{1, \dots, K\} : \widehat{Y}_{D,c,b}^{(0,m)} > 0 \right\}$$

5. **Target conditioning on active support**:
   $$p_{c,b}^{\mathrm{cond}} = \frac{Y_{D,c,b} \mathbb{I}(b \in \mathcal{A}_c^{(m)})}{\sum_{r \in \mathcal{A}_c^{(m)}} Y_{D,c,r}}$$

6. **Soft response ratio and weighting**: For $b \in \mathcal{A}_c^{(m)}$, the unnormalized adjustment ratio is modulated by calibration strength $q \in [0, 1]$:
   $$w_{c,b}(q) = \left( \frac{p_{c,b}^{\mathrm{cond}}}{\widehat{Y}_{D,c,b}^{(0,m)}} \right)^q$$

7. **Mass-preserving normalizing scalar**:
   $$Z_c^{(m)}(q) = \sum_{r \in \mathcal{A}_c^{(m)}} \widehat{Y}_{D,c,r}^{(0,m)} w_{c,r}(q), \qquad s_{c,b}(q) = \frac{w_{c,b}(q)}{Z_c^{(m)}(q)}$$

8. **Calibrated flow prediction**:
   $$\widehat{T}_{c,ij}^{(1,m)} = \operatorname{Calibrate}\left(\widehat{T}_{c,ij}^{(0,m)}, \mathbf{Y}_{D,c}\right) = s_{c,b(i,j)}(q) \cdot \widehat{T}_{c,ij}^{(0,m)}$$
   where $b(i,j)$ denotes the distance interval enclosing $d_{c,ij}$.

This mapping transforms baseline predictions into calibrated predictions:
- GNN baseline $\widehat{T}_{c,ij}^{(0,\mathrm{GNN})}$ ($M_0$) $\longrightarrow$ GNN calibrated $\widehat{T}_{c,ij}^{(1,\mathrm{GNN})}$ ($M_1$ or $M1_{\mathrm{city}}$);
- MLP baseline $\widehat{T}_{c,ij}^{(0,\mathrm{MLP})}$ $\longrightarrow$ MLP calibrated $\widehat{T}_{c,ij}^{(1,\mathrm{MLP})}$;
- Gravity baseline $\widehat{T}_{c,ij}^{(0,\mathrm{Grav})}$ $\longrightarrow$ Gravity calibrated $\widehat{T}_{c,ij}^{(1,\mathrm{Grav})}$.

For the sub-metropolitan spatial resolution variant (`M1_county`), the identical operator is applied independently within each origin-county group $\Omega_{c,\ell}^+ = \{(i,j) \in \Omega_{c,\mathrm{inter}}^+ : g(i) = \ell\}$ using county-level distribution $\mathbf{Y}_{D,c,\ell}$, yielding $\widehat{\mathbf{T}}_c^{\mathrm{county},m}$.

#### Calibration-Strength Selection
The parameter $q \in [0, 1]$ governs calibration response strength: at $q = 0$, $w_{c,b} = 1$, which leaves predictions unaltered ($\widehat{T}^{(1)} \equiv \widehat{T}^{(0)}$, $M_0$ baseline); at $q = 1$, the operator enforces full proportional matching with the target distribution. Intermediate values $q \in (0, 1)$ provide continuous shrinkage.

In our experimental pipeline, calibration strength is **pre-specified and fixed a priori at $q = 1.0$** (`Q_CALIB = 1.0` across all experiments). The parameter $q$ is not selected via an empirical validation grid search, nor is it tuned per fold, per backbone, or per city. Crucially, target-city ground truth flow volumes $t_{c,ij}$, test-city CPC values, and test aggregate performance metrics are strictly forbidden from participating in any selection or tuning of $q$. Following this pre-specified design, $q = 1.0$ is held identical and invariant across all cross-validation folds, test cities, and model architectures.

**Non-Iterative Post-Processing Principle**: Calibration is strictly an analytical, closed-form post-processing operator, not an iterative training, fine-tuning, or retraining step. The bin scaling weights $s_{c,b}(q)$ are evaluated directly in closed form from baseline predictions and the oracle observation $\mathbf{Y}_{D,c}$. The calibration operator executes zero gradient descent passes, does not update neural weights, does not re-fit gravity parameters, and introduces no test-city parameter optimization.

*(Tiếng Việt: Với bất kỳ mô hình baseline nào $m \in \{\text{GNN}, \text{MLP}, \text{Grav}\}$ tạo ra dự báo $\widehat{T}_{c,ij}^{(0,m)}$ trên $\Omega_{c,\mathrm{inter}}^+$, toán tử hiệu chỉnh giải tích thực hiện tái phân bổ xác định bằng 8 bước giải tích: (1) khối lượng dự báo theo bin $\widehat{F}_{c,b}^{(0,m)}$; (2) tổng thể tích liên vùng $\widehat{S}_c^{(0,m)}$; (3) phân phối khoảng cách ngầm định $\widehat{Y}_{D,c,b}^{(0,m)}$; (4) tập khoảng hoạt động $\mathcal{A}_c^{(m)}$; (5) phân phối mục tiêu điều kiện hóa $p_{c,b}^{\mathrm{cond}}$; (6) tỷ lệ hiệu chỉnh mềm $w_{c,b}(q) = (p_{c,b}^{\mathrm{cond}} / \widehat{Y}_{D,c,b}^{(0,m)})^q$; (7) hệ số chuẩn hóa bảo toàn khối lượng $Z_c^{(m)}(q)$ và $s_{c,b}(q)$; và (8) dự báo đã hiệu chỉnh $\widehat{T}_{c,ij}^{(1,m)} = s_{c,b(i,j)}(q) \cdot \widehat{T}_{c,ij}^{(0,m)}$. Tham số điều tiết $q$ được khóa tiên nghiệm cố định tại $q = 1.0$ (`Q_CALIB = 1.0`), không thực hiện grid search trên tập validation và tuyệt đối không sử dụng dữ liệu kiểm tra để tinh chỉnh $q$. Hiệu chỉnh là toán tử hậu xử lý phi lặp nghiệm đóng, không cập nhật trọng số hay chạy gradient descent trên thành phố mục tiêu.)*

---

### 3.5.10 Preserved Mathematical Invariants
*(Tiếng Việt: **3.5.10. Các đặc tính toán học bất biến được bảo toàn**)*

The analytical calibration operator strictly guarantees three mathematical properties:

1. **Support Invariance**: Calibration is strictly restricted to candidate pairs within $\Omega_{c,\mathrm{inter}}^+$. The operator neither creates new OD links where none existed nor sets existing candidate links to zero. The evaluation domain remains identical across all stages:
   $$\Omega_{c,\mathrm{inter}}^+(M_0) \equiv \Omega_{c,\mathrm{inter}}^+(M_1) \equiv \Omega_{c,\mathrm{inter}}^+(M1_{\mathrm{county}})$$

2. **Within-Bin Rank Preservation**: Because all OD pairs $(i,j)$ belonging to distance interval $b$ are scaled by the exact same positive scalar factor $s_{c,b}(q) > 0$, the ratio between any two predictions in the same interval is strictly invariant:
   $$\frac{\widehat{T}_{c,ij}^{(1,m)}}{\widehat{T}_{c,uv}^{(1,m)}} = \frac{s_{c,b}(q) \cdot \widehat{T}_{c,ij}^{(0,m)}}{s_{c,b}(q) \cdot \widehat{T}_{c,uv}^{(0,m)}} = \frac{\widehat{T}_{c,ij}^{(0,m)}}{\widehat{T}_{c,uv}^{(0,m)}}, \qquad \forall (i,j), (u,v) \in \mathcal{B}_{c,b}$$
   Consequently, intra-bin ranking is mathematically preserved (Kendall's rank correlation within every non-degenerate distance interval is identically $\tau = 1.00000000$). Calibration acts exclusively as a macro-scale mass reallocation across distance bins; it cannot reorder misranked pairs within a bin.

3. **Total-Mass Preservation**: Normalization by $Z_c^{(m)}(q)$ mathematically guarantees that total predicted interzonal flow volume is strictly conserved:
   $$\sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{c,ij}^{(1,m)} = \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{c,ij}^{(0,m)}$$
   This conservation holds analytically across all $q \in [0, 1]$ and is enforced in the software pipeline by numerical sanity checks verifying relative error $|S_{\mathrm{cal}} - S_0| / S_0 < 10^{-5}$.

Figure 1 illustrates the complete support-conditioned zero-shot modeling and inference-time calibration pipeline.

*(Tiếng Việt: Toán tử hiệu chỉnh giải tích bảo đảm nghiêm ngặt ba đặc tính toán học: (1) **Bảo toàn tập hỗ trợ**: miền đánh giá $\Omega_{c,\mathrm{inter}}^+$ được giữ nguyên tuyệt đối, không sinh thêm liên kết mới và không xóa bỏ liên kết cũ; (2) **Bảo toàn thứ hạng nội khoảng**: do tất cả các cặp trong cùng khoảng $b$ đều được nhân với cùng một hệ số dương $s_{c,b}(q) > 0$, tỷ số và thứ bậc tương đối giữa chúng không thay đổi, tương quan hạng Kendall nội khoảng đạt chính xác $\tau = 1.0$; và (3) **Bảo toàn tổng khối lượng**: hệ số chuẩn hóa $Z_c(q)$ đảm bảo tổng lưu lượng liên vùng dự báo sau hiệu chỉnh bằng đúng tổng lưu lượng dự báo trước hiệu chỉnh.)*

![Figure 1](figures/fig1_oracle_calibration_framework.svg)
**Figure 1. Support-conditioned oracle calibration framework.** The cross-city model $M_0$ is trained on source cities and frozen before target-city inference. For a target city, $M_0$ first produces baseline intensities $\widehat{\mathbf{T}}_c^{(0)}$ on the known positive support $\Omega_{c,\mathrm{inter}}^+$. The oracle distance-binned distribution $\mathbf{Y}_{D,c}$ is deterministically derived from the same target-city positive ground-truth OD flows used for evaluation and is introduced only at inference time. Bin-specific scaling factors reallocate predicted mass across distance intervals to obtain $\widehat{\mathbf{T}}_c^{(1)}$ without updating model parameters or creating new OD links. The schematic represents an oracle information intervention, not an independently collected external telemetry pipeline.

*(Tiếng Việt: **Hình 1. Framework hiệu chỉnh oracle có điều kiện theo support.** Mô hình cross-city $M_0$ được huấn luyện trên các thành phố nguồn và đóng băng trước khi suy luận trên thành phố mục tiêu. Đối với một thành phố mục tiêu, $M_0$ trước hết tạo ra dự báo cường độ baseline $\widehat{\mathbf{T}}_c^{(0)}$ trên tập hỗ trợ dương đã biết $\Omega_{c,\mathrm{inter}}^+$. Phân phối theo nhóm khoảng cách oracle $\mathbf{Y}_{D,c}$ được xác định trực tiếp từ chính các luồng OD ground-truth dương của thành phố mục tiêu đang được sử dụng để đánh giá và chỉ được đưa vào tại thời điểm suy luận. Các hệ số theo bin tái phân bổ khối lượng dự báo giữa các khoảng cự ly để tạo $\widehat{\mathbf{T}}_c^{(1)}$ mà không cập nhật tham số mô hình hoặc tạo liên kết OD mới. Sơ đồ biểu diễn một can thiệp thông tin oracle, không phải pipeline telemetry bên ngoài được thu thập độc lập.)*

---

## 3.6 Cross-City Evaluation Protocol and Statistical Inference
*(Tiếng Việt: **3.6. Giao thức đánh giá cross-city và suy luận thống kê**)*

### 3.6.1 5-Fold Cross-City Validation Scheme
The empirical benchmark is structured around a 5-fold cross-validation protocol over $N=50$ U.S. metropolitan areas. In each fold, 35 cities are used for model training, 5 cities for model selection (validation), and 10 cities for evaluation (testing). Every city appears in the test partition exactly once, covering all 50 metropolitan areas across folds.

The partitioning unit is the entire city rather than OD pairs, tracts, or observation samples within the same city. Consequently, all tracts and OD pairs belonging to a given city reside exclusively within a single partition (training, validation, or testing) in each fold, and are not dispersed across splits. This city-level division provides the necessary condition to support the cross-city zero-shot evaluation claim.

Distance bin edges are calculated independently for each fold using only interzonal OD pairs from the training cities. Following training completion, backbone model parameters are permanently frozen prior to target-city inference.

For each target city, three primary model conditions are evaluated:
- $M_0$: Zero-shot predicted flows $\widehat{\mathbf{T}}_c^{(0)}$ without access to $\mathbf{Y}_{D,c}$;
- $M1_{\mathrm{city}}$: Analytically calibrated flows $\widehat{\mathbf{T}}_c^{(1)}$ using a single oracle $\mathbf{Y}_{D,c}$ at the city level (Primary Benchmark);
- $M1_{\mathrm{county}}$: Analytically calibrated flows $\widehat{\mathbf{T}}_c^{\mathrm{county}}$ using multiple oracle distributions grouped by origin county (Spatial Resolution Variant).

The comparison between $M_0$ and $M1_{\mathrm{city}}$ represents the primary experiment designed to evaluate whether target distance distributions provide incremental information for zero-shot reconstruction (RQ1). The comparison between $M1_{\mathrm{city}}$ and $M1_{\mathrm{county}}$ provides empirical evidence for the spatial observational resolution aspect of RQ2.

Across all configurations, predictions are evaluated on the exact same observed positive interzonal support $\Omega_{c,\mathrm{inter}}^+$ for the entire city.

*(Tiếng Việt: Nghiên cứu áp dụng giao thức kiểm định chéo liên thành phố 5-fold trên 50 vùng đô thị của Hoa Kỳ. Trong mỗi fold, 35 thành phố được dùng để huấn luyện, 5 thành phố dùng để lựa chọn mô hình (validation) và 10 thành phố dùng để đánh giá (testing). Mỗi thành phố xuất hiện trong tập kiểm tra đúng một lần, bao phủ toàn bộ 50 đô thị qua các fold. Đơn vị phân chia fold là toàn bộ thành phố, không phải các cặp OD, tract hoặc mẫu quan sát trong cùng một thành phố. Do đó, các cặp OD hoặc tract của cùng một thành phố không bị phân tán giữa training, validation và test mà nằm trọn vẹn trong một tập duy nhất của mỗi fold. Việc phân chia ở cấp thành phố này là điều kiện cần để hỗ trợ claim zero-shot liên thành phố. Các biên khoảng cách được tính riêng cho từng fold và chỉ sử dụng khoảng cách của các cặp OD thuộc tập thành phố huấn luyện. Sau khi huấn luyện hoàn tất, tham số của mô hình được giữ cố định trước khi dự báo trên các thành phố kiểm tra. Đối với mỗi thành phố mục tiêu, ba cấu hình được phân biệt: $M_0$ (dự báo zero-shot không sử dụng $Y_D$), $M1_{\mathrm{city}}$ (hiệu chỉnh bằng một $Y_D$ oracle ở cấp city), và $M1_{\mathrm{county}}$ (hiệu chỉnh bằng nhiều $Y_D$ oracle được phân nhóm theo county). So sánh giữa $M_0$ và $M1_{\mathrm{city}}$ là thí nghiệm chính nhằm trả lời liệu phân phối khoảng cách của thành phố mục tiêu có bổ sung thông tin cho dự báo zero-shot hay không (RQ1). So sánh giữa $M1_{\mathrm{city}}$ và $M1_{\mathrm{county}}$ cung cấp bằng chứng cho khía cạnh độ phân giải không gian của quan sát trong RQ2. Trong tất cả cấu hình, mô hình dự báo và được đánh giá trên cùng tập hỗ trợ dương $\Omega_{c,\mathrm{inter}}^+$ của toàn thành phố.)*

### 3.6.2 Evaluation Metrics and Model Comparison
*(Tiếng Việt: **3.6.2. Thước đo đánh giá và so sánh mô hình**)*

#### Primary evaluation metric: Common Part of Commuters (CPC)
The primary quantitative metric for evaluating zero-shot travel flow reconstruction is the Common Part of Commuters (CPC) [@lenormand2016comparison], evaluated on the known positive interzonal support $\Omega_{c,\mathrm{inter}}^+$:

$$\operatorname{CPC}_c = \frac{2 \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \min\left(t_{c,ij}, \widehat{T}_{c,ij}\right)}{\sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{c,ij} + \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{c,ij}}$$

where the positive interzonal evaluation domain is formally defined as:

$$\Omega_{c,\mathrm{inter}}^+ = \left\{ (i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij}\geq1,\ i\neq j,\ d_{c,ij}>0 \right\}.$$

The properties and operational boundaries of CPC within this evaluation protocol are as follows:
1. **Shared volume proportion**: CPC measures the fraction of travel demand volume jointly shared between the observed ground-truth flow field and the model's predicted intensity surface. It is strictly bounded in $[0, 1]$, where $\operatorname{CPC} = 1$ denotes perfect agreement across all OD pairs and $\operatorname{CPC} = 0$ indicates zero overlap.
2. **Filtering of intrazonal self-flows and zero-distance pairs**: Intrazonal loops ($i = j$) and pairs with non-positive spatial displacement ($d_{c,ij} \le 0$) are strictly excluded (`pair_o_idx != pair_d_idx` and `dist_km > 0.0`). The benchmark focuses exclusively on displacement flows connecting distinct geographic zones.
3. **Continuous intensity evaluation**: Model predictions $\widehat{T}_{c,ij} \in (0, \infty)$ represent expected positive intensities under the conditional ZTNB head. Predictions are evaluated as continuous floating-point values without integer discretization or rounding.
4. **Restriction to positive interzonal support**: CPC is computed over $\Omega_{c,\mathrm{inter}}^+$, evaluating displacement intensity reconstruction conditional on the known positive link support. It does not measure binary classification accuracy or the identification of structural zero-flow pairs.
5. **Flow normalization and scale preservation**: Zero-shot baseline predictions $\widehat{\mathbf{T}}_c^{(0)}$ are evaluated directly without rescaling to ground-truth total volume $N_c$ (which is unobserved at inference time). The analytical calibration operator strictly preserves the baseline model's total predicted flow volume ($\sum_{(i,j)} \widehat{T}_{c,ij}^{(1)} = \sum_{(i,j)} \widehat{T}_{c,ij}^{(0)}$), ensuring that any change in CPC reflects a genuine spatial reallocation across distance intervals rather than an artificial volume adjustment.

#### Paired improvement estimand
For target city $c$, predictor architecture $m$, and model initialization seed $s$, the paired performance change induced by distance calibration is defined as:

$$\Delta_{c,s}^{(m)} = \operatorname{CPC}_{c,s}\left(\widehat{\mathbf{T}}_c^{(1,m)}\right) - \operatorname{CPC}_{c,s}\left(\widehat{\mathbf{T}}_c^{(0,m)}\right) = \operatorname{CPC}_{c,s}\left(M1_{\mathrm{city}}^{(m)}\right) - \operatorname{CPC}_{c,s}\left(M_0^{(m)}\right)$$

where $m \in \{\text{GNN}, \text{MLP}, \text{Gravity}\}$.

This formulation establishes a rigorous paired counterfactual comparison:
* Baseline ($M_0$) and calibrated ($M_1$) predictions are generated for the exact same target city;
* Evaluated on the identical positive interzonal support $\Omega_{c,\mathrm{inter}}^+$;
* Produced by the identical frozen neural weights or fitted parameters under model seed $s$;
* Conditioned on the identical target urban features and distance bin partitions;
* The sole experimental intervention is the presence versus absence of target distance-binned conditioning during inference-time post-processing.

The Gravity-Informed Urban GNN ($m = \text{GNN}$) defines the primary confirmatory estimand of this study. The Pairwise Node MLP ($m = \text{MLP}$) and Classical Two-Parameter Gravity ($m = \text{Gravity}$) provide structural robustness comparisons; their results are evaluated independently and are never pooled into a single composite score.

#### Secondary error metrics
To verify that empirical findings do not depend idiosyncratically on the functional form of CPC, six secondary error metrics are computed on the identical support $\Omega_{c,\mathrm{inter}}^+$:
1. **Mean Absolute Error (MAE)**:
   $$\operatorname{MAE}_c = \frac{1}{|\Omega_{c,\mathrm{inter}}^+|} \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \left| t_{c,ij} - \widehat{T}_{c,ij} \right|$$
   Expressed in commuters per link; lower values indicate higher accuracy. MAE applies a linear penalty across all link magnitudes.
2. **Root Mean Squared Error (RMSE)**:
   $$\operatorname{RMSE}_c = \sqrt{\frac{1}{|\Omega_{c,\mathrm{inter}}^+|} \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \left( t_{c,ij} - \widehat{T}_{c,ij} \right)^2}$$
   Expressed in commuters per link; lower values indicate higher accuracy. RMSE penalizes large flow discrepancies quadratically, emphasizing performance on major travel corridors.
3. **Normalized Root Mean Squared Error (NRMSE)**:
   $$\operatorname{NRMSE}_c = \frac{\operatorname{RMSE}_c}{\bar{t}_c}, \qquad \bar{t}_c = \frac{1}{|\Omega_{c,\mathrm{inter}}^+|} \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{c,ij}$$
   Dimensionless; lower values indicate higher accuracy. NRMSE standardizes link errors by the mean observed flow, facilitating cross-city comparability across urban areas with differing population scales.
4. **Log-Transformed RMSE ($\operatorname{RMSE}_{\log1p}$)**:
   $$\operatorname{RMSE}_{\log1p,c} = \sqrt{\frac{1}{|\Omega_{c,\mathrm{inter}}^+|} \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \left[ \log(1 + t_{c,ij}) - \log\left(1 + \widehat{T}_{c,ij}\right) \right]^2}$$
   Dimensionless; lower values indicate higher accuracy. By compressing the heavy-tailed distribution of commuter flows, $\operatorname{RMSE}_{\log1p}$ evaluates proportional accuracy across small, medium, and high-volume OD links.
5. **Spearman Rank Correlation Coefficient ($\rho_{\mathrm{Spearman}}$)**:
   Measures the monotonicity of predicted versus observed flows on $\Omega_{c,\mathrm{inter}}^+$; dimensionless in $[-1, 1]$, where higher values indicate better preservation of relative traffic hierarchy.
6. **Total Flow Relative Error ($\operatorname{RelError}_c$)**:
   $$\operatorname{RelError}_c = \frac{\left| \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{c,ij} - \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{c,ij} \right|}{\sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{c,ij}}$$
   Dimensionless; lower values indicate closer macroeconomic alignment with total city-wide commuter volume.

CPC remains the primary evaluation criterion for all hypothesis testing and headline claims. Secondary metrics serve strictly as sensitivity and diagnostic checks.

#### Distance-distribution diagnostic
In addition to link-level OD accuracy, the pipeline tracks bin-level aggregate flow distributions:
$$\widehat{Y}_{D,c,b}^{(m)} = \frac{\sum_{(i,j) \in \mathcal{B}_{c,b}} \widehat{T}_{c,ij}^{(m)}}{\sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{c,ij}^{(m)}}, \qquad b = 1, \dots, K$$

This metric serves exclusively as an internal mechanistic diagnostic:
* It verifies whether the analytical calibration operator successfully realigns the baseline model's distance-decay profile toward the target profile $\mathbf{Y}_{D,c}$;
* It does not evaluate microscopic OD pair reconstruction or intra-bin flow allocations (since intra-bin rankings are mathematically invariant under scalar bin multiplication);
* Because the target distribution $\mathbf{Y}_{D,c}$ is supplied as an input to the calibration operator, agreement between calibrated aggregate flows and $\mathbf{Y}_{D,c}$ is an operational verification of algorithm execution, not independent evidence of OD link prediction quality;
* No arbitrary goodness-of-fit heuristics (such as absolute percentage error thresholds) or chi-square distribution tests are employed as substitutes for microscopic link-level evaluation.

#### Cross-model comparison framework
To assess whether the informational benefit of distance-binned calibration generalizes across diverse model families, three distinct predictor backbones are evaluated:
1. **Gravity-Informed Urban GNN** (Primary): Graph neural network combining edge-conditioned message passing with spatial geographic coordinates and gravity priors;
2. **Pairwise Spatial Node MLP** (Neural Baseline): Multi-layer perceptron operating strictly on concatenated origin and destination tract attributes and distance, without graph neighborhood convolutions;
3. **Classical Two-Parameter Gravity** (Parametric Baseline): Non-neural spatial interaction model $T_{ij} = \exp(G) P_i P_j d_{ij}^{-\alpha}$ fitted via log-linear ordinary least squares.

All three model families are subjected to the exact same cross-city evaluation protocol:
* Evaluated on the identical positive interzonal support $\Omega_{c,\mathrm{inter}}^+$ for each of the 50 test cities;
* Calibrated using the identical deterministic calibration operator ($q = 1.0$) with identical distance bin edges;
* Compared in terms of baseline CPC ($M_0$), calibrated CPC ($M_1$), and paired improvement ($\Delta\operatorname{CPC}$);
* Goodness-of-fit metrics such as AIC or BIC are not used for cross-model comparisons because the non-neural gravity model is estimated via OLS log-linear regression, whereas the neural backbones are optimized under a zero-truncated negative binomial likelihood;
* The objective of cross-model comparison is strictly to test whether target distance distributions provide marginal predictive gains across predictors with fundamentally different inductive biases, rather than to claim that calibration converts an inferior architecture into a superior one.

*(Tiếng Việt: **3.6.2. Thước đo đánh giá và so sánh mô hình**:
(1) **Thước đo chính CPC**: Common Part of Commuters (CPC) được tính trên tập hỗ trợ liên vùng dương $\Omega_{c,\mathrm{inter}}^+ = \{ (i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij}\geq1,\ i\neq j,\ d_{c,ij}>0 \}$. CPC đo tỷ lệ khối lượng lưu lượng chung giữa quan sát thực tế và cường độ dự báo, nằm trong đoạn $[0, 1]$. Toàn bộ self-flows ($i=j$) và các cặp có khoảng cách bằng 0 ($d \le 0$) bị loại bỏ nghiêm ngặt (`o_np != d_np` và `dist_km > 0.0`). Dự báo là các giá trị cường độ kỳ vọng liên tục (float), không làm tròn thành số nguyên. CPC tập trung đánh giá chất lượng tái tạo luồng di chuyển trên các liên kết đã biết, không đánh giá khả năng phân loại các cặp bằng 0. Dự báo baseline không được co giãn theo tổng lưu lượng ground-truth trước khi đánh giá, nhưng phép hiệu chỉnh bảo toàn tuyệt đối tổng lưu lượng dự báo của baseline.
(2) **Estimand cải thiện ghép cặp**: Mức thay đổi hiệu năng ghép cặp $\Delta_{c,s}^{(m)} = \operatorname{CPC}_{c,s}(M1_{\mathrm{city}}^{(m)}) - \operatorname{CPC}_{c,s}(M_0^{(m)})$ so sánh cùng thành phố, cùng kiến trúc $m$, cùng seed $s$, cùng tập hỗ trợ $\Omega^+$, và cùng dữ liệu mục tiêu. Urban GNN xác định estimand chính; MLP và Gravity cung cấp so sánh độ bền và không bị gộp chung vào một estimand duy nhất.
(3) **Các thước đo sai số phụ**: Sáu metric phụ được tính trên cùng tập hỗ trợ gồm MAE, RMSE, NRMSE, $\operatorname{RMSE}_{\log1p}$, tương quan hạng Spearman, và sai số tương đối tổng luồng $\operatorname{RelError}$. Chúng đóng vai trò kiểm tra độ nhạy để đảm bảo kết luận không phụ thuộc đơn lẻ vào CPC.
(4) **Chẩn đoán phân phối khoảng cách**: Đo mức độ khớp giữa phân phối khoảng cách gộp sau hiệu chỉnh và phân phối mục tiêu $\mathbf{Y}_{D,c}$. Đây là chẩn đoán cơ chế nội bộ nhằm kiểm tra thuật toán hiệu chỉnh có tái phân bổ khối lượng theo bin như thiết kế hay không, không phải thước đo độc lập về độ chính xác của các cặp OD.
(5) **Khung so sánh giữa các mô hình**: So sánh GNN, MLP và Classical Gravity trên cùng tập hỗ trợ, cùng giao thức đánh giá, báo cáo CPC baseline, CPC hiệu chỉnh và mức tăng ghép cặp. Không sử dụng AIC/BIC vì các mô hình có objective tối ưu hóa khác nhau (OLS vs ZTNB likelihood). Mục tiêu là kiểm tra xem phân phối khoảng cách có mang lại giá trị gia tăng biên trên các kiến trúc có inductive bias khác nhau hay không.)*

---

### 3.6.3 Statistical Analysis and Uncertainty Quantification
*(Tiếng Việt: **3.6.3. Phân tích thống kê và lượng hóa độ bất định**)*

#### City-level estimand and model-seed aggregation
To account for stochasticity in neural initialization and training optimization, each neural architecture is trained across three independent model seeds $\mathcal{S} = \{1, 10, 100\}$.

For each city $c$, baseline and calibrated predictions are evaluated paired within each model seed, and the paired performance differences are averaged across seeds:

$$\Delta_c = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \Delta_{c,s} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left[ \operatorname{CPC}_{c,s}(M1_{\mathrm{city}}) - \operatorname{CPC}_{c,s}(M_0) \right]$$

The primary population-level headline estimand is defined as the macro-average across all $C = 50$ benchmark metropolitan areas:

$$\overline{\Delta} = \frac{1}{C} \sum_{c=1}^{C} \Delta_c$$

This macro-averaging protocol guarantees that:
1. Each metropolitan area contributes exactly one unit of weight to the headline estimand, preventing megacities with hundreds of thousands of candidate OD pairs from dominating smaller urban regions;
2. Baseline and calibrated predictions are compared paired within the same city and identical model seed before aggregation;
3. The estimand measures the expected gain across heterogeneous cities, rather than an unweighted average pooled indiscriminately over millions of individual OD pairs.

#### Fold-stratified city-level nonparametric bootstrap
Uncertainty in the macro-average improvement $\overline{\Delta}$ is quantified using a fold-stratified city-level nonparametric bootstrap ($B = 10,000$ resamples, random seed fixed at 42 via `np.random.default_rng(42)`) [@efron1993bootstrap].

In each bootstrap replicate $b \in \{1, \dots, B\}$, resampling is performed with replacement within each of the 5 cross-validation fold strata (sampling 10 cities with replacement from the 10 evaluation cities of that fold):

$$\overline{\Delta}^{*(b)} = \frac{1}{C} \sum_{c \in \mathcal{C}^{*(b)}} \Delta_c, \qquad b = 1, \dots, B$$

The 95% confidence interval is constructed via the empirical percentile method:

$$\mathrm{CI}_{95\%} = \left[ Q_{0.025}\left(\overline{\Delta}^*\right), Q_{0.975}\left(\overline{\Delta}^*\right) \right]$$

Crucially:
* The resampling unit is strictly the metropolitan area ($C = 50$);
* Individual OD pairs within a city are never resampled, avoiding invalid independence assumptions among spatially clustered links;
* Baseline ($M_0$) and calibrated ($M_1$) results remain strictly paired within each resampled city;
* Neural networks are not re-fitted during bootstrap resampling; the procedure resamples the pre-computed city-level paired deltas;
* The bootstrap is fully nonparametric, requiring no distributional assumptions on flow volumes.

#### Paired Wilcoxon signed-rank test
To determine whether the directional gains represent a statistically significant shift rather than symmetric variation around zero, a two-sided paired Wilcoxon signed-rank test [@wilcoxon1945ranking] is conducted across the $N = 50$ city-level paired deltas $\{\Delta_c\}_{c=1}^{50}$.

The null hypothesis posited is:

$$H_0: \operatorname{median}(\Delta_c) = 0$$

against the two-sided alternative:

$$H_1: \operatorname{median}(\Delta_c) \neq 0$$

The test statistic is evaluated using `scipy.stats.wilcoxon` with default zero-handling (discarding zero differences). The test operates strictly on the 50 paired city observations. The proportion of improved cities (win rate, e.g., 45/50) is reported as a descriptive statistic and does not substitute for formal rank-based hypothesis testing.

#### Multiple-testing correction protocol
The primary benchmark evaluation tests a single pre-specified confirmatory hypothesis ($M_1$ versus $M_0$ at $K = 8$), requiring no multiple-testing penalty.

For secondary sensitivity investigations involving families of related hypotheses, the Holm-Bonferroni step-down procedure [@holm1979sequential] is applied to control the family-wise error rate (FWER) at level $\alpha = 0.05$:
1. **Distance resolution family**: Corrected across the set of 9 secondary bin resolutions ($K \in \{2, 4, 6, 10, 12, 14, 16, 18, 20\}$) compared against the locked $K = 8$ anchor;
2. **Noise robustness family**: Corrected across the set of synthetic perturbation levels $\epsilon \in \{0.00, \dots, 0.05\}$;
3. **Direct-OD sampling family**: Corrected across the tested revealed sampling fractions $p \in \{0.05\%, \dots, 1.0\%\}$.

Both unadjusted raw $p$-values and Holm-adjusted $p$-values are reported where multiple testing applies.

*(Tiếng Việt: **Phân tích thống kê và lượng hóa độ bất định**: (1) **Estimand cấp thành phố**: Với mỗi thành phố $c$, chênh lệch CPC ghép cặp được tính riêng cho từng seed rồi lấy trung bình qua 3 model seeds $\mathcal{S}=\{1,10,100\}$: $\Delta_c = \frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}} [\operatorname{CPC}_{c,s}(M1_{\mathrm{city}}) - \operatorname{CPC}_{c,s}(M_0)]$. Estimand chính là macro-average trên 50 thành phố: $\overline{\Delta} = \frac{1}{C}\sum_{c=1}^C \Delta_c$, đảm bảo mỗi thành phố đóng góp trọng số như nhau và các cặp luồng được so sánh ghép cặp; (2) **Nonparametric bootstrap cấp thành phố**: Độ bất định được lượng hóa bằng bootstrap phân tầng theo fold với $B = 10,000$ lần lấy mẫu có hoàn lại (seed ngẫu nhiên 42). Đơn vị lấy mẫu là toàn bộ thành phố, không lấy mẫu từng cặp OD, không coi các cặp OD là độc lập, không huấn luyện lại mô hình trong từng replicate, và khoảng tin cậy 95% được xác định bằng phương pháp percentile $[Q_{0.025}, Q_{0.975}]$; (3) **Kiểm định Wilcoxon signed-rank ghép cặp**: Thực hiện kiểm định hai phía trên 50 quan sát $\Delta_c$ với giả thuyết không $H_0: \operatorname{median}(\Delta_c) = 0$. Số thành phố cải thiện (win rate) là thống kê mô tả, không thay thế kiểm định thứ bậc; (4) **Hiệu chỉnh multiple testing**: Estimand chính không cần hiệu chỉnh; các họ giả thuyết phân tích độ nhạy (quét $K$, quét nhiễu $\epsilon$, quét mẫu trực tiếp $p$) được kiểm soát FWER ở mức $\alpha = 0.05$ bằng quy trình Holm-Bonferroni step-down.)*

---

### 3.6.4 Robustness and Diagnostic Experiments
*(Tiếng Việt: **3.6.4. Các thí nghiệm độ bền và chẩn đoán cơ chế**)*

Supplementary diagnostic experiments evaluate the operational boundaries, failure modes, and mechanistic drivers of distance-binned calibration (addressing RQ2). These stress tests isolate specific information channels without modifying the primary benchmark estimand.

#### Distance-bin resolution ($K$-Sensitivity)
The distance continuum partition is varied across ten granularities:

$$K \in \{2, 4, 6, 8, 10, 12, 14, 16, 18, 20\}$$

For each configuration $K$, bin edges are estimated strictly from training cities $\mathcal{C}_{\mathrm{train}}^{(f)}$ using empirical distance quantiles, and the target distribution $\mathbf{Y}_{D,c}^{(K)} \in \Delta^{K-1}$ is extracted. All other components—including frozen baseline predictions $\widehat{\mathbf{T}}_c^{(0)}$, evaluation support $\Omega_{c,\mathrm{inter}}^+$, and ground-truth flows $t_{c,ij}$—remain strictly identical. Distance intervals with zero predicted mass are handled via active support conditioning $\mathcal{A}_c^{(m)}$. Multiple comparisons against the primary anchor ($K = 8$) are adjusted via Holm-Bonferroni correction.

#### Observation-noise robustness
To determine tolerance to observation inaccuracy, synthetic perturbation is injected into the active target distribution $\mathbf{p}_{\mathrm{active}}$. Perturbations are generated continuously along a random Gaussian vector $\mathbf{z} \in \mathbb{R}^{K_{\mathrm{act}}}$ on the probability simplex using softmax reweighting:

$$\log p_b(\sigma) = \log p_{c,b} + \sigma z_b, \qquad p_b(\sigma) = \frac{\exp(\log p_b(\sigma))}{\sum_{r} \exp(\log p_r(\sigma))}$$

A root-finding bisection solver determines the exact scaling parameter $\sigma(\epsilon) > 0$ such that the Total Variation (TV) error satisfies:

$$\mathrm{TV}\left(\mathbf{p}(\sigma), \mathbf{p}_{\mathrm{active}}\right) = \frac{1}{2} \sum_{b=1}^{K_{\mathrm{act}}} \left| p_b(\sigma) - p_{c,b} \right| = \epsilon$$

Evaluations are conducted over a controlled grid of noise magnitudes:

$$\epsilon \in \{0.00, 0.01, 0.02, 0.03, 0.04, 0.05\}$$

This formulation guarantees that perturbed distributions remain valid probability vectors ($p_b \ge 0$, $\sum p_b = 1$) with exact TV displacement. Baseline predictions, evaluation supports, and model parameters are held strictly frozen. The observed empirical crossover threshold is interpreted as a characteristic of the benchmark rather than a universal physical constant.

#### Bin-order permutation
To verify that calibration exploits genuine spatial decay semantics rather than superficial variance stretching, the elements of the active target distribution $\mathbf{p}_{\mathrm{active}}$ are subjected to random permutations across distance bins while keeping distance cutoffs $[a_{b-1}, a_b)$ and baseline predictions unchanged:

$$\mathbf{Y}_{D,c}^{\mathrm{perm}} = \operatorname{Permute}\left(\mathbf{Y}_{D,c}\right)$$

This intervention preserves the exact multiset of bin probabilities and total probability mass ($\sum Y_D = 1$), but scrambles their spatial distance associations. This test is conceptually distinct from varying random seeds.

#### Donor-city placebos (Target specificity)
To test whether calibration benefits require target-specific travel patterns or merely generic smooth deterrence priors, target distributions are replaced with three placebo donor variants:
1. **Dose-Matched Donors**: For each target city $c$ and training donor $c' \in \mathcal{C}_{\mathrm{train}}^{(f)}$, log-odds adjustment ratios are rescaled so that the perturbation magnitude $D_D = \|\tilde{\mathbf{r}}_D\|_2$ matches the target's natural intervention dose $D_T = \|\tilde{\mathbf{r}}_T\|_2$;
2. **In-Fold Unadjusted Donors**: Randomly selected donor distributions from within the same training fold without dose adjustment;
3. **Fold Training-Mean Donor**: The empirical mean distribution $\bar{\mathbf{Y}}_{D,\mathrm{train}} = \frac{1}{|\mathcal{C}_{\mathrm{train}}^{(f)}|} \sum_{c' \in \mathcal{C}_{\mathrm{train}}^{(f)}} \mathbf{Y}_{D,c'}$ averaged across all 35 training cities of the fold.

In all donor arms, baseline predictions $\widehat{\mathbf{T}}_c^{(0)}$ and target support $\Omega_{c,\mathrm{inter}}^+$ remain frozen. Specificity gain is measured by $\Delta\operatorname{CPC}_c(Y_{D,c}) - \Delta\operatorname{CPC}_c(Y_{D,\mathrm{donor}})$.

#### Spatial-resolution analysis (County-level conditioning)
To investigate the effect of sub-metropolitan spatial resolution, city-wide calibration ($M1_{\mathrm{city}}$) is compared against origin-county conditioned calibration ($M1_{\mathrm{county}}$).

Across the 50 benchmark cities, 39 metropolitan areas map to a single county, where origin-county conditioning is mathematically identical to city-wide calibration ($\mathbf{Y}_{D,c,\ell} \equiv \mathbf{Y}_{D,c}$), yielding $\Delta\operatorname{CPC}_{\mathrm{res},c} = 0$ as an invariant algorithmic sanity check. The remaining 11 multi-county metropolitan areas (Kansas City, New York, Dallas, Denver, Omaha, Tulsa, Detroit, Chicago, Boston, Milwaukee, and Atlanta) contain tracts spanning 2 to 7 counties and provide the genuine test of spatial resolution refinement. In all cases, calibrated predictions are evaluated over the identical city-wide positive support $\Omega_{c,\mathrm{inter}}^+$.

#### Initialization and predictor robustness
To establish that calibration performance is not idiosyncratic to a single neural initialization or model family, experiments are replicated across:
1. **Three independent model initializations**: $\mathcal{S} = \{1, 10, 100\}$;
2. **Three distinct predictor families**: Gravity-Informed Urban GNN (primary), Pairwise Node MLP (non-graph neural baseline), and Two-Parameter Power-Law Gravity (non-neural parametric baseline).

The identical deterministic calibration operator is applied to the frozen baseline predictions of each predictor family without re-tuning.

*(Tiếng Việt: **Các thí nghiệm độ bền và chẩn đoán cơ chế**: (1) **Độ phân giải khoảng cách ($K$-Sensitivity)**: Khảo sát $K \in \{2, 4, \dots, 20\}$ với các biên khoảng cách ước lượng từ training cities, giữ nguyên baseline và hiệu chỉnh Holm cho 9 so sánh phụ với mốc $K=8$; (2) **Độ bền trước nhiễu quan sát**: Thêm nhiễu Total Variation $\epsilon \in [0\%, 5\%]$ vào $\mathbf{p}_{\mathrm{active}}$ dọc theo hướng Gaussian trên simplex, sử dụng nghiệm bisection để đạt chính xác khoảng cách TV; (3) **Hoán vị thứ tự bin**: Hoán vị ngẫu nhiên xác suất các khoảng của $Y_D$ trong khi giữ nguyên biên cự ly để kiểm tra ngữ nghĩa suy giảm khoảng cách; (4) **Donor-city placebos**: Thay thế phân phối mục tiêu bằng donor đã khớp liều lượng ($D_T$), donor ngẫu nhiên trong fold, và phân phối trung bình của tập huấn luyện; (5) **Phân tích độ phân giải không gian**: So sánh $M1_{\mathrm{city}}$ với $M1_{\mathrm{county}}$, trong đó 39 single-county cities đóng vai trò kiểm tra bất biến thuật toán ($\Delta\operatorname{CPC}_{\mathrm{res}}=0$) và 11 multi-county cities cung cấp kiểm định thực nghiệm thực sự; (6) **Độ bền theo seed và kiến trúc**: Đánh giá trên 3 seeds $\{1, 10, 100\}$ và 3 họ mô hình (GNN, MLP, Gravity) nhằm khẳng định tính ổn định tổng quát.)*
