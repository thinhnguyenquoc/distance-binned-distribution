# Tài Liệu Đặc Tả Phân Chia Fold, Validation và Dữ Liệu Kiểm Thử E1-v2

> **Giao thức**: E1-v2 (Oracle Aggregated-Distance Existence Test)  
> **Phiên bản Manifest**: `e1-splits-v2` (`results/e1/splits_manifest_v2.json`)  
> **Nguyên tắc cốt lõi**: Không rò rỉ dữ liệu (zero leakage), phân tầng kích thước chuẩn hóa (size-stratified), khóa cứng manifest trước khi chạy (manifest locking), và đối chứng đa donor (9-donor placebo).

---

## 1. Tổng Quan Kiến Trúc Phân Chia Dữ Liệu (50 Đô Thị)

Toàn bộ 50 đô thị tại Hoa Kỳ được phân bổ theo mô hình **5-Fold Stratified Cross-Validation**, đảm bảo:
- Mỗi fold có đúng:
  - **35 đô thị Huấn luyện (Train)**: Dùng để fit Zero-Shot Backbone GNN và tính $K_{\text{move}}=8$ quantile bin edges.
  - **5 đô thị Đánh giá (Validation)**: Dùng cho early stopping và model selection (đo Interzonal CPC sau mỗi epoch).
  - **10 đô thị Kiểm thử (Held-out Test)**: Đánh giá out-of-fold cuối cùng cho 3 điều kiện (Baseline M0, Treatment $+Y_D^{\text{target}}$, Multi-donor Placebo $+Y_D^{\text{wrong}}$).
- **Phân định vai trò thống kê**:
  - **Folds 2–5 ($n=40$ đô thị)**: Nhóm kiểm định chính thức (**Primary Confirmatory Benchmark**).
  - **Fold 1 ($n=10$ đô thị)**: Nhóm phát triển / khám phá (**Exploratory Development Fold**).
  - **Toàn bộ 50 đô thị**: Cung cấp bức tranh toàn cảnh (**Descriptive Out-of-Fold Coverage**).

---

## 2. Quy Trình Tạo 5 Outer Test Folds (Snake Stratification)

### 2.1 Sắp xếp và Tie-Breaking
1. Thu thập số lượng census tract ($n_{\text{tracts}}$) của tất cả 50 đô thị từ file `meta.csv` của từng thành phố.
2. Sắp xếp 50 đô thị tăng dần theo cặp khóa:
   $$\text{Key} = (n_{\text{tracts}}, \text{city\_name})$$
   *Việc thêm tên thành phố làm tie-breaker đảm bảo thứ tự sắp xếp hoàn toàn xác định trên mọi hệ điều hành.*

### 2.2 Phân bổ 10 Strata theo hình thức Snake (Ziczac)
- 50 thành phố được chia thành 10 nhóm kích thước (strata), mỗi stratum có 5 thành phố:
  - Stratum 0: 5 thành phố nhỏ nhất.
  - ...
  - Stratum 9: 5 thành phố lớn nhất.
- Phân bổ vào 5 Fold theo nguyên tắc ziczac:
  - Stratum chẵn ($0, 2, 4, 6, 8$): Gán theo chiều thuận (thành phố $i$ vào Fold $i+1$).
  - Stratum lẻ ($1, 3, 5, 7, 9$): Đảo ngược danh sách trước khi gán (thành phố $i$ vào Fold $5 - i$).
- **Kết quả**: Mỗi fold nhận đúng 10 test cities phân bố đều từ nhỏ đến lớn, giữ nguyên trọn vẹn 10 test cities của E1-v1 để phục vụ so sánh đối chuẩn.

---

## 3. Quy Trình Tạo Validation Set theo Phân Tầng Kích Thước (Stratified Validation)

Khác với E1-v1 (chọn 5 thành phố validation cuối theo bảng chữ cái), E1-v2 áp dụng **Size Stratification**:

### 3.1 Thuật toán chọn Validation
Đối với 40 thành phố non-test của mỗi fold:
1. Lấy danh sách 40 thành phố và sắp xếp theo $(n_{\text{tracts}}, \text{city})$.
2. Chia 40 thành phố thành 5 strata kích thước, mỗi stratum có đúng 8 thành phố:
   - **Stratum 1 (Nhỏ)**: `ordered[0:8]`
   - **Stratum 2 (Nhỏ – Trung bình)**: `ordered[8:16]`
   - **Stratum 3 (Trung bình)**: `ordered[16:24]`
   - **Stratum 4 (Trung bình – Lớn)**: `ordered[24:32]`
   - **Stratum 5 (Lớn)**: `ordered[32:40]`
3. Khởi tạo bộ sinh số ngẫu nhiên độc lập với seed cố định:
   $$\text{RNG} = \text{Random}(\text{seed} + \text{fold\_id}), \quad \text{với seed} = 20260818$$
4. Chọn ngẫu nhiên đúng 1 thành phố từ mỗi stratum:
   $$\text{Val\_Cities} = [\text{RNG.choice}(\text{stratum}_k) \quad \forall k \in \{0, 1, 2, 3, 4\}]$$
5. 35 thành phố còn lại được đưa vào tập **Train**.

> [!NOTE]
> Quy trình này hoàn toàn độc lập với luồng di chuyển (OD trips), phân bố khoảng cách ($Y_D$), và CPC kết quả. Validation set luôn đại diện đầy đủ 5 phân khúc quy mô đô thị.

---

## 4. Khóa Cứng Manifest (`results/e1/splits_manifest_v2.json`)

Toàn bộ cấu trúc phân chia được sinh trước một lần và lưu cố định tại `results/e1/splits_manifest_v2.json`. Mã thực nghiệm `run_e1.py` chỉ đọc manifest từ đĩa, tuyệt đối không tự sinh split động.

### 4.1 Định dạng Manifest
```json
{
  "version": "e1-splits-v2",
  "outer_split_rule": "tract-count snake stratification",
  "validation_rule": "five tract-count strata, fixed-seed selection",
  "seed": 20260818,
  "folds": {
    "1": {
      "train": [ "Albuquerque", "Atlanta", "..." ],
      "val": [ "Chicago", "Portland", "Sacramento", "San_Francisco", "Tulsa" ],
      "test": [ "Arlington", "Austin", "El_Paso", "Long_Beach", "Memphis", "Milwaukee", "New_York", "San_Diego", "Seattle", "Virginia_Beach" ]
    },
    ...
  }
}
```

### 4.2 Các Ràng Buộc Bất Biến (Assertions)
Khi nạp manifest, hệ thống kiểm tra bắt buộc:
1. Kích thước tập: $|\text{Train}|=35$, $|\text{Val}|=5$, $|\text{Test}|=10$.
2. Tính rời rạc:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
3. Phủ đầy đủ:
   $$\text{Train} \cup \text{Val} \cup \text{Test} = \mathcal{C}_{\text{all}} \quad (|\mathcal{C}_{\text{all}}| = 50)$$
4. Phân hoạch Test toàn cục: Mỗi đô thị trong 50 đô thị xuất hiện làm test đúng 1 lần duy nhất trong toàn bộ 5 fold:
   $$\sum_{f=1}^5 \mathbb{I}(c \in \text{Test}_f) = 1 \quad \forall c \in \mathcal{C}_{\text{all}}$$

---

## 5. Danh Sách Chi Tiết Các Folds trong Manifest v2

| Fold | Role | Validation Cities (5) | Test Cities (10) |
|---|---|---|---|
| **Fold 1** | Exploratory | `Chicago`, `Portland`, `Sacramento`, `San_Francisco`, `Tulsa` | `Arlington`, `Austin`, `El_Paso`, `Long_Beach`, `Memphis`, `Milwaukee`, `New_York`, `San_Diego`, `Seattle`, `Virginia_Beach` |
| **Fold 2** | Confirmatory | `Albuquerque`, `Baltimore`, `Jacksonville`, `New_York`, `Virginia_Beach` | `Atlanta`, `Boston`, `Fort_Worth`, `Indianapolis`, `Los_Angeles`, `Mesa`, `Oklahoma_City`, `Raleigh`, `Sacramento`, `San_Antonio` |
| **Fold 3** | Confirmatory | `Dallas`, `El_Paso`, `Kansas_City`, `Milwaukee`, `Raleigh` | `Baltimore`, `Chicago`, `Detroit`, `Fresno`, `Jacksonville`, `Las_Vegas`, `Louisville`, `Oakland`, `Tulsa`, `Washington_DC` |
| **Fold 4** | Confirmatory | `Las_Vegas`, `Oakland`, `Raleigh`, `San_Diego`, `Washington_DC` | `Colorado_Springs`, `Columbus`, `Houston`, `Minneapolis`, `Nashville`, `Omaha`, `Phoenix`, `Portland`, `San_Francisco`, `Tampa` |
| **Fold 5** | Confirmatory | `Houston`, `Memphis`, `Nashville`, `Oakland`, `Virginia_Beach` | `Albuquerque`, `Charlotte`, `Dallas`, `Denver`, `Kansas_City`, `Miami`, `Philadelphia`, `San_Jose`, `Tucson`, `Wichita` |

---

## 6. Xử Lý Test Data & Thiết Kế Thí Nghiệm Đối Chứng (9-Donor Placebo)

Với mỗi đô thị $c$ trong tập Test của Fold ($\text{Test}_f$, $|\text{Test}_f| = 10$), thực nghiệm đánh giá 3 điều kiện:

### Điều kiện A: Zero-Shot Baseline ($M_0$)
- Nạp đặc trưng đô thị $(X_c, G_c^{\text{urban}}, D_c)$.
- Chạy suy luận qua mô hình đóng băng $\Theta^*$:
  $$\hat{T}_c^{(0)} = \mathbb{E}[T \mid T \ge 1]$$
- Tính $\text{CPC}_0$ trên tập liên vùng $\Omega_c^+ = \{(i,j) \in \Omega_c : i \neq j, D_{ij} > 0\}$.

### Điều kiện B: Treatment ($+ Y_{D,c}^{\text{target}}$)
- Trích xuất biểu đồ khoảng cách $K=8$ bins từ ground-truth của chính target city:
  $$Y_{D,c}^{\text{GT},+} = \text{extract\_yd\_kbins}(D_c, T_c^{\text{GT}}, \text{bin\_edges}_f)$$
- Hiệu chuẩn đóng kín (closed-form calibration với $q=1.0$):
  $$T_c^{(\text{YD})} = \text{calibrate\_kbins}(\hat{T}_c^{(0)}, D_c, Y_{D,c}^{\text{GT},+}, \text{bin\_edges}_f)$$
- Tính $\Delta_c^{\text{target}} = \text{CPC}(T_c^{(\text{YD})}, T_c^{\text{GT}}) - \text{CPC}_0$.

### Điều kiện C: Multi-Donor Placebo Control (Toàn bộ 9 Wrong Donors)
- Thay vì lấy 1 donor tùy ý, giao thức E1-v2 duyệt qua **toàn bộ 9 đô thị kiểm thử còn lại** trong cùng fold:
  $$\mathcal{D}_c^{\text{wrong}} = \text{Test}_f \setminus \{c\}, \quad |\mathcal{D}_c^{\text{wrong}}| = 9$$
- Với mỗi wrong donor $d \in \mathcal{D}_c^{\text{wrong}}$:
  - Trích xuất $Y_{D,d}^{\text{GT},+}$ từ đô thị $d$.
  - Hiệu chuẩn dự báo của target city $c$ bằng $Y_{D,d}^{\text{GT},+}$ $\rightarrow T_{c,d}^{(\text{wrong})}$.
  - Tính $\Delta_{c,d}^{\text{wrong}} = \text{CPC}(T_{c,d}^{(\text{wrong})}, T_c^{\text{GT}}) - \text{CPC}_0$.
- Lấy giá trị trung bình cộng làm chỉ số Placebo chính thức của đô thị $c$:
  $$\overline{\Delta}_c^{\text{wrong}} = \frac{1}{9} \sum_{d \in \mathcal{D}_c^{\text{wrong}}} \Delta_{c,d}^{\text{wrong}}$$
  $$\overline{\text{CPC}}_c^{\text{wrong}} = \frac{1}{9} \sum_{d \in \mathcal{D}_c^{\text{wrong}}} \text{CPC}_{c,d}^{\text{wrong}}$$
- Chi tiết kết quả của từng donor trong số 9 donors được ghi lại vào trường `wrong_donor_breakdown` của file kết quả `e1_per_city_results.json` để đảm bảo tính minh bạch và khả năng tái lập 100%.
