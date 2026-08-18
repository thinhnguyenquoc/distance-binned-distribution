# Đặc Tả Giao Thức Phân Chia Fold, Validation và Dữ Liệu Kiểm Thử E1-v2

> **Giao thức**: E1-v2 (Oracle Aggregated-Distance Existence Test)  
> **Trạng thái Phương pháp luận**: **Amended replication under a locked pre-specified protocol**  
> **Phiên bản Manifest**: `e1-splits-v2` (`results/e1/splits_manifest_v2.json`)  
> **Mã băm toàn vẹn (SHA-256)**: `7f9afe02725c7798dab018b6a353ed99ceaf6c36a9f77316aa47ea21297ebd14`  
> **Nguyên tắc cốt lõi**: Khóa cứng 100% outer test folds từ E1-v1 (zero test perturbation), phân tầng kích thước chuẩn hóa (5 size strata), khóa cứng manifest có hash, và đối chứng đa donor (9-donor placebo) với estimand tính đặc hiệu chính xác trên đơn vị đô thị.

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

## 2. Quy Trình Khóa Cứng Outer Test Sets từ E1-v1

> [!IMPORTANT]
> **Nguyên tắc bất biến Outer Test Sets**:  
> Để loại bỏ hoàn toàn nguy cơ xáo trộn tập kiểm thử sau khi đã quan sát kết quả E1-v1 (post-hoc selection / tie-breaking shift), danh sách 10 test cities của mỗi fold được **sao chép trực tiếp và khóa bất biến từ manifest E1-v1**. Tuyệt đối không sinh lại outer test folds ngẫu nhiên.

Danh sách 10 test cities cố định cho 5 folds:
- **Fold 1**: `Arlington`, `Austin`, `El_Paso`, `Long_Beach`, `Memphis`, `Milwaukee`, `New_York`, `San_Diego`, `Seattle`, `Virginia_Beach`
- **Fold 2**: `Atlanta`, `Boston`, `Fort_Worth`, `Indianapolis`, `Los_Angeles`, `Mesa`, `Oklahoma_City`, `Raleigh`, `Sacramento`, `San_Antonio`
- **Fold 3**: `Baltimore`, `Chicago`, `Detroit`, `Fresno`, `Jacksonville`, `Las_Vegas`, `Louisville`, `Oakland`, `Tulsa`, `Washington_DC`
- **Fold 4**: `Colorado_Springs`, `Columbus`, `Houston`, `Minneapolis`, `Nashville`, `Omaha`, `Phoenix`, `Portland`, `San_Francisco`, `Tampa`
- **Fold 5**: `Albuquerque`, `Charlotte`, `Dallas`, `Denver`, `Kansas_City`, `Miami`, `Philadelphia`, `San_Jose`, `Tucson`, `Wichita`

---

## 3. Quy Trình Chọn Validation Set theo Phân Tầng Kích Thước (Stratified Validation)

Đối với $40$ đô thị non-test trong mỗi fold:

### 3.1 Thuật toán 5 Strata Kích thước
1. Sắp xếp 40 đô thị non-test tăng dần theo $(n_{\text{tracts}}, \text{city})$.
2. Chia thành 5 strata kích thước, mỗi stratum có đúng 8 đô thị:
   - `stratum_0_small`: 8 đô thị nhỏ nhất.
   - `stratum_1_small_med`: 8 đô thị nhỏ – trung bình.
   - `stratum_2_med`: 8 đô thị trung bình.
   - `stratum_3_med_large`: 8 đô thị trung bình – lớn.
   - `stratum_4_large`: 8 đô thị lớn nhất.
3. Khởi tạo bộ sinh số ngẫu nhiên với seed cố định định sẵn:
   $$\text{RNG} = \text{Random}(\text{seed} + \text{fold\_id}), \quad \text{với seed} = 20260818$$
4. Rút ngẫu nhiên đúng 1 đô thị từ mỗi stratum:
   $$\text{Val\_Cities} = [\text{RNG.choice}(\text{stratum}_k) \quad \forall k \in \{0, 1, 2, 3, 4\}]$$
5. 35 đô thị còn lại tạo thành tập **Train**.

### 3.2 Ghi nhận Danh Sách Ứng Viên (Candidate Audit Metadata)
Manifest lưu toàn bộ 8 ứng viên của từng stratum kèm cờ `selected_for_val: true/false`, đảm bảo bất kỳ ai đọc manifest cũng thấy rõ tính đại diện quy mô đô thị của tập validation.

---

## 4. Cấu Trúc Khóa Manifest v2 (`results/e1/splits_manifest_v2.json`)

```json
{
  "version": "e1-splits-v2",
  "protocol_status": "amended replication under a locked protocol",
  "outer_split_source": "locked from E1-v1 outer test sets (zero test perturbation)",
  "validation_selection_rule": "five tract-count strata (8 cities each), fixed-seed selection (1 per stratum)",
  "validation_seed": 20260818,
  "manifest_sha256": "7f9afe02725c7798dab018b6a353ed99ceaf6c36a9f77316aa47ea21297ebd14",
  "folds": {
    "1": {
      "train": [ "Albuquerque", "Atlanta", "..." ],
      "val": [ "Chicago", "Portland", "Sacramento", "San_Francisco", "Tulsa" ],
      "test": [ "Arlington", "Austin", "El_Paso", "Long_Beach", "Memphis", "Milwaukee", "New_York", "San_Diego", "Seattle", "Virginia_Beach" ],
      "validation_candidates_by_stratum": {
        "stratum_0_small": [ ...8 cities... ],
        "stratum_1_small_med": [ ...8 cities... ],
        "stratum_2_med": [ ...8 cities... ],
        "stratum_3_med_large": [ ...8 cities... ],
        "stratum_4_large": [ ...8 cities... ]
      }
    }
  }
}
```

### Các Bất Biến Kiểm Tra tại Runtime (Assertion Gates)
- $|\text{Train}|=35, |\text{Val}|=5, |\text{Test}|=10$.
- Không có phần tử trùng lặp trong bất kỳ tập hợp nào.
- $\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$.
- $\text{Train} \cup \text{Val} \cup \text{Test} = \mathcal{C}_{50}$.
- $\text{Test}_f$ khớp chính xác với `LOCKED_V1_TEST_FOLDS[f]`.
- Mỗi đô thị được kiểm thử đúng 1 lần trên 5 folds: $\sum_{f=1}^5 \mathbb{I}(c \in \text{Test}_f) = 1$.

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

## 6. Xử Lý Test Data, 9 Wrong Donors & Estimand Tính Đặc Hiệu

Với mỗi đô thị $c \in \text{Test}_f$:

### 6.1 Các Điều Kiện Thí Nghiệm
1. **Condition A (Zero-Shot Baseline $M_0$)**: Suy luận $\hat{T}_c^{(0)} = \mathbb{E}[T \mid T \ge 1]$, tính $\text{CPC}_0$.
2. **Condition B (Treatment $+Y_{D,c}^{\text{target}}$)**: Hiệu chuẩn $T_c^{(\text{YD})} = \text{calibrate\_kbins}(\hat{T}_c^{(0)}, D_c, Y_{D,c}^{\text{GT},+})$, tính $\Delta_c^{\text{target}} = \text{CPC}_c^{\text{target}} - \text{CPC}_0$.
3. **Condition C (Multi-Donor Placebo)**: Hiệu chuẩn lần lượt với toàn bộ 9 wrong donors còn lại trong fold $\mathcal{D}_c^{\text{wrong}} = \text{Test}_f \setminus \{c\}$.
   $$\overline{\Delta}_c^{\text{wrong}} = \frac{1}{9}\sum_{d \in \mathcal{D}_c^{\text{wrong}}} \Delta_{c,d}^{\text{wrong}}$$

### 6.2 Estimand Tính Đặc Hiệu (Specificity Estimand)
$$\Delta_c^{\text{specificity}} = \Delta_c^{\text{target}} - \overline{\Delta}_c^{\text{wrong}}$$

### 6.3 Nguyên Tắc Thống Kê
- **Đơn vị phân tích**: Nghiêm ngặt là **Đô thị ($N=40$ Confirmatory, $N=50$ Full Coverage)**. Không xem 9 wrong donors là các mẫu quan sát độc lập để tránh lạm dụng bậc tự do (pseudoreplication).
- **Chỉ số báo cáo chính**:
  1. Mean, Median, IQR (Q3 - Q1), SD (ddof=1) cho $\Delta_c^{\text{target}}$, $\overline{\Delta}_c^{\text{wrong}}$, và $\Delta_c^{\text{specificity}}$.
  2. 95% Fold-Stratified Bootstrap CI cho $\Delta_c^{\text{specificity}}$ và $\Delta_c^{\text{target}}$.
  3. Paired Wilcoxon Signed-Rank Test trực tiếp trên vector $(\Delta_c^{\text{specificity}})$.
  4. Specificity Win Rate: Tỷ lệ đô thị có $\Delta_c^{\text{specificity}} > 0$.
