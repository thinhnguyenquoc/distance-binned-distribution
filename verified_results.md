# MASTER VERIFIED RESULTS AUDIT — SINGLE SOURCE OF TRUTH

> **Status**: 100% Verified against frozen raw result artifacts and code implementations.  
> **Master Research Contract Status**: **55/55 PASS** (execution time: 48.82s).  
> **Repository Path**: `d:\DBD\distance-binned-distribution`  
> **Date**: 2026-08-27  

---

## TỔNG KẾT THEO MỨC ƯU TIÊN (PRIORITY OVERVIEW)

| Mức ưu tiên | Nhóm hạng mục kiểm toán | Tình trạng | Kết luận chính |
|---|---|---|---|
| **P0 (Bắt buộc trước viết Results)** | 1. Main 50 cities, 2. Bootstrap CI, 3. Wilcoxon test, 10. Support audit, 11. Split audit, 12. Hyperparameter leakage | **100% PASS** | 50 cities, $\Delta\text{CPC} = +0.00354$ ($p=1.93\times 10^{-9}$), 95% CI $[+0.0026, +0.0045]$, 45/50 wins (90.0%), support và split hoàn toàn sạch |
| **P1 (Bắt buộc trước submission)** | 4. Seed robustness, 5. K sensitivity, 6. Noise robustness, 7. Placebo tests, 8. Backbone robustness | **100% PASS** | Bền vững qua 3 seeds (SD=0.0007); K=2..20 đơn điệu tăng; noise breakdown tại 4.44% TV; Specificity 50/50 ($p=8.88\times 10^{-16}$); xác nhận cả GNN & MLP |
| **P2 (Tăng sức mạnh bài báo)** | 9. Direct-OD / Calibration equivalence, 14. Mechanism diagnostic, 15. Secondary metrics (MAE/RMSE) | **100% PASS** | Rank invariance = 0.0; Correlation $r=0.80$ giữa $d_{\text{pre}}$ và $\Delta\text{CPC}$; $\Delta\text{MAE} = -2.54$ (90% win), $\Delta\text{RMSE} = -2.98$ (64% win) |

---

## 1. KẾT QUẢ CHÍNH 50 CITIES (P0)

* **Source file**: `results/5fold_results.json`, `results/e1_canonical_specificity_v2/e1_specificity_results.json`, `results/e1_canonical_specificity_v2/tables/e1_main_table.md`
* **Field / Function nguồn**: `rq1_delta_r.city`, `summary`, `src.training.evaluate.compute_cpc_pair` trên $\Omega_c^+$
* **Protocol check**: **PASS**

| Chỉ số / Trường | Giá trị chính xác (Exact Value) | Làm tròn báo cáo | Ghi chú & Đối chiếu |
|---|---|---|---|
| `n_cities` | `50` | `50` | 5 folds $\times$ 10 held-out test cities |
| Mean $\Delta\text{CPC}$ | `0.0035394915503731395` | `+0.00354` | Model seeds {1, 10, 100} averaged per city |
| Median $\Delta\text{CPC}$ | `0.0019531085392572867` | `+0.00195` | Phân phối lệch phải, trung vị dương rõ rệt |
| Min $\Delta\text{CPC}$ | `-0.0028416339182815165` | `-0.00284` | Thành phố El Paso (Fold 1) |
| Max $\Delta\text{CPC}$ | `0.015430743907011824` | `+0.01543` | Thành phố Los Angeles (Fold 2) |
| Win / Tie / Loss | `45 / 0 / 5` | `45 / 0 / 5` | 45 cải thiện, 0 hòa, 5 suy giảm |
| Win rate | `90.0%` (`45/50`) | `90.0%` | Tỷ lệ thắng áp đảo trên đơn vị đô thị |
| Mean CPC của `M0` (Baseline) | `0.7128072948832009` | `0.71281 ± 0.04434` | Zero-shot GNN baseline trước hiệu chỉnh |
| Mean CPC của `M1` (Calibrated) | `0.7163467864335741` | `0.71635 ± 0.04454` | Sau khi hiệu chỉnh với target $Y_D$ |
| Support đánh giá | $\Omega_c^+ = \{(i,j) : i \ne j, D_{ij} > 0\}$ | $\Omega_c^+$ | Chỉ tính trên các cặp liên vùng quan sát dương |

### Danh sách các thành phố không cải thiện ($\Delta\text{CPC} \le 0$)
Trong kết quả seed-averaged chính thức (Urban GNN, $K=8$), có **chính xác 5 thành phố** không cải thiện (tất cả đều có $\Delta\text{CPC} < 0$, không có thành phố nào tie):

1. **El_Paso** (Fold 1): $\Delta\text{CPC} = -0.002842$ ($M_0 = 0.663658, M_1 = 0.660817$)
2. **Oklahoma_City** (Fold 2): $\Delta\text{CPC} = -0.002610$ ($M_0 = 0.685055, M_1 = 0.682446$)
3. **Jacksonville** (Fold 3): $\Delta\text{CPC} = -0.001552$ ($M_0 = 0.666827, M_1 = 0.665275$)
4. **Louisville** (Fold 3): $\Delta\text{CPC} = -0.000215$ ($M_0 = 0.766276, M_1 = 0.766061$)
5. **Long_Beach** (Fold 1): $\Delta\text{CPC} = -0.000103$ ($M_0 = 0.741666, M_1 = 0.741563$)

> [!NOTE]
> **Giải thích nguồn gốc con số "3 thành phố" trong trao đổi trước đây**:
> - Với mô hình **MLP backbone** (`results/mlp_backbone_results.json`), win rate đạt **47/50 (94.0%)**, và danh sách không cải thiện có **đúng 3 thành phố**: `['El_Paso', 'Oklahoma_City', 'Louisville']`.
> - Với mô hình Urban GNN khi thử nghiệm ở $K=18$, hai thành phố biên là `Long_Beach` và `Louisville` (vốn chỉ âm nhẹ $-0.0001$ đến $-0.0002$) chuyển sang dương, cũng để lại đúng 3 thành phố suy giảm là `['El_Paso', 'Oklahoma_City', 'Jacksonville']`.
> - Trong báo cáo main paper theo canonical GNN ($K=8$), số lượng chính xác là **5 thành phố** (45/50).

---

## 2. BOOTSTRAP 95% CONFIDENCE INTERVAL (P0)

* **Source file**: `src/experiment/compute_delta_r.py` (`_fold_stratified_bootstrap`), `src/experiment/e1_core.py` (`fold_bootstrap`), `results/5fold_results.json`
* **Field / Function nguồn**: `rq1_delta_r.city.delta_cpc_inter.ci_95_lower` / `upper`
* **Protocol check**: **PASS** (Gate 8 passed)

| Thuộc tính Bootstrap | Giá trị chính xác | Ghi chú kỹ thuật |
|---|---|---|
| Bootstrap Random Seed | `42` (trong `compute_delta_r.py`) / `2024` (trong `e1_core.py`) | Seed cố định, determinism 100% |
| Số lượng Resamples ($B$) | `10,000` | $B=10,000$ phân tầng theo fold |
| Phương pháp tính CI | **Percentile Bootstrap** | `np.percentile(boot_means, [2.5, 97.5])` |
| 95% CI chính xác (Seed 42) | `[0.002611396866843778, 0.004511227050375786]` | Làm tròn: `[+0.0026, +0.0045]` |
| 95% CI chính xác (Seed 2024) | `[0.002606763556788414, 0.004482720290926319]` | Làm tròn: `[+0.0026, +0.0045]` |
| Empirical Mean | `0.00353949155` | Mean của 50 đô thị quan sát thực |
| Bootstrap Resample Mean | `0.00354011` | Độ lệch $< 6 \times 10^{-7}$, khớp hoàn toàn |
| Đơn vị lấy mẫu lại (Resampling Unit) | **CITY** ($N=50$) | Lấy mẫu lại các đô thị trong cùng fold với replacement (`fd = values[fold_ids == f]`); **TUYỆT ĐỐI KHÔNG LẤY MẪU OD PAIR**. |

---

## 3. WILCOXON PAIRED TEST (P0)

* **Source file**: `results/5fold_results.json`, `src/experiment/compute_delta_r.py`, `src/training/evaluate.py`
* **Field / Function nguồn**: `scipy.stats.wilcoxon(deltas, alternative="two-sided")`
* **Protocol check**: **PASS**

| Thông số kiểm định | Giá trị chính xác | Ghi chú đối chiếu |
|---|---|---|
| Statistic ($W$) | `83.0` | $W = \min(W^+, W^-) = W^- = 83.0$; $W^+ = 1192.0$ |
| Exact Two-Sided p-value | `1.9326371614170057e-09` | **$p = 1.93 \times 10^{-9}$** (bác bỏ $H_0$ cực mạnh) |
| Diagnostic One-Sided p-value | `9.663185807085029e-10` | $p = 9.66 \times 10^{-10}$ (chỉ dùng đối chiếu) |
| Số cặp hữu hiệu ($n$) | `50` | Đầy đủ 50 cặp, không có cặp nào bị loại |
| Số lượng Zero Differences | `0` | Không có cặp nào có $\Delta\text{CPC} = 0.0$ |
| Scipy Settings | `zero_method="wilcox"`, `correction=False` | Mặc định chuẩn của SciPy khi không có zero difference |
| Effect Size ($r_{\text{rb}}$) | `0.8698039215686274` | **$r_{\text{rb}} \approx 0.870$** (Matched-pairs rank-biserial correlation, large effect size) |

Công thức effect size:
$$r_{\text{rb}} = \frac{W^+ - W^-}{W^+ + W^-} = \frac{1192.0 - 83.0}{1275.0} = 0.8698$$

---

## 4. SEED ROBUSTNESS (SEEDS 1, 10, 100) (P1)

* **Source file**: `results/e1_canonical_specificity_v2/e1_specificity_results.json` (`per_city_per_seed`), `seed.md`
* **Field / Function nguồn**: Bảng `per_city_per_seed` lọc theo `model_seed`
* **Protocol check**: **PASS** (Gate 2, Gate 24 passed)

| Model Seed | $n$ | Mean $\Delta\text{CPC}$ | Median | 95% Fold-Stratified CI | Win Rate | Pos / Neg | Baseline $M_0$ | Calibrated $M_1$ |
|---|---:|---|---|---|---|---|---|---|
| **Seed 1** | 50 | `+0.004344` | `+0.002074` | `[+0.003224, +0.005473]` | **82.0%** (41/50) | 41 / 9 | 0.708606 | 0.712950 |
| **Seed 10** | 50 | `+0.003077` | `+0.001824` | `[+0.002164, +0.004037]` | **88.0%** (44/50) | 44 / 6 | 0.714774 | 0.717850 |
| **Seed 100** | 50 | `+0.003198` | `+0.002167` | `[+0.002355, +0.004078]` | **88.0%** (44/50) | 44 / 6 | 0.715042 | 0.718240 |
| **Seed-Averaged** | 50 | `+0.003539` | `+0.001953` | `[+0.002611, +0.004511]` | **90.0%** (45/50) | 45 / 5 | 0.712807 | 0.716347 |

* **Overall SD across seeds**:
  - Độ lệch chuẩn của mean $\Delta\text{CPC}$ giữa 3 seeds: **`0.000699`** ($\approx 0.0007$).
  - Mean per-city SD giữa các seeds: **`0.001264`**.
* **Đồng nhất tập dữ liệu**: Cả 3 seeds đều chạy trên cùng một partition 50 cities (`splits_manifest_v2.json`, SHA256: `7f9afe02725c7798dab018b6a353ed99ceaf6c36a9f77316aa47ea21297ebd14`), cùng tập observed support $\Omega_c^+$.

---

## 5. K SENSITIVITY (P1)

* **Source file**: `results/k_sensitivity_v1/k_sensitivity_summary.json`, `results/k_sensitivity_v1/k_sensitivity_per_city.csv`
* **Field / Function nguồn**: `summary`, `contrasts`, `src.experiment.run_k_sensitivity_v1`
* **Protocol check**: **PASS** (Gate 12 passed)

### Kết quả theo từng cấp phân giải khoảng cách ($K$)

| $K$ bins | Mean $\Delta\text{CPC}$ | Median | 95% Bootstrap CI | Win Rate | $K_{\text{active}}$ | Paired Diff vs $K=8$ | Raw $p$-value vs $K=8$ | Holm-adj $p$ (4 tests) | Holm-adj $p$ (9 tests) |
|---|---|---|---|---|---|---|---|---|---|
| **$K=2$** | `+0.000976` | `+0.000335` | `[+0.000520, +0.001513]` | 78.0% (39/50) | 2.0 | `-0.002564` | `3.49e-10` | `1.05e-09` | `1.05e-09` |
| **$K=4$** | `+0.001976` | `+0.000877` | `[+0.001251, +0.002789]` | 78.0% (39/50) | 4.0 | `-0.001564` | `1.77e-11` | `5.31e-11` | `7.07e-11` |
| **$K=8$ (Anchor)**| `+0.003539` | `+0.001953` | `[+0.002621, +0.004474]` | 90.0% (45/50) | 8.0 | `0.000000` (ref) | — | — | — |
| **$K=12$** | `+0.004796` | `+0.002882` | `[+0.003724, +0.005897]` | 92.0% (46/50) | 12.0 | `+0.001256` | `1.95e-13` | `5.86e-13` | `1.17e-12` |
| **$K=16$** | `+0.005741` | `+0.004326` | `[+0.004545, +0.006943]` | 92.0% (46/50) | 16.0 | `+0.002202` | `1.78e-14` | `7.11e-14` | `1.60e-13` |

*Ghi chú bổ sung*: Lưới đầy đủ 10 mức trong file gốc ($K \in \{2,4,6,8,10,12,14,16,18,20\}$) đều cho xu hướng tăng đơn điệu theo phân giải khoảng cách ($K=6$: $+0.00289$, $K=10$: $+0.00413$, $K=14$: $+0.00538$, $K=18$: $+0.00603$, $K=20$: $+0.00639$).
* **Xác nhận số comparisons**: Đối với lưới báo cáo con 5 điểm ($K=2, 4, 8, 12, 16$), số phép so sánh đối đầu với anchor $K=8$ chính xác là **4 comparisons**. Nếu kiểm toán toàn bộ lưới 10 điểm của runner thì là 9 comparisons. Cả hai mức điều chỉnh Holm đều đạt $p < 10^{-8}$.

---

## 6. NOISE ROBUSTNESS (P1)

* **Source file**: `results/noise_robustness_fine_v1/noise_summary.json`, `results/noise_robustness_fine_v1/noise_summary.md`
* **Field / Function nguồn**: `src.experiment.run_noise_robustness.generate_nested_noisy_yd`, `results_by_eps`
* **Protocol check**: **PASS** (Gate 10 passed)

### Kết quả thực nghiệm 50 thành phố với độ nhiễu phân rã mịn (Total Variation Noise)

| Mức nhiễu TV ($\epsilon$) | Mean $M_1$ CPC | Mean $\Delta\text{CPC}$ | 95% Bootstrap CI | Pos Cities | Harm Rate | Relative vs Clean | Benefit $p$-val (vs $M_0$) | Degrad $p$-val (vs Clean) |
|---|---|---|---|---|---|---|---|---|
| **0.0% ($\epsilon=0.00$)** | 0.71635 | `+0.00354` | `[+0.00261, +0.00451]` | 45/50 | 10.0% | **100.0%** | `9.66e-10` | — |
| **1.0% ($\epsilon=0.01$)** | 0.71617 | `+0.00336` | `[+0.00243, +0.00432]` | 44/50 | 12.0% | **94.9%** | `2.44e-08` | `4.44e-15` |
| **2.0% ($\epsilon=0.02$)** | 0.71563 | `+0.00282` | `[+0.00189, +0.00379]` | 36/50 | 28.0% | **79.7%** | `1.35e-05` | `4.44e-15` |
| **3.0% ($\epsilon=0.03$)** | 0.71474 | `+0.00193` | `[+0.00100, +0.00290]` | 28/50 | 44.0% | **54.6%** | `0.0446` | `4.44e-15` |
| **4.0% ($\epsilon=0.04$)** | 0.71351 | `+0.00070` | `[-0.00025, +0.00167]` | 18/50 | 64.0% | **19.7%** | `0.9695` | `4.44e-15` |
| **5.0% ($\epsilon=0.05$)** | 0.71193 | `-0.00087` | `[-0.00183, +0.00012]` | 17/50 | 66.0% | **-24.7%** | `0.9696` | `4.44e-15` |

* **Điểm phá vỡ tín hiệu (Crossover Threshold $\epsilon_{\text{cross}}$)**:
  - Qua **1,000 hướng nhiễu độc lập**: $\epsilon_{\text{cross}} = \mathbf{4.44\%} \ [95\%\text{ CI}: 4.16\%, 4.77\%]$.
  - Qua **10,000 lần bootstrap resampling đô thị**: $\epsilon_{\text{cross}} = \mathbf{4.39\%} \ [95\%\text{ CI}: 3.66\%, 4.94\%]$.
  - Chi tiết tại `results/noise_robustness_fine_v1/noise_crossover_uncertainty.md`.
* **Giải thích về lưới $0\%, 5\%, 10\%, 20\%$**:
  - Tại mức nhiễu $5\%$ ($\epsilon=0.05$), mean $\Delta\text{CPC}$ đã âm ($-0.00087$, độ suy giảm $-24.7\%$, chỉ còn $17/50$ thành phố dương).
  - Do đó, tại mức $10\%$ và $20\%$, phân phối bị méo nghiêm trọng khiến hiệu chỉnh gây hại (đã được xác nhận qua pilot toy-test trong `test_noise_summary`). Runner chính thức chuyển sang lưới mịn $0\% - 5\%$ để định vị chính xác điểm phá vỡ sinh học của mô hình.
* **Cơ chế bơm nhiễu**: Nhiễu được sinh độc quyền trên vector phân phối $Y_D$ thông qua hàm `generate_nested_noisy_yd`. Ground-truth OD và dự đoán $M_0$ hoàn toàn không bị can thiệp.

---

## 7. PLACEBO & MATCHED PLACEBO AUDIT (P1)

* **Source file**: `results/e1_canonical_specificity_v2/e1_specificity_results.json`, `results/placebo_matched_v2/matched_placebo_per_city.csv`, `placebo.md`
* **Field / Function nguồn**: `e1_core.py`, `run_placebo_matched_v2.py`
* **Protocol check**: **PASS**

### Phân tích đối đầu Target $Y_D$ vs các điều kiện Placebo (50 Đô thị $\times$ 3 Seeds, $B=1000$ Draws, Fold-Stratified Bootstrap)

| Điều kiện kiểm nghiệm | Mean $\Delta\text{CPC}$ | 95% Fold-Stratified CI | Benefit vs $M_0$ ($p_{\text{2-sided}}$) | Benefit vs $M_0$ ($p_{\text{1-sided}}$) | Specificity Gain ($Target - Placebo$) | Specificity 95% CI | Target vs Placebo ($p_{\text{1-sided}}$) | Win Rate ($Target > Placebo$) |
|---|---|---|---|---|---|---|---|---|
| **1. Oracle Target $Y_D$** | `+0.003539` | `[+0.00260, +0.00450]` | `1.93e-09` | `9.66e-10 (greater)` | **`—`** | `—` | `—` | **45/50 (vs M0)** |
| **2. Raw Test Donors (E1-v2 exact 9 donors)** | `-0.037721` | `[-0.04357, -0.03268]` | `1.78e-15` | `8.88e-16 (less)` | **`+0.041261`** | `[+0.03641, +0.04688]` | `8.88e-16` | **50/50** |
| **3. Raw Test Donors ($B=1000$ draws)** | `-0.037787` | `[-0.04358, -0.03278]` | `1.78e-15` | `8.88e-16 (less)` | **`+0.041326`** | `[+0.03646, +0.04688]` | `8.88e-16` | **50/50** |
| **4. Raw Training Donors ($B=1000$ draws)** | `-0.035148` | `[-0.04014, -0.03067]` | `1.78e-15` | `8.88e-16 (less)` | **`+0.038687`** | `[+0.03431, +0.04349]` | `8.88e-16` | **50/50** |
| **5. Dose-Matched Training Donors ($B=1000$)** | `-0.000091` | `[-0.00089, +0.00071]` | `0.4097` (n.s.) | `0.2049 (less)` | **`+0.003630`** | `[+0.00287, +0.00445]` | `2.19e-11` | **46/50** |
| **6. Raw Fold Train-Mean $Y_D$** | `-0.017735` | `[-0.02365, -0.01243]` | `4.91e-12` | `2.46e-12 (less)` | **`+0.021275`** | `[+0.01613, +0.02706]` | `4.44e-15` | **48/50** |
| **7. Dose-Matched Fold Train-Mean $Y_D$** | `+0.000914` | `[+0.00001, +0.00186]` | `0.4319` (n.s.) | `0.2160 (greater)` | **`+0.002626`** | `[+0.00197, +0.00336]` | `4.03e-11` | **47/50** |
| **8. Permuted Target $Y_D$ ($B=1000$ draws)** | `-0.006964` | `[-0.00914, -0.00512]` | `1.78e-15` | `8.88e-16 (less)` | **`+0.010504`** | `[+0.00843, +0.01279]` | `1.78e-15` | **49/50** |

* **Giải thích thống nhất về chênh lệch Placebo (Reconciliation)**:
  - **Dose-Matched Placebo làm bằng chứng chính (Primary Specificity Evidence)**: Khi chuẩn hóa độ lệch L2 về đúng liều lượng của target ($D_T$), can thiệp sai hướng không mang lại cải thiện ($\Delta\text{CPC} \approx -0.000091, p_{\text{spec}} = 2.19 \times 10^{-11}$). Hướng train-mean dose-matched tạo ra mean gain dương nhẹ ($+0.000914$, bootstrap CI $[+0.00001, +0.00186]$ do phản ánh quy luật trọng lực chung), nhưng không có cải thiện hệ thống theo cặp ($p_{\text{Wilcoxon}} = 0.4319$, n.s.). Ngược lại, Oracle Target $Y_D$ mang lại bước nhảy vọt hệ thống trên 45/50 đô thị ($+0.003539$, vượt trội áp đảo train-mean với $p = 4.03 \times 10^{-11}$).
  - **Raw Donors làm kiểm tra áp lực (Secondary Stress-Test Evidence)**: Áp raw distribution ngoại lai ($-0.035$ đến $-0.038$) hoặc raw train-mean ($-0.0177$) gây méo cự ly nghiêm trọng do lệch bán kính vật lý đô thị ($p < 10^{-15}$).
  - Chi tiết tại `results/unified_placebo_v1/unified_placebo_reconciled_summary.md`.

---

## 8. BACKBONE ROBUSTNESS (P1)

* **Source file**: `results/tables/table7_backbone_robustness.md`, `results/backbone_robustness_results.json`, `results/mlp_backbone_results.json`
* **Field / Function nguồn**: `src.experiment.run_backbone_robustness`, `src.experiment.run_mlp_backbone_test`
* **Protocol check**: **PASS** (Gate 13, Gates 26-50 passed)

| Kiến trúc Backbone | $M_0$ CPC | $M_1$ CPC | Mean $\Delta\text{CPC}$ | Median $\Delta\text{CPC}$ | 95% Bootstrap CI | Win Rate | Wilcoxon $p$-value | $\Delta\text{RMSE}$ |
|---|---|---|---|---|---|---|---|---|
| **Gravity-Informed Urban GNN** | 0.71281 ± 0.04434 | 0.71635 ± 0.04454 | **`+0.003539`** | `+0.001953` | `[+0.00261, +0.00453]` | **90.0%** (45/50) | `1.93e-09` | -2.9826 |
| **Node MLP (No Graph MP)** | 0.70913 ± 0.04754 | 0.71242 ± 0.04737 | **`+0.003288`** | `+0.002060` | `[+0.00249, +0.00417]` | **94.0%** (47/50) | `4.38e-11` | -2.5714 |
| **Classical 2-Param Gravity** | 0.38868 ± 0.15312 | 0.38952 ± 0.15435 | **`+0.000835`** | `-0.000059` | `[+0.00018, +0.00156]` | **44.0%** (22/50) | `0.3545` (n.s.) | -0.9335 |

* **Kiểm tra luận điểm khoa học**:
  - Luận điểm: *"$Y_D$ gain không phụ thuộc vào một kiến trúc duy nhất mà mang tính tổng quát cho các neural mobility models."*
  - **XÁC NHẬN (CONFIRMED)**: Cả Urban GNN ($\Delta\text{CPC}=+0.00354, p=1.93\times 10^{-9}$) và Node MLP ($\Delta\text{CPC}=+0.00329, p=4.38\times 10^{-11}$) đều đạt mức tăng trưởng có ý nghĩa thống kê cực kỳ vững chắc với win rate $\ge 90\%$.
  - Ngược lại, Classical Gravity truyền thống không có khả năng học biểu diễn không gian ($M_0 = 0.3887$), do đó hiệu chỉnh vĩ mô không tạo ra bước nhảy vọt thống kê ($p=0.3545$, không có ý nghĩa).
  - **Khoảng tin cậy chuẩn thống nhất (Primary 95% CI)**: Đạt mức đồng nhất tuyệt đối giữa Fold-Stratified City Bootstrap $[+0.00261, +0.00451]$ và Fold-Stratified Hierarchical City $\times$ Seed Bootstrap $[+0.00259, +0.00451]$. Do đó, paper sử dụng duy nhất một primary CI chuẩn là **`[+0.0026, +0.0045]`** (chi tiết tại `results/audit/fold_stratified_hierarchical_bootstrap.json`).

---

## 9. DIRECT-OD EQUIVALENCE & CALIBRATION CORRECTNESS (P2)

* **Source file**: `src/calibration/bin_calibration.py`, `results/direct_od_equivalence_v1/audit_report.json`, `results/k_sensitivity_v1/k_sensitivity_per_city.csv`
* **Field / Function nguồn**: `calibrate_kbins`, `audit_direct_od_v1.py`
* **Protocol check**: **PASS** (Gate 5, Gate 6 passed)

| Tiêu chí kiểm định | Kết quả kiểm toán thực tế | Trạng thái |
|---|---|---|
| Bảo toàn prediction support | 100% giữ nguyên tập index $\Omega_c^+$; chỉ điều chỉnh giá trị flow | **PASS** |
| Sinh thêm OD pair mới | Hoàn toàn không sinh thêm bất kỳ OD pair nào | **PASS** |
| Empty-bin rate ($K=8$) | **`0.0%`** (Tất cả 50 cities đều có đủ $8/8$ active bins) | **PASS** |
| Số lượng active bins / city | Đúng **`8.0`** trên toàn bộ 50 đô thị | **PASS** |
| Phân bố trọng số hiệu chỉnh $w_k$ | 100% đô thị có $w_{\min} < 1.0$ (mean $0.755$, range $[0.224, 0.976]$) và $w_{\max} > 1.0$ (mean $1.310$). Thống kê $\max_k w_k$ qua 50 đô thị: $\min=1.017, \text{mean}=1.310, \max=3.345$ (xem `results/audit/calibration_weight_audit.md`) | **PASS** |
| Clipping / Truncation | Không dùng ad-hoc clipping; chuẩn hóa bảo toàn khối lượng chính tắc | **PASS** |
| Bảo toàn tổng flow dự đoán | Sai số khối lượng tương đối $< 3.72 \times 10^{-16}$ (mức máy tính) | **PASS** |
| Đồng nhất giữa Code và Methods | 100% tương đương toán học với công thức Soft KL Projection ($q=1.0$) | **PASS** |

---

## 10. SUPPORT AUDIT (P0 — CỰC KỲ QUAN TRỌNG)

* **Source file**: `results/audit/ordered_support_manifest.json`, `src/data/dataset.py`, `src/training/evaluate.py`, `run_research_contract_tests.py` (Gates 7, 18, 52)
* **Field / Function nguồn**: `load_city`, `test_gate_7`, `test_gate_18`, `test_gate_52`
* **Protocol check**: **PASS**

### Bằng chứng trực tiếp từ Code & Data:
1. **Missing pair = UNKNOWN**: Dữ liệu nạp từ `data/{city}/pairs/od.csv` chỉ chứa các cặp có ghi nhận luồng thực tế. Ma trận OD không bị zero-filled.
2. **Evaluation Domain**: Toàn bộ metric chính chỉ tính trên tập observed positive interzonal support:
   $$\Omega_c^+ = \{(i,j) \in \Omega_c : i \ne j, D_{ij} > 0, T_{ij}^{\text{GT}} \ge 1\}$$
3. **Inference Wiring**: Các mảng `pair_o_idx` và `pair_d_idx` của target city được nạp trực tiếp vào decoder tại thời điểm inference zero-shot. Không có nhầm lẫn thứ tự node hay index.
4. **Không trộn lẫn zero-negative sampling**: Quá trình kiểm tra Gate 7 và Gate 18 đã xác minh: $100\%$ các cặp trong domain đều có $T_{ij} \ge 1$; intrazonal ($i=j$) và distance $=0$ bị tách biệt nghiêm ngặt.
5. **Đóng băng Hash**: Toàn bộ 50 file `od.csv` khớp 100% mã SHA256 được khóa tại `results/audit/ordered_support_manifest.json` (Gate 52 PASS).

---

## 11. DATA SPLIT AUDIT (P0)

* **Source file**: `results/e1/splits_manifest_v2.json`, `src/data/city_splits.py`, `results/5fold_results.json`
* **Field / Function nguồn**: `generate_35_5_10_splits`, `test_gate_1_split_integrity`, `test_gate_2_data_leakage`
* **Protocol check**: **PASS** (Gate 1, Gate 2, Gate 54 passed)

| Thuộc tính phân hoạch | Giá trị xác nhận | Trạng thái |
|---|---|---|
| Tổng số đô thị | `50` unique cities | **PASS** |
| Số lượng Folds | `5` folds cross-validation | **PASS** |
| Cơ cấu từng Fold | **35 Train / 5 Validation / 10 Test** | **PASS** |
| Độc lập giữa các tập | Train $\cap$ Val = $\emptyset$, Train $\cap$ Test = $\emptyset$, Val $\cap$ Test = $\emptyset$ | **PASS** |
| Độ phủ tập Test | Mỗi đô thị xuất hiện **đúng 1 lần duy nhất** ở tập Test ($5 \times 10 = 50$) | **PASS** |
| Split Random Seed | `VALIDATION_STRATA_SEED = 20260818` (5-stratum size stratification) | **PASS** |
| Ngăn chặn rò rỉ Scaler | `StandardScaler` chỉ `fit()` trên 35 train cities của fold đó (Gate 2 PASS) | **PASS** |
| Tính toàn vẹn tổng hợp | Cả 5 folds đều đã chạy hoàn tất và được tổng hợp đầy đủ | **PASS** |

---

## 12. HYPERPARAMETER-SELECTION LEAKAGE AUDIT (P0)

* **Source file**: `src/calibration/bin_calibration.py`, `src/training/train.py`, `PROTOCOL_CONTRACT.md`
* **Field / Function nguồn**: `calibrate_kbins`, `train_model`, `compute_qstar.py`
* **Protocol check**: **PASS** (Gate 4, Gate 28 passed)

### Kết quả thẩm định độc lập:
1. **$q=1.0$ được khóa tiên nghiệm**: Trong toàn bộ các thử nghiệm chính thức (E1 và 5-Fold), $q$ **không hề qua tuning hay hyperparameter search**; $q$ được cố định chặt chẽ ở $q=1.0$ (Closed-form mass-preserving projection).
2. **Không dùng CPC mục tiêu để chọn $q$**: Hoàn toàn không có bước dò $q^*$ trên test city để lấy kết quả báo cáo.
3. **$K=8$ được quy định tiên nghiệm**: $K=8$ được định sẵn trong Protocol Contract v1 trước khi có kết quả test.
4. **Early Stopping thuần khiết**: Checkpoint được lựa chọn dựa trên interzonal CPC của **5 validation cities** trong fold (`patience=15, max_epochs=200`).
5. **Test city hoàn toàn held-out**: Test cities không tham gia học trọng số, không tham gia chọn epoch, không tham gia xác định bin edges.

---

## 13. $Y_D$ EXTRACTION AUDIT (P0)

* **Source file**: `src/data/yd_extractor.py`, `PROTOCOL_CONTRACT.md`
* **Field / Function nguồn**: `compute_kbin_edges`, `extract_yd_kbins`
* **Protocol check**: **PASS** (Gate 51 passed)

| Khía cạnh kỹ thuật | Hiện trạng xác minh trực tiếp từ mã nguồn |
|---|---|
| Nguồn dữ liệu $Y_D$ | $Y_D$ trong E1 được trích xuất từ ground-truth positive interzonal flows của target city (`yd_source = "target_ground_truth_positive_od"`). Đây là Oracle Existence Test đã tuyên bố minh bạch. |
| Xây dựng ranh giới bin (Edges) | Bin edges được tính **hoàn toàn từ 35 training cities** bằng quantile phân vị gộp trọng số theo cặp (`compute_kbin_edges`). Test city không tham gia tạo bin edges. |
| Phân loại Quantile Bins | Pooled training quantiles (theo từng fold), không phải phân vị riêng của từng target city. |
| Chuẩn hóa phân phối | $\sum_{k=1}^K Y_{D,k} = 1.0$ (tổng xác suất luôn bằng đúng 1.0). |
| Xử lý Empty/Inactive Bins | Nếu một bin không có cặp ứng viên trong target city, bin đó được đánh dấu inactive và phân phối $Y_D$ được tái chuẩn hóa có điều kiện trên các active bins. |
| Khối lượng thông tin rò rỉ | Calibration chỉ nhận duy nhất vector 1D histogram $Y_D \in \mathbb{R}^K$ (8 con số tỷ lệ). Mô hình **không nhận** ma trận luồng hay thông tin từng cặp $T_{ij}^{\text{GT}}$. |

---

## 14. MECHANISM DIAGNOSTIC: INTRA-BIN ALLOCATION (P2)

* **Source file**: `results/intra_bin_mechanism_diagnostic.json`, `src/experiment/run_intra_bin_mechanism_diagnostic.py`
* **Field / Function nguồn**: `correlations`, `rank_invariance`, `per_city`
* **Protocol check**: **PASS**

### Tương quan giữa độ lệch phân phối ban đầu ($d_{\text{pre}}$) và $\Delta\text{CPC}$
* **Định nghĩa $d_{\text{pre}}$**: Total Variation (TV) distance giữa phân phối khoảng cách dự đoán của $M_0$ và target $Y_D$:
  $$d_{\text{pre}} = \frac{1}{2} \sum_{k=1}^K |\hat{Y}_k^{ZS,+} - Y_{D,k}^{\text{cond},+}|$$
  - Mean $d_{\text{pre}} = 0.044368$, Median $= 0.041572$, Range: $[0.008345, 0.101717]$.
* **Tương quan với $\Delta\text{CPC}$**:
  - **Pearson $r$**: **`0.799496`** ($\approx \mathbf{0.80}$), **$p = 3.36 \times 10^{-12}$**, 95% CI: `[0.670317, 0.881645]`
  - **Spearman $\rho$**: **`0.746363`** ($\approx \mathbf{0.75}$), **$p = 4.92 \times 10^{-10}$**
  - *Ý nghĩa khoa học*: Khẳng định cơ chế mạnh nhất — các thành phố có sai lệch vĩ mô ban đầu càng lớn thì việc bổ sung $Y_D$ mang lại bước nhảy vọt $\Delta\text{CPC}$ càng cao!

### Tương quan chất lượng phân bổ nội bin ($Q_c^{\text{intra}}$)
* **Chất lượng phân bổ nội bin ($Q_c^{\text{intra, alloc}}$)**: Mean $= 0.726200$, Median $= 0.726594$
  - Pearson $r = -0.005403$ ($p = 0.9703$, 95% CI: `[-0.283, 0.273]`), Spearman $\rho = 0.073133$ ($p = 0.6138$).
* **Chất lượng thứ bậc nội bin ($Q_c^{\text{intra, rank}}$)**: Mean $= 0.715502$, Median $= 0.730247$
  - Pearson $r = 0.063518$ ($p = 0.6612$, 95% CI: `[-0.219, 0.336]`), Spearman $\rho = 0.179736$ ($p = 0.2117$).
* **Bảo toàn thứ bậc tuyệt đối (Rank Invariance)**:
  - `max_abs_q_rank_m0_minus_m1`: **`0.0`** (Toán học và thực nghiệm chứng minh toán tử hiệu chỉnh nhân vô hướng bảo toàn $100\%$ thứ tự xếp hạng các cặp OD trong từng bin khoảng cách).

### Các đô thị ngoại vi (Outlier Cities)
* **Top 3 đô thị hưởng lợi nhiều nhất**:
  1. **Los_Angeles**: $\Delta\text{CPC} = +0.015431$ ($d_{\text{pre}} = 0.101717$)
  2. **Phoenix**: $\Delta\text{CPC} = +0.012381$ ($d_{\text{pre}} = 0.080716$)
  3. **Houston**: $\Delta\text{CPC} = +0.011219$ ($d_{\text{pre}} = 0.096119$)
* **Top 3 đô thị suy giảm nhiều nhất**:
  1. **El_Paso**: $\Delta\text{CPC} = -0.002842$ ($d_{\text{pre}} = 0.060018$)
  2. **Oklahoma_City**: $\Delta\text{CPC} = -0.002610$ ($d_{\text{pre}} = 0.048287$)
  3. **Jacksonville**: $\Delta\text{CPC} = -0.001552$ ($d_{\text{pre}} = 0.045999$)

---

## 15. SECONDARY METRICS: MAE & RMSE (P2)

* **Source file**: `results/5fold_results.json` (`city_level_results`), `results/tables/table7_backbone_robustness.md`
* **Field / Function nguồn**: `M0.mae_inter`, `M1_city_oracle_obs.mae_inter`, `M0.rmse_inter`, `M1_city_oracle_obs.rmse_inter`
* **Protocol check**: **PASS**

| Metric phụ | $M_0$ Baseline | $M_1$ Calibrated | Delta Mean ($\Delta$) | Delta Median | Win Rate (Giảm sai số) | Tính nhất quán với CPC |
|---|---|---|---|---|---|---|
| **MAE Interzonal** | `204.0734` | `201.5344` | **`-2.5390`** | **`-1.5043`** | **90.0%** (45/50 cities) | **Hoàn toàn nhất quán** (cùng 45/50 cities cải thiện) |
| **RMSE Interzonal** | `531.9865` | `529.0039` | **`-2.9826`** | **`-1.8035`** | **64.0%** (32/50 cities) | **Nhất quán về dấu** (Mean RMSE giảm 2.98 đơn vị) |

---

## BẢNG TRA CỨU FILE NGUỒN VÀ DẤU VẾT XÁC MINH (PROVENANCE REGISTRY)

| Hạng mục kiểm toán | Tệp kết quả nguồn (Source File Path) | Khóa JSON / Trường dữ liệu chính |
|---|---|---|
| Main 50 cities | `results/5fold_results.json` | `rq1_delta_r.city` |
| E1 Specificity Benchmark | `results/e1_canonical_specificity_v2/e1_specificity_results.json` | `summary`, `per_city_seed_averaged` |
| 55 Research Gates | `run_research_contract_tests.py` | `GATE_RESULTS` (55/55 PASS) |
| Random Seeds | `seed.md`, `PROTOCOL_CONTRACT.md` | Section 2 |
| K-Sensitivity | `results/k_sensitivity_v1/k_sensitivity_summary.json` | `summary`, `contrasts` |
| Noise Robustness | `results/noise_robustness_fine_v1/noise_summary.json` | `results_by_eps` |
| Matched Placebo | `results/placebo_matched_v2/matched_placebo_per_city.csv` | `target_delta_mean`, `wrong_delta_mean` |
| Backbone Robustness | `results/tables/table7_backbone_robustness.md`, `results/mlp_backbone_results.json` | `summary`, `rq1_delta_r.city` |
| Mechanism Diagnostic | `results/intra_bin_mechanism_diagnostic.json` | `correlations`, `rank_invariance`, `per_city` |
| Data Splits Manifest | `results/e1/splits_manifest_v2.json` | SHA256: `7f9afe02725c7798dab018b6...` |
| OD Support Manifest | `results/audit/ordered_support_manifest.json` | SHA256 của 50 tệp `od.csv` |

---
*Tài liệu này là căn cứ duy nhất (Single Source of Truth) đã được xác minh toàn diện từ tệp nguồn phục vụ cho việc viết bản thảo bài báo khoa học.*
