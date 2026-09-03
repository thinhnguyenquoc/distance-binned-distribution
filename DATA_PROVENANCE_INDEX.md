# Master Data Provenance & Cross-Reference Index (Sổ Tay Đối Chứng Dữ Liệu Toàn Bài Báo)

Tài liệu này cung cấp bản đồ đối chứng toàn diện (provenance index) giữa mọi **Bảng số liệu (Tables)**, **Hình vẽ (Figures)**, và **Số liệu thống kê chính (Key Metrics)** trong bài báo với chính xác các file dữ liệu nguồn nằm trong thư mục `results/`. 

Mỗi mục đều có đường dẫn có thể nhấp trực tiếp (direct links), cấu trúc key/cột dữ liệu, script sinh kết quả và lệnh Python 1 dòng để kiểm tra đối chiếu tức thì.

---

## Mục lục tra cứu nhanh

1. [Các số liệu cốt lõi toàn bài báo (Abstract, Intro, Conclusion)](#1-các-số-liệu-cốt-lõi-toàn-bài-báo)
2. [Bảng số liệu (Tables 1 - 8)](#2-bảng-số-liệu-tables-1---8)
   - [Table 1: Benchmark tái tạo luồng chính (K=8, N=50)](#table-1-primary-benchmark-k8-n50)
   - [Table 2: Tính đặc thù mục tiêu & Placebo controls](#table-2-target-specificity--placebo-controls)
   - [Table 3: Thang độ phân giải khoảng cách (K=2 đến 20)](#table-3-distance-bin-resolution-scaling)
   - [Table 4: Độ nhạy nhiễu Total Variation & Ngưỡng Crossover](#table-4-noise-sensitivity--tv-crossover)
   - [Table 5: Độ ổn định qua các model seeds (Seeds 1, 10, 100)](#table-5-model-initialization-robustness)
   - [Table 6: Tính tổng quát qua các kiến trúc backbone (GNN, MLP, Gravity)](#table-6-backbone-architecture-generality)
   - [Table 7: So sánh vận hành với quan sát cặp OD trực tiếp (OD-FE)](#table-7-direct-od-observations-comparison)
   - [Table 8: Phân tích cơ chế hồi quy lệch khoảng cách ban đầu (d_pre)](#table-8-mechanistic-regression--partial-correlation)
3. [Biểu đồ & Hình ảnh (Figures 1 - 6)](#3-biểu-đồ--hình-ảnh-figures-1---6)
   - [Figure 1: Khung hiệu chỉnh Oracle (Sơ đồ kiến trúc)](#figure-1-oracle-calibration-framework)
   - [Figure 2: Cải thiện CPC theo từng thành phố (50 MSAs)](#figure-2-city-level-delta-cpc-50-cities)
   - [Figure 3: Độ nhạy độ phân giải (K-sweep & County)](#figure-3-resolution-sensitivity)
   - [Figure 4: Quan hệ liều-đáp ứng nhiễu quan sát (Noise dose-response)](#figure-4-observation-noise-dose-response)
   - [Figure 5: Kiểm chứng Placebo đối chứng ghép cặp](#figure-5-matched-placebo-controls)
   - [Figure 6: Chẩn đoán cơ chế (d_pre vs Delta CPC)](#figure-6-mechanistic-diagnostic)
4. [Tập lệnh kiểm tra nhanh tất cả các bảng (One-line Verification Scripts)](#4-tập-lệnh-kiểm-tra-nhanh)

---

## 1. Các số liệu cốt lõi toàn bài báo

Các số liệu này xuất hiện đồng nhất trong **Section 0 (Abstract)**, **Section 1 (Introduction)**, **Section 4 (Results)**, và **Section 6 (Conclusion)**:

| Chỉ số trong bài báo | Giá trị công bố | File dữ liệu nguồn | Key / Cột đối chứng | Lệnh kiểm tra nhanh |
|---|:---:|---|---|---|
| **Mean $\Delta\mathrm{CPC}$** | `+0.00354` (`0.0035395`) | [`results/5fold_results.json`](results/5fold_results.json) | `rq1_delta_r.city.delta_cpc_inter.mean` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['delta_cpc_inter']['mean'])"` |
| **95% Bootstrap CI** | `[+0.0026, +0.0045]` | [`results/5fold_results.json`](results/5fold_results.json) | `rq1_delta_r.city.delta_cpc_inter.ci_95_lower`, `ci_95_upper` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['delta_cpc_inter']['ci_95_lower'], d['rq1_delta_r']['city']['delta_cpc_inter']['ci_95_upper'])"` |
| **Median $\Delta\mathrm{CPC}$** | `+0.00195` (`0.0019531`) | [`results/5fold_results.json`](results/5fold_results.json) | `rq1_delta_r.city.delta_cpc_inter.median` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['delta_cpc_inter']['median'])"` |
| **Tỷ lệ thắng (Win Rate)** | `45 / 50 (90.0%)` | [`results/5fold_results.json`](results/5fold_results.json) | `rq1_delta_r.city.p_improved` (=0.9) | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(sum(x['delta_city']>0 for x in d['city_level_results']))"` |
| **Wilcoxon test (2-sided)** | `1.93 x 10^-9` | [`results/5fold_results.json`](results/5fold_results.json) | `rq1_delta_r.city.wilcoxon_two_sided_p` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['wilcoxon_two_sided_p'])"` |
| **Rank-biserial $r_{\mathrm{rb}}$** | `0.8698` | [`results/5fold_results.json`](results/5fold_results.json) | `rq1_delta_r.city.rank_biserial_r` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['rank_biserial_r'])"` |

---

## 2. Bảng số liệu (Tables 1 - 8)

### Table 1: Primary Benchmark ($K=8, N=50$)
* **Vị trí trong bài báo:** [`paper/section4_results.md:L30-L38`](paper/section4_results.md#L30-L38) | [`paper/full_paper_en.md:L439-L446`](paper/full_paper_en.md#L439-L446)
* **File dữ liệu nguồn:**
  1. [`results/5fold_results.json`](results/5fold_results.json)
  2. [`results/audit/dpre_mechanism_data.csv`](results/audit/dpre_mechanism_data.csv)
  3. [`results/PROTOCOL_LOCK.md`](results/PROTOCOL_LOCK.md)
* **Mapping chi tiết từng cell:**
  - Zero-Shot $M_0$ CPC ($0.71281 \pm 0.04434$, Median $0.71632$): `rq1_delta_r.city.m0_cpc_inter` (`mean`, `std`, `median`).
  - Calibrated $M_1$ CPC ($0.71635 \pm 0.04454$, Median $0.71988$): `rq1_delta_r.city.m1_cpc_inter` (`mean`, `std`, `median`).
  - Mean $\Delta\mathrm{CPC} = +0.00354$, CI $[+0.0026, +0.0045]$, Win rate $45/50$, $p=1.93\times 10^{-9}$.

---

### Table 2: Target Specificity & Placebo Controls
* **Vị trí trong bài báo:** [`paper/section4_results.md:L62-L76`](paper/section4_results.md#L62-L76) | [`paper/full_paper_en.md:L470-L484`](paper/full_paper_en.md#L470-L484)
* **File dữ liệu nguồn:**
  1. [`results/e1_canonical_specificity_v2/e1_specificity_results.json`](results/e1_canonical_specificity_v2/e1_specificity_results.json)
  2. [`results/placebo_matched_v2/matched_placebo_per_city.csv`](results/placebo_matched_v2/matched_placebo_per_city.csv)
  3. [`results/placebo_matched_v2/matched_placebo_raw.csv`](results/placebo_matched_v2/matched_placebo_raw.csv)
* **Mapping chi tiết:**
  - **Row 1 (Oracle Target $Y_D$):** `e1_specificity_results.json -> summary.delta_cpc_target_mean` ($+0.003539$), `ci_l` ($0.002607$), `ci_h` ($0.004483$), `n_positive_target` ($45$).
  - **Row 2 (Dose-Matched Training Donors):** `matched_placebo_per_city.csv -> wrong_delta_mean` (Mean: $-0.000091$), `specificity_wrong_mean` ($+0.003630$), $p=2.19\times 10^{-11}$, Win: $46/50$.
  - **Row 3 (Dose-Matched Train-Mean):** `matched_placebo_per_city.csv -> trainmean_delta_mean` ($+0.000914$), `specificity_trainmean_mean` ($+0.002626$), $p=4.03\times 10^{-11}$, Win: $47/50$.
  - **Row 4 (Raw Test Donors, In-Fold):** `e1_specificity_results.json -> summary.delta_cpc_wrong_mean` ($-0.037721$), `delta_specificity_mean` ($+0.041261$), $p=8.88\times 10^{-16}$, Win: $50/50$.
  - **Row 8 (Permuted Target $Y_D$):** `matched_placebo_per_city.csv -> permuted_delta_mean` ($-0.006964$), `specificity_permuted_mean` ($+0.010504$), $p=1.78\times 10^{-15}$, Win: $49/50$.

---

### Table 3: Distance-Bin Resolution Scaling ($K \in \{2, 4, \dots, 20\}$)
* **Vị trí trong bài báo:** [`paper/section4_results.md:L93-L110`](paper/section4_results.md#L93-L110) | [`paper/full_paper_en.md:L501-L518`](paper/full_paper_en.md#L501-L518)
* **File dữ liệu nguồn:**
  1. [`results/k_sensitivity_v1/k_sensitivity_summary.json`](results/k_sensitivity_v1/k_sensitivity_summary.json)
  2. [`results/k_sensitivity_v1/k_sensitivity_per_city.csv`](results/k_sensitivity_v1/k_sensitivity_per_city.csv)
  3. [`results/k_sensitivity_v1/k_sensitivity_raw.csv`](results/k_sensitivity_v1/k_sensitivity_raw.csv)
* **Mapping chi tiết:**
  - `summary.json -> summary`: Mảng chứa thông số cho từng $K$:
    - $K=2$: Mean $+0.00098$, CI $[+0.00052, +0.00151]$, Win: $39/50$, Gain/bin: $0.000488$
    - $K=8$: Mean $+0.00354$, CI $[+0.00262, +0.00447]$, Win: $45/50$, Gain/bin: $0.000442$
    - $K=20$: Mean $+0.00639$, CI $[+0.00508, +0.00769]$, Win: $46/50$, Gain/bin: $0.000319$

---

### Table 4: Noise Sensitivity & TV Crossover
* **Vị trí trong bài báo:** [`paper/section4_results.md:L154-L166`](paper/section4_results.md#L154-L166) | [`paper/full_paper_en.md:L562-L574`](paper/full_paper_en.md#L562-L574)
* **File dữ liệu nguồn:**
  1. [`results/noise_robustness_fine_v1/noise_summary.json`](results/noise_robustness_fine_v1/noise_summary.json)
  2. [`results/noise_robustness_fine_v1/noise_per_city.csv`](results/noise_robustness_fine_v1/noise_per_city.csv)
  3. [`results/noise_robustness_fine_v1/noise_raw.csv`](results/noise_robustness_fine_v1/noise_raw.csv)
* **Mapping chi tiết:**
  - `noise_summary.json -> eps_cross_zero_dCPC`: Ngưỡng crossover $= 0.044439$ ($4.44\%$, CI: $[4.16\%, 4.77\%]$).
  - `results_by_eps`:
    - $\epsilon=0.00$: Mean $+0.00354$, Positives: $45/50$
    - $\epsilon=0.01$: Mean $+0.00336$, Positives: $44/50$
    - $\epsilon=0.02$: Mean $+0.00282$, Positives: $36/50$
    - $\epsilon=0.03$: Mean $+0.00193$, Positives: $28/50$
    - $\epsilon=0.04$: Mean $+0.00070$, Positives: $18/50$
    - $\epsilon=0.05$: Mean $-0.00087$, Positives: $17/50$

---

### Table 5: Model Initialization Robustness (Seeds 1, 10, 100)
* **Vị trí trong bài báo:** [`paper/section4_results.md:L193-L203`](paper/section4_results.md#L193-L203) | [`paper/full_paper_en.md:L601-L611`](paper/full_paper_en.md#L601-L611)
* **File dữ liệu nguồn:**
  1. [`results/k_sensitivity_v1/k_sensitivity_per_seed.csv`](results/k_sensitivity_v1/k_sensitivity_per_seed.csv)
  2. [`results/sampling_robustness_v1/sampling_per_seed.csv`](results/sampling_robustness_v1/sampling_per_seed.csv)
* **Mapping chi tiết:**
  - Seed 1: Mean $\Delta\mathrm{CPC} = +0.00434$, Median $+0.00207$, CI $[+0.00322, +0.00547]$, Win: $41/50$ ($82\%$).
  - Seed 10: Mean $\Delta\mathrm{CPC} = +0.00308$, Median $+0.00182$, CI $[+0.00216, +0.00404]$, Win: $44/50$ ($88\%$).
  - Seed 100: Mean $\Delta\mathrm{CPC} = +0.00320$, Median $+0.00217$, CI $[+0.00236, +0.00408]$, Win: $44/50$ ($88\%$).
  - Across-seed SD $= 0.00070$.

---

### Table 6: Backbone Architecture Generality
* **Vị trí trong bài báo:** [`paper/section4_results.md:L218-L227`](paper/section4_results.md#L218-L227) | [`paper/full_paper_en.md:L626-L635`](paper/full_paper_en.md#L626-L635)
* **File dữ liệu nguồn:**
  1. [`results/backbone_robustness_results.json`](results/backbone_robustness_results.json)
  2. [`results/mlp_backbone_results.json`](results/mlp_backbone_results.json)
  3. [`results/mlp_backbone_execution.log`](results/mlp_backbone_execution.log)
* **Mapping chi tiết:**
  - Urban GNN: Mean $+0.00354$, CI $[+0.0026, +0.0045]$, Win $45/50$, $p=1.93\times 10^{-9}$.
  - Node MLP: Mean $+0.00329$, CI $[+0.0025, +0.0042]$, Win $47/50$ ($94\%$), $p=4.38\times 10^{-11}$.
  - Classical Gravity: Mean $+0.00084$, CI $[+0.0002, +0.0016]$, Win $22/50$ ($44\%$), $p=0.3545$ (n.s.).

---

### Table 7: Direct-OD Observations Comparison (OD-FE)
* **Vị trí trong bài báo:** [`paper/section4_results.md:L246-L260`](paper/section4_results.md#L246-L260) | [`paper/full_paper_en.md:L654-L668`](paper/full_paper_en.md#L654-L668)
* **File dữ liệu nguồn:**
  1. [`results/direct_od_equivalence_v1/combined/summary.json`](results/direct_od_equivalence_v1/combined/summary.json)
  2. [`results/partial_od_equivalence_v2/combined/summary.json`](results/partial_od_equivalence_v2/combined/summary.json)
  3. [`results/direct_od_equivalence_v1/audit_report.json`](results/direct_od_equivalence_v1/audit_report.json)
* **Mapping chi tiết:**
  - Điểm giao cắt nội suy: $p_{\mathrm{eq}} \approx 0.20\%$ (CI: $[0.133\%, 0.287\%]$), tại đó Direct OD Gain $\approx +0.00354$, tương đương khoảng 35 cặp OD lộ diện.
  - Tỷ lệ $p=0.10\%$: $\Delta\mathrm{CPC} = +0.00180$ (thấp hơn $Y_D$ với $D=-0.00174$).
  - Tỷ lệ $p=0.25\%$: $\Delta\mathrm{CPC} = +0.00448$ (cao hơn $Y_D$ với $D=+0.00094$).

---

### Table 8: Mechanistic Regression & Partial Correlation ($d_{\text{pre}}$)
* **Vị trí trong bài báo:** [`paper/section4_results.md:L305-L317`](paper/section4_results.md#L305-L317) | [`paper/full_paper_en.md:L713-L725`](paper/full_paper_en.md#L713-L725)
* **File dữ liệu nguồn:**
  1. [`results/audit/dpre_mechanism_data.csv`](results/audit/dpre_mechanism_data.csv)
  2. [`results/audit/dpre_mechanism_summary.json`](results/audit/dpre_mechanism_summary.json)
  3. [`results/intra_bin_mechanism_diagnostic.json`](results/intra_bin_mechanism_diagnostic.json)
* **Mapping chi tiết:**
  - Bivariate Pearson: $r = +0.7995$, $p = 3.36 \times 10^{-12}$.
  - Bivariate Spearman: $\rho = +0.7464$, $p = 4.92 \times 10^{-10}$.
  - Full Partial Correlation ($M_0 + \log N_{\text{pairs}} + \log N_{\text{tracts}} + \text{MeanDist}$): $r_{\mathrm{part}} = +0.7951$, $p = 5.35 \times 10^{-12}$.
  - Multivariate OLS $\beta(d_{\mathrm{pre}}) = +0.1487$, $p = 4.12 \times 10^{-11}$, $R^2 = 73.7\%$.

---

## 3. Biểu đồ & Hình ảnh (Figures 1 - 6)

| Biểu đồ | Tên & Mô tả | File hình ảnh | File dữ liệu nguồn | Script tái tạo |
|---|---|---|---|---|
| **Figure 1** | Oracle calibration framework schematic | [`paper/figures/fig1_oracle_calibration_framework.svg`](paper/figures/fig1_oracle_calibration_framework.svg) | Sơ đồ phương pháp luận | Vẽ vector SVG trực tiếp |
| **Figure 2** | City-level $\Delta\mathrm{CPC}$ across 50 test cities | [`paper/figures/fig2_main_per_city.png`](paper/figures/fig2_main_per_city.png) ([.pdf](paper/figures/fig2_main_per_city.pdf)) | [`results/5fold_results.json`](results/5fold_results.json) | [`scripts/generate_all_paper_figures.py`](scripts/generate_all_paper_figures.py) (Hàm `generate_figure2()`) |
| **Figure 3** | Resolution sensitivity (K-sweep & County) | [`paper/figures/fig3_resolution_sensitivity.png`](paper/figures/fig3_resolution_sensitivity.png) ([.pdf](paper/figures/fig3_resolution_sensitivity.pdf)) | [`results/k_sensitivity_v1/k_sensitivity_summary.json`](results/k_sensitivity_v1/k_sensitivity_summary.json) & [`results/spatial_resolution/spatial_resolution_summary.json`](results/spatial_resolution/spatial_resolution_summary.json) | [`scripts/generate_all_paper_figures.py`](scripts/generate_all_paper_figures.py) (Hàm `generate_figure3()`) |
| **Figure 4** | Noise dose-response & TV crossover | [`paper/figures/fig4_noise_dose_response.png`](paper/figures/fig4_noise_dose_response.png) ([.pdf](paper/figures/fig4_noise_dose_response.pdf)) | [`results/noise_robustness_fine_v1/noise_summary.json`](results/noise_robustness_fine_v1/noise_summary.json) | [`scripts/generate_all_paper_figures.py`](scripts/generate_all_paper_figures.py) (Hàm `generate_figure4()`) |
| **Figure 5** | Matched placebo controls (Authentic vs Donor vs Permuted) | [`paper/figures/fig5_structural_validity_placebo.png`](paper/figures/fig5_structural_validity_placebo.png) ([.pdf](paper/figures/fig5_structural_validity_placebo.pdf)) | [`results/placebo_matched_v2/matched_placebo_per_city.csv`](results/placebo_matched_v2/matched_placebo_per_city.csv) | [`scripts/generate_all_paper_figures.py`](scripts/generate_all_paper_figures.py) (Hàm `generate_figure5()`) |
| **Figure 6** | Mechanistic scatter: $d_{\mathrm{pre}}$ vs $\Delta\mathrm{CPC}$ | [`paper/figures/fig6_mechanistic_dpre.png`](paper/figures/fig6_mechanistic_dpre.png) ([.pdf](paper/figures/fig6_mechanistic_dpre.pdf)) | [`results/audit/dpre_mechanism_data.csv`](results/audit/dpre_mechanism_data.csv) | [`scripts/generate_all_paper_figures.py`](scripts/generate_all_paper_figures.py) (Hàm `generate_figure6()`) |

---

## 4. Tập lệnh kiểm tra nhanh (Automated Verification Scripts)

### Cách 1: Chạy script tự động kiểm tra toàn bộ (Khuyên dùng)
Chạy lệnh sau tại thư mục gốc của repository để in ra báo cáo đối chiếu toàn bộ các chỉ số của Paper với dữ liệu thực nghiệm trong `results/`:
```bash
python scripts/verify_all_numbers.py
```

### Cách 2: Các lệnh kiểm tra từng phần (One-line Snippets)
```bash
# 1. Kiểm tra Table 1 & Abstract core numbers
python -c "import json; d=json.load(open('results/5fold_results.json'))['rq1_delta_r']['city']; print(f'Mean: {d[\"delta_cpc_inter\"][\"mean\"]:.5f}, Median: {d[\"delta_cpc_inter\"][\"median\"]:.5f}, WinRate: {d[\"p_improved\"]*100:.1f}%, p: {d[\"wilcoxon_two_sided_p\"]:.2e}')"

# 2. Kiểm tra Table 2 Placebo controls
python -c "import pandas as pd; df=pd.read_csv('results/placebo_matched_v2/matched_placebo_per_city.csv'); print(f'Donor mean: {df[\"wrong_delta_mean\"].mean():.6f}, Permuted mean: {df[\"permuted_delta_mean\"].mean():.6f}, Specificity: {df[\"specificity_wrong_mean\"].mean():.6f}')"

# 3. Kiểm tra Table 3 K-sweep range
python -c "import json; k=json.load(open('results/k_sensitivity_v1/k_sensitivity_summary.json'))['summary']; print(f'K=2: {k[0][\"mean_delta\"]:.5f}, K=8: {k[3][\"mean_delta\"]:.5f}, K=20: {k[-1][\"mean_delta\"]:.5f}')"

# 4. Kiểm tra Table 4 Noise crossover threshold
python -c "import json; n=json.load(open('results/noise_robustness_fine_v1/noise_summary.json')); print(f'TV Crossover: {n[\"eps_cross_zero_dCPC\"]*100:.2f}%')"

# 5. Kiểm tra Table 8 Mechanism correlation
python -c "import pandas as pd, scipy.stats as st; df=pd.read_csv('results/audit/dpre_mechanism_data.csv'); print(f'Pearson r: {st.pearsonr(df[\"d_pre_tv\"], df[\"delta_cpc\"])[0]:.4f}, Spearman rho: {st.spearmanr(df[\"d_pre_tv\"], df[\"delta_cpc\"])[0]:.4f}')"
```
