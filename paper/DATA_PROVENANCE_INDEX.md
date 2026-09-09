# Master Data Provenance & Cross-Reference Index (Sổ Tay Đối Chứng Dữ Liệu Toàn Bài Báo)

Tài liệu này cung cấp bản đồ đối chứng toàn diện (provenance index) giữa mọi **Bảng số liệu (Tables)**, **Hình vẽ (Figures)**, và **Số liệu thống kê chính (Key Metrics)** trong bài báo với chính xác các file dữ liệu nguồn nằm trong thư mục `results/`. 

Mỗi mục đều có đường dẫn có thể nhấp trực tiếp (direct links), cấu trúc key/cột dữ liệu, script sinh kết quả và lệnh Python 1 dòng để kiểm tra đối chiếu tức thì.

---

## Mục lục tra cứu nhanh

1. [Các số liệu cốt lõi toàn bài báo (Abstract, Intro, Conclusion)](#1-các-số-liệu-cốt-lõi-toàn-bài-báo)
2. [Bảng số liệu chính trong bài báo (Tables 1 - 5 / Bảng 1 - 5)](#2-bảng-số-liệu-chính-trong-bài-báo-tables-1---5--bảng-1---5)
   - [Table 1 / Bảng 1: Ký hiệu cốt lõi, nguồn dữ liệu và trạng thái sẵn có của thông tin](#table-1--bảng-1-ký-hiệu-cốt-lõi-nguồn-dữ-liệu-và-trạng-thái-sẵn-có-của-thông-tin)
   - [Table 2 / Bảng 2: Benchmark tái tạo luồng chính với Urban GNN (K=8, N=50)](#table-2--bảng-2-benchmark-tái-tạo-luồng-chính-với-urban-gnn-k8-n50)
   - [Table 3 / Bảng 3: Tính đặc thù mục tiêu & Placebo controls](#table-3--bảng-3-tính-đặc-thù-mục-tiêu--placebo-controls)
   - [Table 4 / Bảng 4: Thang độ phân giải khoảng cách (K=2 đến 20)](#table-4--bảng-4-thang-độ-phân-giải-khoảng-cách-k2-đến-20)
   - [Table 5 / Bảng 5: Tính tổng quát qua các kiến trúc backbone (GNN, MLP, Gravity)](#table-5--bảng-5-tính-tổng-quát-qua-các-kiến-trúc-backbone-gnn-mlp-gravity)
3. [Các phân tích cốt lõi trong văn bản và hình vẽ (Narrative & Figures)](#3-các-phân-tích-cốt-lõi-trong-văn-bản-và-hình-vẽ-narrative--figures)
   - [Phân tích A: Độ nhạy nhiễu Total Variation & Ngưỡng Crossover (Mục 4.3 & Figure 5 / Hình 5)](#phân-tích-a-độ-nhạy-nhiễu-total-variation--ngưỡng-crossover)
   - [Phân tích B: Độ ổn định qua các model seeds (Mục 4.4 text)](#phân-tích-b-độ-ổn-định-qua-các-model-seeds)
   - [Phân tích C: Phân tích cơ chế lệch khoảng cách ban đầu d_pre (Mục 4.5 & Figure 6 / Hình 6)](#phân-tích-c-phân-tích-cơ-chế-lệch-khoảng-cách-ban-đầu-d_pre)
   - [Phân tích D (Bổ trợ/Audit): So sánh vận hành với quan sát cặp OD trực tiếp (OD-FE)](#phân-tích-d-bổ-trợaudit-so-sánh-vận-hành-với-quan-sát-cặp-od-trực-tiếp-od-fe)
4. [Biểu đồ & Hình ảnh (Figures 1 - 6 / Hình 1 - 6)](#4-biểu-đồ--hình-ảnh-figures-1---6--hình-1---6)
   - [Figure 1 / Hình 1: Khung hiệu chỉnh Oracle (Sơ đồ kiến trúc)](#figure-1--hình-1-khung-hiệu-chỉnh-oracle-sơ-đồ-kiến-trúc)
   - [Figure 2 / Hình 2: Cải thiện CPC theo từng thành phố (50 MSAs)](#figure-2--hình-2-cải-thiện-cpc-theo-từng-thành-phố-50-msas)
   - [Figure 3 / Hình 3: Kiểm chứng Placebo đối chứng ghép cặp](#figure-3--hình-3-kiểm-chứng-placebo-đối-chứng-ghép-cặp)
   - [Figure 4 / Hình 4: Độ nhạy độ phân giải (K-sweep & County)](#figure-4--hình-4-độ-nhạy-độ-phân-giải-k-sweep--county)
   - [Figure 5 / Hình 5: Quan hệ liều-đáp ứng nhiễu quan sát (Noise dose-response)](#figure-5--hình-5-quan-hệ-liều-đáp-ứng-nhiễu-quan-sát-noise-dose-response)
   - [Figure 6 / Hình 6: Chẩn đoán cơ chế (d_pre vs Delta CPC)](#figure-6--hình-6-chẩn-đoán-cơ-chế-d_pre-vs-delta-cpc)
5. [Tập lệnh kiểm tra nhanh tất cả các bảng (Automated Verification Scripts)](#5-tập-lệnh-kiểm-tra-nhanh-automated-verification-scripts)

---

## 1. Các số liệu cốt lõi toàn bài báo

Các số liệu này xuất hiện đồng nhất trong **Section 0 (Abstract)**, **Section 1 (Introduction)**, **Section 4 (Results)**, và **Section 6 (Conclusion)**:

| Chỉ số trong bài báo | Giá trị công bố | File dữ liệu nguồn | Key / Cột đối chứng | Lệnh kiểm tra nhanh |
|---|:---:|---|---|---|
| **Mean $\Delta\mathrm{CPC}$** | `+0.00354` (`0.0035395`) | [`results/5fold_results.json`](../results/5fold_results.json) | `rq1_delta_r.city.delta_cpc_inter.mean` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['delta_cpc_inter']['mean'])"` |
| **95% Bootstrap CI** | `[+0.0026, +0.0045]` | [`results/5fold_results.json`](../results/5fold_results.json) | `rq1_delta_r.city.delta_cpc_inter.ci_95_lower`, `ci_95_upper` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['delta_cpc_inter']['ci_95_lower'], d['rq1_delta_r']['city']['delta_cpc_inter']['ci_95_upper'])"` |
| **Median $\Delta\mathrm{CPC}$** | `+0.00195` (`0.0019531`) | [`results/5fold_results.json`](../results/5fold_results.json) | `rq1_delta_r.city.delta_cpc_inter.median` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['delta_cpc_inter']['median'])"` |
| **Tỷ lệ thắng (Win Rate)** | `45 / 50 (90.0%)` | [`results/5fold_results.json`](../results/5fold_results.json) | `rq1_delta_r.city.p_improved` (=0.9) | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(sum(x['delta_city']>0 for x in d['city_level_results']))"` |
| **Wilcoxon test (2-sided)** | `1.93 x 10^-9` | [`results/5fold_results.json`](../results/5fold_results.json) | `rq1_delta_r.city.wilcoxon_two_sided_p` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['wilcoxon_two_sided_p'])"` |
| **Rank-biserial $r_{\mathrm{rb}}$** | `0.8698` | [`results/5fold_results.json`](../results/5fold_results.json) | `rq1_delta_r.city.rank_biserial_r` | `python -c "import json; d=json.load(open('results/5fold_results.json')); print(d['rq1_delta_r']['city']['rank_biserial_r'])"` |

---

## 2. Bảng số liệu chính trong bài báo (Tables 1 - 5 / Bảng 1 - 5)

> [!NOTE]
> Bản thảo hiện tại sử dụng hệ thống đánh số 5 bảng chính thức trong bài báo (Bảng 1–5 / Tables 1–5): Bảng 1 dành riêng cho bảng ký hiệu và dữ liệu đầu vào. Vì vậy, các bảng thực nghiệm được đánh số từ 2 đến 5. Các phân tích về nhiễu TV, model seeds và cơ chế $d_{\text{pre}}$ được trình bày dưới dạng văn bản và hình vẽ (xem [Mục 3](#3-các-phân-tích-cốt-lõi-trong-văn-bản-và-hình-vẽ-narrative--figures)). Thí nghiệm Direct-OD (OD-FE) không còn trong bản thảo chính mà được lưu trữ như phân tích kiểm toán bổ trợ.

### Table 1 / Bảng 1: Ký hiệu cốt lõi, nguồn dữ liệu và trạng thái sẵn có của thông tin
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L46-L65`](full_paper_vi.md#L46-L65) (Bảng 1)
  - Tiếng Anh: [`full_paper_en.md:L57-L76`](full_paper_en.md#L57-L76) (Table 1)
* **Mô tả:** Bảng định nghĩa các ký hiệu toán học ($c, \mathcal{V}_c, t_{c,ij}, d_{c,ij}, \mathcal{P}_c, \Omega_c, I_b, K, Y_{c,b}, Y_{D,c}, \widehat{t}_{c,ij}^{(0)}$), nguồn dữ liệu (LODES 2019, US Census TIGER/Line 2019, 50 MSAs) và trạng thái sẵn có của thông tin ở các bước zero-shot và oracle calibration.

---

### Table 2 / Bảng 2: Benchmark tái tạo luồng chính với Urban GNN ($K=8, N=50$)
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L229-L236`](full_paper_vi.md#L229-L236) (Bảng 2)
  - Tiếng Anh: [`full_paper_en.md:L210-L217`](full_paper_en.md#L210-L217) (Table 2)
* **File dữ liệu nguồn:**
  1. [`results/5fold_results.json`](../results/5fold_results.json)
  2. [`results/audit/dpre_mechanism_data.csv`](../results/audit/dpre_mechanism_data.csv)
  3. [`results/PROTOCOL_LOCK.md`](../results/PROTOCOL_LOCK.md)
* **Mapping chi tiết từng cell:**
  - Zero-Shot $M_0$ CPC ($0.71281 \pm 0.04434$, Median $0.71632$): `rq1_delta_r.city.m0_cpc_inter` (`mean`, `std`, `median`).
  - Calibrated $M_1$ CPC ($0.71635 \pm 0.04454$, Median $0.71988$): `rq1_delta_r.city.m1_cpc_inter` (`mean`, `std`, `median`).
  - Mean $\Delta\mathrm{CPC} = +0.00354$, CI $[+0.0026, +0.0045]$, Win rate $45/50$ ($90.0\%$), $p=1.93\times 10^{-9}$.

---

### Table 3 / Bảng 3: Tính đặc thù mục tiêu & Placebo controls
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L251-L270`](full_paper_vi.md#L251-L270) (Bảng 3)
  - Tiếng Anh: [`full_paper_en.md:L224-L234`](full_paper_en.md#L224-L234) (Table 3)
* **File dữ liệu nguồn chuẩn duy nhất:**
  - [`../results/unified_placebo_v1/unified_placebo_per_city.csv`](../results/unified_placebo_v1/unified_placebo_per_city.csv)
  - [`../results/unified_placebo_v1/unified_placebo_reconciled_summary.json`](../results/unified_placebo_v1/unified_placebo_reconciled_summary.json)
* **Thông tin provenance & Checkpoint hash:**
  - **Frozen run checkpoint**: Được tạo tại commit `a780b1d` (2026-08-24).
  - **Checkpoint SHA256 (ví dụ Fold 1 Seed 1)**: `705f751172a71b45d3b853e77c9e01a0c91767f341e228a93fa6b8164bd3fa33`.
  - **Baseline zero-shot CPC**: $0.712807 \pm 0.04434$ (khớp $100\%$ Bảng 2 và toàn bài).
  - **Target $\Delta\mathrm{CPC}$**: $+0.003539$ ($45/50$ positive, $p = 1.93 \times 10^{-9}$).
* **LƯU Ý QUAN TRỌNG VỀ RERUN**:
  - `placebo_matched_v2` was a methodologically equivalent rerun generated from a different realization of the baseline checkpoints (mean baseline CPC 0.713623).
  - It was not used in the manuscript and was removed from the active repository to prevent accidental mixing with the frozen manuscript run.
  - **Canonical placebo artifact:** `results/unified_placebo_v1/unified_placebo_per_city.csv`
  - **Canonical baseline mean CPC:** 0.712807
  - **Historical script/artifact recovery:** Git commit `021cf7a` (`src/experiment/run_placebo_matched_v2.py`), and `results.zip` (contains original `results/placebo_matched_v2/` archive).
* **Mapping chi tiết từ `unified_placebo_per_city.csv`:**
  - **Phần A / Row 1 (Oracle Target $Y_D$):** `d_cpc_target` ($+0.003539$, CI $[+0.00260, +0.00450]$, Wilcoxon 2-sided $p=1.93\times 10^{-9}$, Win rate $45/50$).
  - **Phần A / Row 2 (Dose-Matched Training Donors):** `d_cpc_matched` (Mean: $-0.000091$, CI $[-0.00089, +0.00071]$, Wilcoxon 2-sided $p=0.4097$, Win rate $19/50$).
  - **Phần A / Row 3 (Dose-Matched Train-Mean):** `d_cpc_matched_train_mean` ($+0.000914$, CI $[+0.00001, +0.00186]$, Wilcoxon 2-sided $p=0.4319$, Median $+0.00007$, 27/50 dương và 23/50 âm).
  - **Phần A / Row 4 (Permuted Target $Y_D$):** `d_cpc_perm` ($-0.006964$, CI $[-0.00914, -0.00512]$, Wilcoxon 2-sided $p=1.78\times 10^{-15}$, Win rate $0/50$).
  - **Phần B / Specificity contrasts ($Target - Placebo$):**
    - vs Training Donors: $+0.003630$, CI $[+0.00287, +0.00445]$, Wilcoxon 1-sided $p=2.19\times 10^{-11}$, Win: $46/50$ ($92.0\%$).
    - vs Fold Train-Mean: $+0.002626$, CI $[+0.00197, +0.00336]$, Wilcoxon 1-sided $p=4.03\times 10^{-11}$, Win: $47/50$ ($94.0\%$).
    - vs Permuted Target: $+0.010504$, CI $[+0.00843, +0.01279]$, Wilcoxon 1-sided $p=1.78\times 10^{-15}$, Win: $49/50$ ($98.0\%$).
  - **Raw Test Donors (In-Fold, đối chiếu kiểm tra):** `d_cpc_raw_test_exact` ($-0.037721$, $p=8.88\times 10^{-16}$, Win: $50/50$).

---

### Table 4 / Bảng 4: Thang độ phân giải khoảng cách ($K \in \{2, 4, \dots, 20\}$)
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L279-L294`](full_paper_vi.md#L279-L294) (Bảng 4)
  - Tiếng Anh: [`full_paper_en.md:L241-L256`](full_paper_en.md#L241-L256) (Table 4)
* **File dữ liệu nguồn:**
  1. [`results/k_sensitivity_v1/k_sensitivity_summary.json`](../results/k_sensitivity_v1/k_sensitivity_summary.json)
  2. [`results/k_sensitivity_v1/k_sensitivity_per_city.csv`](../results/k_sensitivity_v1/k_sensitivity_per_city.csv)
  3. [`results/k_sensitivity_v1/k_sensitivity_raw.csv`](../results/k_sensitivity_v1/k_sensitivity_raw.csv)
* **Mapping chi tiết:**
  - `summary.json -> summary`: Mảng chứa thông số cho từng $K$:
    - $K=2$: Mean $+0.00098$, Median $+0.00034$, CI $[+0.00052, +0.00151]$, Win: $39/50$ ($78.0\%$), Gain/bin: $0.000488$
    - $K=4$: Mean $+0.00198$, Median $+0.00088$, CI $[+0.00125, +0.00279]$, Win: $39/50$ ($78.0\%$)
    - $K=6$: Mean $+0.00289$, Median $+0.00152$, CI $[+0.00201, +0.00384]$, Win: $44/50$ ($88.0\%$)
    - $K=8$ (Cấu hình chính / Anchor): Mean $+0.00354$, Median $+0.00195$, CI $[+0.00262, +0.00447]$, Win: $45/50$ ($90.0\%$), Gain/bin: $0.000442$
    - $K=10$: Mean $+0.00413$, Median $+0.00235$, CI $[+0.00311, +0.00514]$, Win: $45/50$ ($90.0\%$)
    - $K=12$: Mean $+0.00480$, Median $+0.00288$, CI $[+0.00372, +0.00590]$, Win: $46/50$ ($92.0\%$)
    - $K=14$: Mean $+0.00538$, Median $+0.00373$, CI $[+0.00424, +0.00654]$, Win: $45/50$ ($90.0\%$)
    - $K=16$: Mean $+0.00574$, Median $+0.00433$, CI $[+0.00455, +0.00694]$, Win: $46/50$ ($92.0\%$)
    - $K=18$: Mean $+0.00603$, Median $+0.00458$, CI $[+0.00480, +0.00726]$, Win: $47/50$ ($94.0\%$)
    - $K=20$: Mean $+0.00639$, Median $+0.00494$, CI $[+0.00508, +0.00769]$, Win: $46/50$ ($92.0\%$), Gain/bin: $0.000319$

---

### Table 5 / Bảng 5: Tính tổng quát qua các kiến trúc backbone
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L318-L327`](full_paper_vi.md#L318-L327) (Bảng 5)
  - Tiếng Anh: [`full_paper_en.md:L283-L292`](full_paper_en.md#L283-L292) (Table 5)
* **File dữ liệu nguồn:**
  1. [`results/backbone_robustness_results.json`](../results/backbone_robustness_results.json)
  2. [`results/mlp_backbone_results.json`](../results/mlp_backbone_results.json)
  3. [`results/mlp_backbone_execution.log`](../results/mlp_backbone_execution.log)
* **Mapping chi tiết:**
  - **Urban GNN (Message passing):** Mean $+0.00354$, CI $[+0.0026, +0.0045]$, Win $45/50$ ($90.0\%$), Wilcoxon $p=1.93\times 10^{-9}$.
  - **Pairwise Node MLP (No graph message passing):** Mean $+0.00329$, CI $[+0.0025, +0.0042]$, Win $47/50$ ($94.0\%$), Wilcoxon $p=4.38\times 10^{-11}$.
  - **Classical Gravity (Two-parameter):** Mean $+0.00084$, CI $[+0.0002, +0.0016]$, Win $22/50$ ($44.0\%$), Wilcoxon $p=0.3545$ (n.s.).

---

## 3. Các phân tích cốt lõi trong văn bản và hình vẽ (Narrative & Figures)

Các phân tích sau được trình bày trực tiếp trong lời văn và đồ thị của bản thảo chính (thay vì lập bảng riêng):

### Phân tích A: Độ nhạy nhiễu Total Variation & Ngưỡng Crossover
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L303-L309`](full_paper_vi.md#L303-L309) (Mục 4.3 & Hình 5)
  - Tiếng Anh: [`full_paper_en.md:L268-L276`](full_paper_en.md#L268-L276) (Section 4.3 & Figure 5)
* **File dữ liệu nguồn:**
  1. [`results/noise_robustness_fine_v1/noise_summary.json`](../results/noise_robustness_fine_v1/noise_summary.json)
  2. [`results/noise_robustness_fine_v1/noise_per_city.csv`](../results/noise_robustness_fine_v1/noise_per_city.csv)
  3. [`results/noise_robustness_fine_v1/noise_raw.csv`](../results/noise_robustness_fine_v1/noise_raw.csv)
* **Mapping chi tiết:**
  - `noise_summary.json -> eps_cross_zero_dCPC`: Ngưỡng crossover thực nghiệm $= 0.044439$ ($4.44\%$, CI: $[4.16\%, 4.77\%]$).
  - Ngưỡng còn ý nghĩa thống kê: $\epsilon^* = 0.03$ ($3.0\%$ TV, $p_{\mathrm{raw}}=0.0149$, $p_{\mathrm{Holm}}=0.0446 < 0.05$).
  - Liều-đáp ứng theo $\epsilon$:
    - $\epsilon=0.00$: Mean $+0.00354$, Positives: $45/50$
    - $\epsilon=0.01$: Mean $+0.00336$, Positives: $44/50$
    - $\epsilon=0.02$: Mean $+0.00282$, Positives: $36/50$
    - $\epsilon=0.03$: Mean $+0.00193$, Positives: $28/50$
    - $\epsilon=0.04$: Mean $+0.00070$, Positives: $18/50$ (Wilcoxon 1-sided $p_{\mathrm{raw}}=0.4847$, $p_{\mathrm{Holm}}=0.9695$, n.s.)
    - $\epsilon=0.05$: Mean $-0.00087$, Positives: $17/50$

---

### Phân tích B: Độ ổn định qua các model seeds (Seeds 1, 10, 100)
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L314-L315`](full_paper_vi.md#L314-L315) (Mục 4.4 text)
  - Tiếng Anh: [`full_paper_en.md:L281`](full_paper_en.md#L281) (Section 4.4 text)
* **File dữ liệu nguồn:**
  1. [`results/k_sensitivity_v1/k_sensitivity_per_seed.csv`](../results/k_sensitivity_v1/k_sensitivity_per_seed.csv)
  2. [`results/sampling_robustness_v1/sampling_per_seed.csv`](../results/sampling_robustness_v1/sampling_per_seed.csv)
* **Mapping chi tiết:**
  - Seed 1: Mean $\Delta\mathrm{CPC} = +0.00434$, Median $+0.00207$, CI $[+0.00322, +0.00547]$, Win: $41/50$ ($82\%$).
  - Seed 10: Mean $\Delta\mathrm{CPC} = +0.00308$, Median $+0.00182$, CI $[+0.00216, +0.00404]$, Win: $44/50$ ($88\%$).
  - Seed 100: Mean $\Delta\mathrm{CPC} = +0.00320$, Median $+0.00217$, CI $[+0.00236, +0.00408]$, Win: $44/50$ ($88\%$).
  - Dải giá trị qua seeds: $+0.0031$ đến $+0.0043$, Across-seed SD $= 0.00070$.

---

### Phân tích C: Phân tích cơ chế lệch khoảng cách ban đầu ($d_{\text{pre}}$)
* **Vị trí trong bài báo:** 
  - Tiếng Việt: [`full_paper_vi.md:L332-L338`](full_paper_vi.md#L332-L338) (Mục 4.5 & Hình 6)
  - Tiếng Anh: [`full_paper_en.md:L295-L301`](full_paper_en.md#L295-L301) (Section 4.5 & Figure 6)
* **File dữ liệu nguồn:**
  1. [`results/audit/dpre_mechanism_data.csv`](../results/audit/dpre_mechanism_data.csv)
  2. [`results/audit/dpre_mechanism_summary.json`](../results/audit/dpre_mechanism_summary.json)
  3. [`results/intra_bin_mechanism_diagnostic.json`](../results/intra_bin_mechanism_diagnostic.json)
* **Mapping chi tiết:**
  - Bivariate Pearson: $r = +0.7995$, $p = 3.36 \times 10^{-12}$.
  - Bivariate Spearman: $\rho = +0.7464$, $p = 4.92 \times 10^{-10}$.
  - Full Partial Correlation ($M_0 + \log N_{\text{pairs}} + \log N_{\text{tracts}} + \text{MeanDist}$): $r_{\mathrm{partial}} = +0.7951$, $p = 5.35 \times 10^{-12}$.
  - Multivariate OLS $\beta(d_{\mathrm{pre}}) = +0.1487$, $p = 4.12 \times 10^{-11}$, $R^2 = 73.7\%$.

---

### Phân tích D (Bổ trợ/Audit): So sánh vận hành với quan sát cặp OD trực tiếp (OD-FE)
* **Vị trí:** Phân tích kiểm toán lưu trữ trong codebase (không xuất hiện trong bảng chính của bản thảo hiện tại).
* **File dữ liệu nguồn:**
  1. [`results/direct_od_equivalence_v1/combined/summary.json`](../results/direct_od_equivalence_v1/combined/summary.json)
  2. [`results/partial_od_equivalence_v2/combined/summary.json`](../results/partial_od_equivalence_v2/combined/summary.json)
  3. [`results/direct_od_equivalence_v1/audit_report.json`](../results/direct_od_equivalence_v1/audit_report.json)
* **Mapping chi tiết:**
  - Điểm giao cắt nội suy: $p_{\mathrm{eq}} \approx 0.20\%$ (CI: $[0.133\%, 0.287\%]$), tại đó Direct OD Gain $\approx +0.00354$, tương đương khoảng 35 cặp OD lộ diện.
  - Tỷ lệ $p=0.10\%$: $\Delta\mathrm{CPC} = +0.00180$ (thấp hơn $Y_D$ với $D=-0.00174$).
  - Tỷ lệ $p=0.25\%$: $\Delta\mathrm{CPC} = +0.00448$ (cao hơn $Y_D$ với $D=+0.00094$).

---

## 4. Biểu đồ & Hình ảnh (Figures 1 - 6 / Hình 1 - 6)

| Biểu đồ (EN / VI) | Tên & Mô tả | File hình ảnh | File dữ liệu nguồn | Vị trí trong bản thảo | Script tái tạo |
|---|---|---|---|---|---|
| **Figure 1 / Hình 1** | Oracle calibration framework schematic | [`figures/fig1_oracle_calibration_framework.svg`](figures/fig1_oracle_calibration_framework.svg) | Sơ đồ phương pháp luận | EN: `L157-L160`, VI: `L170-L173` | Vẽ vector SVG trực tiếp |
| **Figure 2 / Hình 2** | City-level $\Delta\mathrm{CPC}$ across 50 test cities | [`figures/fig2_main_per_city.png`](figures/fig2_main_per_city.png) ([.pdf](figures/fig2_main_per_city.pdf)) | [`results/5fold_results.json`](../results/5fold_results.json) | EN: `L205-L208`, VI: `L225-L228` | [`scripts/generate_all_paper_figures.py`](../scripts/generate_all_paper_figures.py) (`generate_figure2()`) |
| **Figure 3 / Hình 3** | Matched placebo controls (Authentic vs Donor vs Permuted) | [`figures/fig3_structural_validity_placebo.png`](figures/fig3_structural_validity_placebo.png) ([.pdf](figures/fig3_structural_validity_placebo.pdf)) | [`results/unified_placebo_v1/unified_placebo_per_city.csv`](../results/unified_placebo_v1/unified_placebo_per_city.csv) | EN: `L221-L223`, VI: `L246-L250` | [`scripts/generate_all_paper_figures.py`](../scripts/generate_all_paper_figures.py) (`generate_figure3()`) |
| **Figure 4 / Hình 4** | Resolution sensitivity (K-sweep & County) | [`figures/fig4_resolution_sensitivity.png`](figures/fig4_resolution_sensitivity.png) ([.pdf](figures/fig4_resolution_sensitivity.pdf)) | [`results/k_sensitivity_v1/k_sensitivity_summary.json`](../results/k_sensitivity_v1/k_sensitivity_summary.json) & [`results/spatial_resolution/spatial_resolution_summary.json`](../results/spatial_resolution/spatial_resolution_summary.json) | EN: `L257-L259`, VI: `L296-L298` | [`scripts/generate_all_paper_figures.py`](../scripts/generate_all_paper_figures.py) (`generate_figure4()`) |
| **Figure 5 / Hình 5** | Noise dose-response & TV crossover | [`figures/fig5_noise_dose_response.png`](figures/fig5_noise_dose_response.png) ([.pdf](figures/fig5_noise_dose_response.pdf)) | [`results/noise_robustness_fine_v1/noise_summary.json`](../results/noise_robustness_fine_v1/noise_summary.json) | EN: `L274-L276`, VI: `L307-L309` | [`scripts/generate_all_paper_figures.py`](../scripts/generate_all_paper_figures.py) (`generate_figure5()`) |
| **Figure 6 / Hình 6** | Mechanistic scatter: $d_{\mathrm{pre}}$ vs $\Delta\mathrm{CPC}$ | [`figures/fig6_mechanistic_dpre.png`](figures/fig6_mechanistic_dpre.png) ([.pdf](figures/fig6_mechanistic_dpre.pdf)) | [`results/audit/dpre_mechanism_data.csv`](../results/audit/dpre_mechanism_data.csv) | EN: `L299-L301`, VI: `L336-L338` | [`scripts/generate_all_paper_figures.py`](../scripts/generate_all_paper_figures.py) (`generate_figure6()`) |

---

## 5. Tập lệnh kiểm tra nhanh (Automated Verification Scripts)

### Cách 1: Chạy script tự động kiểm tra toàn bộ (Khuyên dùng)
Chạy lệnh sau tại thư mục gốc của repository để in ra báo cáo đối chiếu toàn bộ các chỉ số của Paper với dữ liệu thực nghiệm trong `results/`:
```bash
python scripts/verify_all_numbers.py
```

### Cách 2: Các lệnh kiểm tra từng phần (One-line Snippets)
```bash
# 1. Kiểm tra Table 2 / Bảng 2 & Abstract core numbers
python -c "import json; d=json.load(open('results/5fold_results.json'))['rq1_delta_r']['city']; print(f'Mean: {d[\"delta_cpc_inter\"][\"mean\"]:.5f}, Median: {d[\"delta_cpc_inter\"][\"median\"]:.5f}, WinRate: {d[\"p_improved\"]*100:.1f}%, p: {d[\"wilcoxon_two_sided_p\"]:.2e}')"

# 2. Kiểm tra Table 3 / Bảng 3 Placebo controls
python -c "import pandas as pd; df=pd.read_csv('results/unified_placebo_v1/unified_placebo_per_city.csv'); print(f'Target: {df[\"d_cpc_target\"].mean():.6f}, Donor mean: {df[\"d_cpc_matched\"].mean():.6f}, Train-mean: {df[\"d_cpc_matched_train_mean\"].mean():.6f}, Permuted: {df[\"d_cpc_perm\"].mean():.6f}')"

# 3. Kiểm tra Table 4 / Bảng 4 K-sweep range
python -c "import json; k=json.load(open('results/k_sensitivity_v1/k_sensitivity_summary.json'))['summary']; print(f'K=2: {k[0][\"mean_delta\"]:.5f}, K=8: {k[3][\"mean_delta\"]:.5f}, K=20: {k[-1][\"mean_delta\"]:.5f}')"

# 4. Kiểm tra Section 4.3 & Figure 5 / Hình 5 Noise crossover threshold
python -c "import json; n=json.load(open('results/noise_robustness_fine_v1/noise_summary.json')); print(f'TV Crossover: {n[\"eps_cross_zero_dCPC\"]*100:.2f}%')"

# 5. Kiểm tra Section 4.5 & Figure 6 / Hình 6 Mechanism correlation
python -c "import pandas as pd, scipy.stats as st; df=pd.read_csv('results/audit/dpre_mechanism_data.csv'); print(f'Pearson r: {st.pearsonr(df[\"d_pre_tv\"], df[\"delta_cpc\"])[0]:.4f}, Spearman rho: {st.spearmanr(df[\"d_pre_tv\"], df[\"delta_cpc\"])[0]:.4f}')"
```
