# DBD calibration under limited source OD supervision — focused protocol v2

Ngày sửa: 2026-10-06.

Trạng thái: **ĐÃ CHỐT THIẾT KẾ theo xác nhận của người dùng ngày 2026-10-06**, bao gồm 60% source test cố định. CHƯA triển khai vào runner và CHƯA có kết quả v2. Tài liệu này là nguồn cấu hình khoa học chính thức cho triển khai v2 và thay thế phạm vi thực nghiệm của new_plan trước đây. Bản trước khi sửa được lưu nguyên trạng ở `new_plan_before_compute_revision_2026-10-06.md` để truy xuất các đặc tả kỹ thuật không bị v2 thay thế. Không áp dụng lẫn hai protocol trong một run.

### Hợp đồng triển khai đã chốt

- Protocol ID: `new_plan_v2_focused`; main fraction 0.40; scarcity fractions {0.10, 0.20, 0.30, 0.40}.
- Source test là suffix sau Train_40 của cùng permutation và không đổi theo fraction. OD không thuộc active Train_f hoặc Test_source bị bỏ khỏi supervised loss và source evaluation; không tự tăng test lên 70/80/90%.
- Main comparison: raw zero-shot và calibration bằng normalized DBD từ toàn bộ positive interzonal OD của target; main K=8, TV=0.
- Model coverage, seeds, loss, controls, thống kê và output counts phải theo các mục dưới. Không tự thêm grid hoặc đổi backbone khi triển khai.
- Tài liệu chốt thiết kế không có nghĩa runner hiện tại đã tương thích. Chỉ chạy scientific v2 sau migration và smoke QA ở §8. Nếu đặc tả kỹ thuật còn thiếu hoặc mâu thuẫn, báo rõ vấn đề trước khi chạy; không tự điền bằng lựa chọn làm thay đổi thí nghiệm.

## 1. Mục tiêu và phạm vi kết luận

Câu hỏi chính: **DBD tổng hợp của thành phố đích có cải thiện dự báo OD của mô hình đã đóng băng, và mức cải thiện thay đổi thế nào khi lượng OD supervision của thành phố nguồn giảm?**

- Thành phố A: train với 40%, 30%, 20% hoặc 10% positive interzonal OD; test trong A trên cùng một tập 60% cố định.
- Thành phố B: mô hình A dự báo zero-shot; DBD được trích từ 100% positive interzonal OD của B; calibration và đánh giá trên cùng support B.
- Lặp cho 50 source cities, mỗi source chuyển sang 49 targets còn lại.
- Backbone không nhận target OD labels, không fine-tune, không chọn checkpoint/hyperparameters theo target metrics.
- Calibrator chỉ nhận normalized target DBD, không nhận individual target flows hoặc true total volume.

Đây là **support-conditioned OD intensity reconstruction với oracle aggregate target DBD**. DBD chứa thông tin tổng hợp của ground-truth target đang đánh giá, nên calibrated output không phải dự báo hoàn toàn không có thông tin target. Chưa kiểm chứng external DBD, dự báo tương lai, zero/nonzero link discovery hoặc full-matrix reconstruction.

Không mặc định DBD luôn cải thiện hoặc gain tăng đơn điệu khi dữ liệu nguồn giảm.

## 2. Các thay đổi so với v1

| Thành phần | v1 | v2 |
|---|---|---|
| Main training fraction | 30% | 40% |
| Scarcity fractions | 10/20/30/50/100% | 10/20/30/40% |
| Source test | Main 70%, chỉ ở 30% | 60% cố định ở mọi fraction |
| Model coverage | Ba họ ở mọi fraction | GNN + Gravity ở mọi fraction; MLP chỉ 40% |
| DBD sensitivity | Full K × error grid | Hai sweep riêng ở 40%, chỉ GNN |
| Noise realizations | 20 mỗi error level | 5 mỗi error level |
| Specificity | Ba họ mô hình | GNN ở 40% |
| Gap recovery | So với 100% supervision | Bỏ khỏi phân tích bắt buộc |

40% được chọn theo thiết lập người dùng, không dựa trên kết quả quan sát. MLP ở 40% kiểm tra tính tổng quát của added value, không hỗ trợ kết luận scarcity cho MLP.

## 3. Support, source split và preprocessing

### 3.1. Support cố định

Với mỗi city c, xác định một lần:

`Omega_c = {(i,j): origin != destination, distance_km > 0, true_flow >= 1}`.

Training, source test, target DBD và target evaluation đều dùng positive interzonal support này. Không biến OD thiếu thành zero; không thay support theo dự báo. Oracle positive support phải được công khai trong paper.

### 3.2. Nested split và test cố định

1. Canonical sort theo `(origin, destination)`; kiểm tra pair keys duy nhất.
2. Sinh một permutation với split seed 42 cho mỗi city; lưu manifest, không phụ thuộc model seed.
3. Với N = số pairs trên Omega_c, `Train_f = permutation[:floor(f*N)]` cho f trong {0.10, 0.20, 0.30, 0.40}.
4. `Test_source = permutation[floor(0.40*N):]`, dùng nguyên trạng ở cả bốn fractions.
5. Các pairs trong Train_40 nhưng ngoài Train_f không tham gia supervised loss ở fraction f và không chuyển vào test.

| Active training | Source test cố định | Pairs không dùng cho loss/test (xấp xỉ) |
|---|---|---|
| 40% | 60% | 0% |
| 30% | 60% | 10% |
| 20% | 60% | 20% |
| 10% | 60% | 30% |

Fractions tính trên toàn bộ Omega_s, không tính trên riêng pool 40%. Số pairs thực tế dùng floor như trên. Test cố định giúp đánh giá cùng OD pairs giữa fractions; source test là diagnostic, target evaluation là kết quả chính. Không bắt buộc báo cáo thêm source test theo complement 100%-f.

Bắt buộc `Train_10 <= Train_20 <= Train_30 <= Train_40` và mọi Train_f disjoint với Test_source. Khi floor gây trùng kích thước trên city nhỏ, ghi nhận số pairs thực tế; fail nếu training hoặc test rỗng.

Test_source chỉ dùng báo cáo chẩn đoán sau training, không chọn epoch/checkpoint/model. Giữ fixed 40 epochs của master v1 để không cần một validation split mới. Nếu muốn early stopping phải sửa protocol trước khi chạy; không dùng test 60% làm validation.

### 3.3. Preprocessing và bins

- Giữ node feature schema, source-wide node scaler, geography-only graph radius 5 km và self-loops của master v1.
- Chỉ thay supervised label subset giữa fractions. Không chuyển source graph sang target.
- Fit distance scaler và P99 distance cap từ Train_40; freeze ở mọi fraction. Chỉ dùng khoảng cách, không dùng flow magnitudes từ Train_40 ngoài active Train_f.
- Đây là **label-scarcity**, không phải thiếu quyền truy cập geography/pair distances: reference distances ở 40% vẫn có sẵn khi f=10%.
- Giữ source-specific equal-width nominal bins v1: width = D_cap/K; bin cuối overflow đến infinity. Main K=8.
- Các scaler, bin edges, graph và feature schema không fit lại bằng target OD.

## 4. Backbone và calibration giữ nguyên

### 4.1. Backbone

- Primary model: `urban_gnn`, chạy ở cả bốn fractions.
- Cheap reference: `gravity_2param`, chạy ở cả bốn fractions.
- Architecture robustness: `pairwise_mlp`, chỉ chạy ở 40%.
- Neural seeds: {1, 10, 100}; cùng source split ở mọi seeds/models.
- Gravity giữ deterministic initialization và optimizer v1; fit và infer một lần mỗi source/fraction. Không tạo randomness hoặc chạy lại chỉ để có ba nhãn seeds.
- Giữ kiến trúc, feature schema và optimizer master v1: full-batch AdamW, lr=2e-3, 40 epochs, weight decay=1e-4, grad clip=5.0; các hyperparameters khác giữ theo metadata/config v1 và phải lưu đầy đủ.
- Master v1 đang dùng **Log1p-MSE**, không phải ZTNB. File `implement_new_plan/loss/ztnb.py` thuộc một đường code khác; v2 không đổi loss. Chuyển sang ZTNB là thay đổi backbone riêng và không trộn vào thí nghiệm này.
- Mỗi fraction train độc lập; không warm-start từ fraction lớn, không fine-tune giữa fractions. Freeze sau training.

### 4.2. Calibrator

Giữ nguyên toán tử support-conditioned pure DBD calibration của master v1 (`implement_new_plan/calibration/source_bins.py`), bao gồm bin mass ratio và exact predicted-volume preservation. Không thay bằng toán tử khác chỉ vì cùng tên DBD.

Trên active bins q_b > 0, giữ quy tắc v1 chuẩn hóa p về effective support và phân bổ lại cùng predicted total mass. Không suy ra active bins từ target flow. Báo cáo target DBD mass bị loại do baseline không có mass.

Mandatory invariants: predictions hữu hạn và không âm; normalized p/q tổng bằng 1; identity nếu p=q; matched effective-bin distribution sau pure calibration; predicted total volume bảo toàn theo tolerance v1. NaN/Inf hoặc domain failure phải raise có context, không bỏ city hoặc silently clip predictions.

True target total chỉ được evaluator dùng cho `R_vol`, không rescale baseline/calibrated output. Báo cáo CPC raw, CPC_norm, MAE, RMSE và R_vol; CPC_norm dùng truth normalization chỉ trong evaluator.

## 5. Ma trận thực nghiệm tối thiểu

| Arm | Models | Fractions | DBD conditions | Mục đích |
|---|---|---|---|---|
| A: Main added value | GNN, Gravity, MLP | 40% | raw vs correct target DBD; K=8, TV=0 | DBD có ích không? |
| B: Source scarcity | GNN, Gravity | 10/20/30/40% | raw vs correct target DBD; K=8, TV=0 | Gain đổi thế nào khi supervision giảm? |
| C1: Resolution check | GNN | 40% | K={4,8,12}, TV=0 | Hiệu ứng có phụ thuộc duy nhất K=8 không? |
| C2: Noise check | GNN | 40% | K=8, TV={0,0.05,0.10} | Kiểm tra độ bền cơ bản với DBD sai số |
| D: Specificity control | GNN | 40% | K=8, dose-matched donor DBD | Thông tin đúng target có giá trị hơn control không? |

- A và B tại 40% là cùng kết quả; không train/infer/calibrate lặp lại.
- C1/C2 baseline K=8, TV=0 dùng lại A; không chạy K × TV interaction.
- C2: 5 deterministic realizations mỗi TV>0; giữ exact-TV generator v1 và kiểm tra TV thực đạt. Không dùng 5 realizations như 5 independent cities. Báo cáo Monte Carlo variability; đây là kiểm tra phụ, không kết luận đầy đủ về measurement-error distribution.
- D giữ cyclic donor mapping, effective-support dose definition, Brent matching và failure handling v1. Không chọn donor theo CPC. Donor control là một oracle aggregate intervention bổ sung; công khai provenance.
- Mọi arm vẫn dùng đủ 50 sources × 49 targets; không chọn subset cities theo gain quan sát.
- Không chạy mặc định: full-supervision reference, gap-recovery, county/subzone calibration, partial-target-OD, trip-sampling, spatial resolution, K×error interaction hoặc additional backbones. Các câu hỏi này nằm ngoài v2.

## 6. Outcomes và thống kê tập trung

### 6.1. Outcomes

Primary: paired `gain_f = CPC_calibrated_f - CPC_raw_f` trên target support.

Báo cáo CPC trước, sau, gain, median gain và tỷ lệ cải thiện để tránh gain lớn nhưng chất lượng cuối thấp. Diagnostic CPC_norm phân biệt thay đổi shape khỏi predicted-total mismatch. Source held-out CPC ở tập 60% báo cáo riêng, không gộp với cross-city CPC.

RQ2 primary contrast của GNN: `gain_10 - gain_40`, ghép cặp trên cùng source-target. Các mức 20% và 30% xây learning curve mô tả, không tạo toàn bộ all-pairs tests. Báo cáo thêm raw/calibrated performance ở mỗi fraction; không gọi chênh lệch là thay thế nhân quả OD supervision.

### 6.2. Phụ thuộc source-target

- Average noise realizations trong seed, rồi average neural seeds trong source-target; Gravity có một kết quả deterministic.
- Giữ 2.450 pairs để mô tả heterogeneity, không coi là 2.450 independent samples.
- Target summary: average gain từ 49 sources cho mỗi target; báo cáo 50 target summaries và source summaries.
- Primary dependence-aware analysis: crossed random-intercept model `gain ~ 1 + (1|source) + (1|target)` cho A, và tương tự trên paired contrast `gain_10 - gain_40` cho RQ2. Kiểm tra convergence/singularity, lưu diagnostics; nếu fail thì đánh dấu inference chưa hợp lệ, không tự thay phương pháp.
- City-target bootstrap (10.000 replicates, seed 42) và Wilcoxon giữ như descriptive/supporting analysis. Vì target summaries dùng chung sources, không coi chúng là bằng chứng độc lập thay cho dependence-aware model. Vai trò source/target của cùng city cũng có thể tương quan; crossed model là approximation và phải nêu limitation.
- Một Holm family gồm 4 confirmatory contrasts: A gain của GNN, Gravity, MLP và B gain_10-minus-gain_40 của GNN. Báo cáo raw và Holm-adjusted p-values từ dependence-aware analyses.
- D và C là supporting analyses: effect sizes, uncertainty và diagnostics; không thêm confirmatory claims sau khi xem kết quả.
- B không fit pooled repeated-fraction model v1 như mặc định: primary endpoint contrast tránh bỏ sót within-pair repeated-measure dependence. Không giả định tuyến tính/đơn điệu.

## 7. Execution và tái sử dụng tính toán

### 7.1. Lịch chạy

1. Audit dữ liệu, support, nested split, source preprocessing và deterministic manifest hashes.
2. Chạy GNN/Gravity/MLP ở 40%; ghi source test và A; checkpoint + raw predictions.
3. Chạy GNN/Gravity ở 30/20/10%; ghi B. Mỗi fraction học độc lập.
4. C/D chỉ đọc artifacts GNN 40%; không train hoặc infer lại.
5. Tổng hợp và thống kê; certification sau QA. Smoke run trên 1 source/2 targets chỉ kiểm tra vận hành, không dùng để chọn cấu hình hay làm kết quả khoa học.

### 7.2. Cache và storage

- Load raw city/features/geographic graph một lần mỗi city; cache target graph không chứa OD flow edges.
- Fit source preprocessing một lần mỗi source và tái dùng giữa fractions/models theo §3.
- Train đúng một lần mỗi `(protocol, source, model, seed, fraction)`.
- GNN encode target nodes đúng một lần mỗi checkpoint/target; chunk pair decoder để giảm peak RAM/VRAM. Không chạy message passing lại theo từng pair chunk. Khả năng chunk full-batch training phải giữ đúng gradient của v1, không tự chuyển mini-batch optimization.
- Infer raw predictions một lần mỗi checkpoint/target; cache giữ precision tương đương v1, không tự giảm float precision hoặc dùng AMP.
- Cache target bin assignments và normalized ground-truth DBD theo `(source_bin_hash, target_support_hash, K)`; dùng chung models/seeds/fractions.
- Cache predicted bin masses theo checkpoint/target/K. Calibration/noise/donor dùng lại raw predictions; xử lý từng target, không giữ toàn bộ transfers trong RAM.
- Chỉ lưu pair-level raw predictions một lần; source truth giữ ở dữ liệu gốc; lưu calibrated metrics và bin factors thay vì toàn bộ calibrated vectors cho mỗi noise realization. Có thể tái tạo vectors bằng factors và raw predictions.
- Vectorize metric/calibration operations; exact-equivalence QA trước khi dùng đường optimized.
- Resume chỉ khi protocol, data/split/support hashes, checkpoint hyperparameters, feature/scaler/bin hashes, seed/fraction và calibration version khớp; atomic writes và completed-task index chống duplicated rows.
- Gravity dùng một physical compute; schema ghi `stochastic_training=false`, không coi duplicated seed labels là replicates.

### 7.3. Compute budget (không phải dự báo wall-clock)

v1 có 5 fractions × 50 sources × (3 GNN seeds + 3 MLP seeds + 1 deterministic Gravity) = **1.750 unique trainings**, nếu đã deduplicate Gravity.

v2:

| Model | Unique trainings | Unique target inference jobs |
|---|---:|---:|
| GNN: 4 fractions × 3 seeds × 50 sources | 600 | 29.400 |
| Gravity: 4 fractions × 50 sources | 200 | 9.800 |
| MLP: 40% × 3 seeds × 50 sources | 150 | 7.350 |
| Total | **950** | **46.550** |

Giảm khoảng **45,7% số jobs** so với v1 deduplicated (85.750 target inference jobs). GNN full-batch có chi phí khác MLP/Gravity và source cities khác kích thước, nên không suy ra giảm 45,7% wall-clock. Profile thời gian loading/graph/training/encoding/decoding/calibration/IO theo city để biết bottleneck thực tế.

Grid C v1: 5 K × (1 clean + 6 nonzero TV × 20 realizations) = **605 conditions** mỗi neural seed-transfer. v2 GNN: 3 clean K + 2 nonzero TV × 5 realizations = **13 conditions**, đã bao gồm main K=8; thêm 1 donor control thành 14. Main và noise-free conditions chỉ tính một lần. C/D không chạy cho hai model families khác. Counts này là calibration conditions, không phải số trainings.

## 8. Artifacts, QA và migration

Output root mới: `results/new_plan_v2_focused/`; không ghi đè historical outputs hoặc dùng chung v1 manifests đã thay đổi nội dung.

Required artifacts:

- `protocol.json`: version `new_plan_v2_focused`, exact config, data/code hashes, runtime/device.
- `source_split_manifest.csv`: pair keys, active training membership, fixed source-test flag, split seed.
- Source scalers/bins/checkpoint metadata; deterministic donor mapping.
- `source_test_metrics.csv`, `main_transfer_metrics.csv`, `scarcity_transfer_metrics.csv`, `resolution_metrics.csv`, `noise_metrics.csv`, `specificity_metrics.csv`.
- Raw prediction cache index; actual job timings và peak memory.
- Seed-averaged summaries, source/target summaries, contrasts, dependence-model diagnostics, corrected inference table.

Row QA trước aggregation:

- A: 2.450 × (3 GNN seeds + 3 MLP seeds + 1 Gravity) = **17.150 rows**.
- B: 2.450 × 4 fractions × (3 GNN seeds + 1 Gravity) = **39.200 rows**, gồm A GNN/Gravity 40% như derived view, không thực thi lại.
- Unique main/scarcity rows toàn bộ: **46.550**; main/scarcity overlap là 9.800.
- C/D conditions phải có đúng manifest keys; không duplicate clean/main conditions như independent observations.
- Mỗi target có đủ 49 sources; mỗi stochastic model có đủ 3 seeds; split disjointness, cache metadata và calibration invariants phải pass.

Implementation checklist trước scientific run:

1. Runner v2/config riêng; sửa fraction grid, model coverage, output schemas, source fixed test và reference preprocessing sang 40%.
2. Bỏ hard-coded v1 adjacent contrasts, 100%-reference gap recovery, full K×TV grid và old expected counts.
3. QA numerically equivalent optimized prediction/calibration, meaningful split/resume contracts và small smoke run.
4. Chỉ tạo execution-complete khi đủ computation; certified marker chỉ sau all scientific QA gates.

Historical v1 results vẫn là kết quả v1, không đổi nhãn thành v2. Cần retrain source models cho các cấu hình v2 vì source reference preprocessing/bins chuyển 30% sang 40% và main fraction đổi. Chỉ reuse raw data/geography artifacts nếu provenance khớp. Không mặc định reuse old checkpoints/cache; phải chứng minh metadata và numerical compatibility nếu muốn ngoại lệ.

## 9. Cách diễn giải cho bài báo

“We train source-city models with nested 10%, 20%, 30%, and 40% subsets of observed positive interzonal OD pairs, retaining a fixed 60% source-city test set. Frozen models are transferred to each of the other 49 cities. Post-hoc calibration receives a normalized distance distribution aggregated from all observed positive interzonal target OD flows, while pair-level target labels and target total volume are withheld from the calibrator. Our primary analysis evaluates calibration gains and their change between 10% and 40% source supervision; compact resolution, noise, and donor controls provide supporting checks.”

Conclusions phải giới hạn ở oracle aggregate calibration, known positive support và các model/fractions đã chạy. C1/C2 không đủ để suy ra optimal K, đầy đủ interaction, hoặc mọi loại measurement error. MLP chỉ kiểm chứng added value ở 40%; không tuyên bố scarcity response across all architectures.
