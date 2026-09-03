### Supplementary Table S1. Descriptive city-level results for the multi-county spatial-resolution subset

City-level comparison of the zero-shot baseline ($M_0$), city-level oracle calibration ($M1_{\mathrm{city}}$), and origin-county-conditioned oracle calibration ($M1_{\mathrm{county}}$) for the 11 metropolitan datasets containing tracts assigned to more than one county. The resolution increment is defined as $\Delta\mathrm{CPC}_{\mathrm{res},c}=\mathrm{CPC}(M1_{\mathrm{county}})-\mathrm{CPC}(M1_{\mathrm{city}})$. Values are descriptive city-level estimates. No subgroup confidence interval or hypothesis test is reported unless supported by a separately verified uncertainty artifact.

*(Tiếng Việt: **Bảng bổ sung S1. Kết quả mô tả theo thành phố cho nhóm phân tích độ phân giải không gian đa county.** Bảng so sánh zero-shot baseline ($M_0$), hiệu chỉnh oracle cấp city ($M1_{\mathrm{city}}$) và hiệu chỉnh oracle có điều kiện theo origin-county ($M1_{\mathrm{county}}$) cho 11 bộ dữ liệu đô thị có các tract được gán vào nhiều hơn một county. Mức tăng do độ phân giải được định nghĩa là $\Delta\mathrm{CPC}_{\mathrm{res},c}=\mathrm{CPC}(M1_{\mathrm{county}})-\mathrm{CPC}(M1_{\mathrm{city}})$. Các giá trị là ước lượng mô tả ở cấp city. Không báo cáo khoảng tin cậy hoặc kiểm định giả thuyết cho subgroup nếu không có artifact bất định riêng đã được xác minh.)*

| City | Origin counties | $M_0$ CPC | $M1_{\mathrm{city}}$ CPC | $M1_{\mathrm{county}}$ CPC | $\Delta\mathrm{CPC}_{\mathrm{city}}$ | $\Delta\mathrm{CPC}_{\mathrm{county}}$ | $\Delta\mathrm{CPC}_{\mathrm{res}}$ |
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
| **Multi-county mean** | — | — | — | — | — | — | **+0.000626** |
| **Positive resolution gains** | — | — | — | — | — | — | **9 / 11** |

*Note: Rows are sorted by $\Delta\mathrm{CPC}_{\mathrm{res}}$ in descending order. County labels are assigned from tract centroids using GADM 4.1 and group OD pairs by the county of the origin tract. Destination tracts may belong to the same or another county represented within the city dataset. Prediction and evaluation remain city-wide on the same known positive support. The 39 single-county cities are omitted from this table because $M1_{\mathrm{county}}\equiv M1_{\mathrm{city}}$ by construction. Results are seed-averaged across model seeds $\\{1, 10, 100\\}$.*