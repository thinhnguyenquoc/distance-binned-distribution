### Table 6: Diagnostic Analysis of Negative Marginal Gain Cities ($\Delta R^{\text{real},+} < 0, n=8$)

> **Key Finding**: In all 8 negative cities, ground-truth commuter trips are overwhelmingly short-distance ($>94\%$ in Bin 1, $<10\text{ km}$), while the Meta mobility prior severely over-allocates mass to medium/long-distance bins ($15\%–25\%$ in Bin 2/3), leading to lower distributional overlap (Mean $83.5\%$ vs $93.0\%$ in positive cities). When calibrated against the Oracle distribution ($M_1^{\text{oracle},+}$), **100% of these 8 cities improve**.

| Target City | Tracts ($N$) | Overlap | Real $\Delta R$ | Oracle $\Delta R$ | GT Bin Proportions $[b_1, b_2, b_3]$ | Meta Bin Proportions $[p_1, p_2, p_3]$ | Primary Factor |
|---|---|---|---|---|---|---|---|
| **Long_Beach** | 103 | 84.3% | **-0.0088** | +0.0169 | ['97.1%', '2.9%', '0.0%'] | ['81.4%', '17.0%', '1.7%'] | Compact geometry + Meta medium-bin overestimation |
| **Boston** | 178 | 82.4% | **-0.0159** | +0.0100 | ['96.6%', '3.4%', '0.0%'] | ['79.0%', '18.8%', '2.2%'] | Compact geometry + Meta medium-bin overestimation |
| **Baltimore** | 200 | 86.5% | **-0.0049** | +0.0083 | ['94.8%', '5.2%', '0.0%'] | ['81.2%', '17.2%', '1.6%'] | Compact geometry + Meta medium-bin overestimation |
| **Louisville** | 75 | 82.0% | **-0.0088** | +0.0093 | ['94.1%', '5.9%', '0.0%'] | ['76.1%', '20.8%', '3.1%'] | Compact geometry + Meta medium-bin overestimation |
| **Oakland** | 113 | 85.8% | **-0.0056** | +0.0100 | ['94.8%', '5.2%', '0.0%'] | ['80.5%', '17.9%', '1.6%'] | Compact geometry + Meta medium-bin overestimation |
| **Minneapolis** | 114 | 82.1% | **-0.0175** | +0.0109 | ['97.2%', '2.8%', '0.0%'] | ['79.3%', '17.7%', '3.0%'] | Compact geometry + Meta medium-bin overestimation |
| **San_Francisco** | 193 | 88.3% | **-0.0177** | +0.0023 | ['98.8%', '1.2%', '0.0%'] | ['87.1%', '10.5%', '2.4%'] | Compact geometry + Meta medium-bin overestimation |
| **Miami** | 97 | 76.9% | **-0.0404** | +0.0049 | ['98.4%', '1.6%', '0.0%'] | ['75.4%', '21.1%', '3.5%'] | Compact geometry + Meta medium-bin overestimation |