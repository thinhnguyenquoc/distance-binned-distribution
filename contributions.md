# Research Contributions & Significance

## Core Positioning
> **This study does not propose another zero-shot OD generator; it quantifies how much additional OD-relevant information a coarse target-city mobility observation contributes beyond zero-shot inference.**

The zero-shot model serves as an *experimental instrument* to measure information value, not the central contribution itself. The central question is:
> *How much target-city OD information is contained in a coarse aggregate mobility-distance observation?*

---

## 3-Tier Contribution Framework

### 1. Scientific Contribution
**Information Sufficiency / Complementary Information:**
Determines whether target-city aggregate distance information ($Y_D$) contains additional OD-relevant information after controlling for urban context and geographic distance.
* If $\Delta R > 0$: Proves that even without direct destination identity, $Y_D$ significantly reduces OD ambiguity.
* If $\Delta R \approx 0$: Shows that distance-binned information is largely *redundant* once urban context and distance ($D_{ij}$) are absorbed by the zero-shot model.
Both outcomes provide valuable, publishable scientific insights.

### 2. Methodological Contribution
Proposes a novel evaluation framework to measure the marginal value ($\Delta R$) and mathematically convert it into a highly interpretable metric: **OD-equivalent information** ($p^*\%$).

### 3. Empirical Contribution
Quantifies the actual improvement across multiple held-out cities. Evaluates robustness, sensitivity, and identifies under which specific city contexts $Y_D$ provides the most (or least) value.

---

## Practical Significance
By converting the improvement into an OD-equivalent value ($R_{YD} \approx R_{p^*\%OD}$), the research provides a direct, actionable answer to practical resource-allocation questions for data-scarce cities:
* *Should we collect $Y_D$?*
* *Should we invest in direct OD collection?*

Example conclusion: *"A cheap, aggregate distance-binned mobility distribution provides OD reconstruction information equivalent to directly observing 10% of the target-city's OD flows."*
