# Placebo Test Protocol (Target-Y_D)

This document details the scientific rationale, experimental design, and execution protocol for the `Target-Y_D` Placebo Test implemented in `src/experiment/run_placebo_matched_v2.py`.

---

## 1. Scientific Objective

The primary goal of the Placebo Test is to rigorously answer the following question:
> *Does the performance gain (Delta CPC) of the calibrated model ($M_1$) come from the precise, city-specific distance-binned distribution ($Y_D$) of the target city, or does it merely come from the general mathematical effects of the calibration/reweighting mechanism?*

We hypothesize that if our framework genuinely captures specific urban mobility patterns, calibrating with the *correct* target $Y_D$ should yield significantly higher CPC than calibrating with a *fake* (placebo) $Y_D$.

---

## 2. Experimental Design (The 5 Conditions)

For each target city $c$ and each model seed $s \in \{1, 10, 100\}$, the zero-shot baseline ($M_0$) is calibrated under 4 distinct conditions (producing $M_1$):

### 1. Target Condition (Upper Bound)
- The model is calibrated using the true $Y_D$ of the target city $c$.
- Metric: $\Delta \text{CPC}_{\text{target}} = \text{CPC}(M_{1,\text{target}}) - \text{CPC}(M_0)$.

### 2. Train-Mean Condition (Global Average)
- The model is calibrated using the average $Y_D$ of all 35 training cities within the same fold.
- Purpose: Tests whether simply shifting predictions to the national/fold average is sufficient to get the performance gain.

### 3. Wrong-City Placebo Condition
- The model is calibrated using a $Y_D$ randomly sampled from the 35 training cities in the same fold.
- Replicates: $B = 1000$ independently drawn wrong-city donors per target city.
- Metric: $\Delta \text{CPC}_{\text{wrong}, b}$ for $b \in \{1 \dots 1000\}$.
- Expected Gain: $E_b[\Delta \text{CPC}_{\text{wrong}, b}]$.

### 4. Permuted-Bin Placebo Condition (Stress-Test)
- The true target $Y_D$ is taken, but the probability mass among its **active bins** (bins with $p > 10^{-8}$) is randomly shuffled/permuted.
- Replicates: Up to $B = 1000$ unique permutations.
- Purpose: Acts as a **stress-test specificity under mass-preserving bin permutations**. Due to the lack of weight clipping, random permutations often create highly unrealistic target distributions (e.g., massive probability mass assigned to a bin where the model predicts near-zero mass). While our code guards against literal $w_k = \infty$ division-by-zero, such permutations still create hugely inflated calibration multipliers $w_k \gg 1$. Therefore, the massive CPC gap here should NOT be used to conclude the absolute real-world value of $Y_D$, but provides strong evidence that the improvement depends on the correct target-specific distance-bin structure rather than calibration strength alone.

---

## 3. Strict Protocol Controls

To prevent data leakage and guarantee absolute reproducibility, the test strictly obeys these invariants:

1. **Fold Isolation**: Target cities and wrong-city donors never cross fold boundaries. Donors are exclusively sampled from the 35 training cities of the *same* fold.
2. **Fixed Random Seeds**: 
   - `20260821` is used to sample wrong-city donors and generate permutations. This ensures all 3 model seeds (1, 10, 100) are evaluated against the *exact same* placebo distributions.
   - `42` is used for 95% Bootstrap Confidence Intervals.
3. **Hyperparameter Lockdown**: The calibration parameter is rigidly fixed at $q=1.0$. No oracle tuning of $q$ using the target CPC is permitted.
4. **Number of Bins**: Computations strictly utilize $K=8$ dynamically computed distance bins (derived from training folds).
5. **Masking & Renormalization**: If a placebo $Y_D$ assigns zero probability to a bin that is active in the target, it is masked and renormalized over the active bins to prevent zero-division errors during calibration.

---

## 4. Evaluation Metrics & Statistical Testing

### 4.1. Specificity Gain ($S$)
We define the **Specificity Gain** against a given placebo distribution as the difference between the correct target gain and the placebo gain:
$$ S_{\text{wrong}, c, s} = \Delta \text{CPC}_{\text{target}, c, s} - E_b[\Delta \text{CPC}_{\text{wrong}, c, s, b}] $$

This metric is averaged across the 3 model seeds to produce the city-level specificity gain.

### 4.2. Hypothesis Testing
- **Null Hypothesis ($H_0$)**: $E[S_{\text{wrong}}] \le 0$ (The true target $Y_D$ is no better than a randomly assigned wrong-city $Y_D$).
- **Alternative Hypothesis ($H_1$)**: $E[S_{\text{wrong}}] > 0$ (The precise target $Y_D$ specifically drives the performance).

**Primary Test**: A one-sided **Wilcoxon signed-rank test** is conducted on the city-level specificity gains from the 40 test cities located in the **confirmatory folds** (Folds 2, 3, 4, 5). Fold 1 is reserved strictly as exploratory.
