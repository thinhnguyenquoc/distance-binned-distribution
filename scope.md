# Research Scope & Design

## 1. Research Gap
It remains unclear whether a target-city distance-binned mobility distribution improves OD reconstruction beyond zero-shot inference under the same urban context and pairwise geographic-distance information.
*(Chưa rõ liệu phân phối di chuyển theo khoảng cách của thành phố mục tiêu có cải thiện khả năng tái tạo ma trận OD so với zero-shot khi cùng sử dụng bối cảnh đô thị và thông tin khoảng cách địa lý giữa các vùng hay không.)*

**Core Gap Focus**: The marginal value of $Y_D$, not just proving that distance information in general is useful.

---

## 2. Research Question 1 (RQ1)
**Does $Y_D$ improve OD reconstruction beyond zero-shot?**
*(Does a target-city distance-binned mobility distribution improve OD reconstruction beyond zero-shot inference under the same urban context and pairwise geographic-distance information?)*

### Hypothesis 1 (Main Hypothesis)
$$H_1: R_{YD} > R_{ZS}$$
or:
$$H_1: \Delta R = R_{YD} - R_{ZS} > 0$$

Where:
* $R_{ZS}$: Quality of OD reconstruction from the frozen zero-shot model.
* $R_{YD}$: Quality when incorporating target-city $Y_D$.
* $\Delta R$: Incremental reconstruction gain.

---

## 3. Research Question 2 (RQ2)
**If $Y_D$ improves OD reconstruction, how much directly observed target-city OD information is its incremental value equivalent to?**
*(Nếu $Y_D$ cải thiện tái tạo OD, mức cải thiện đó tương đương với việc quan sát trực tiếp khoảng bao nhiêu phần trăm OD của thành phố mục tiêu?)*

### Hypothesis 2 (Secondary Quantitative Hypothesis)
$$H_2: \exists p^* > 0: R_{YD} \approx R_{p^*\%OD}$$

Where:
$$p^* = \arg\min_p \left| R_{YD} - R_{p\%OD} \right|$$

$p^*$ is the **OD-equivalent information value** of $Y_D$.
(e.g., Target-city distance-binned mobility information provides reconstruction value approximately equivalent to observing 10% of the target-city OD information).

---

## 4. Overall Research Logic
$$ \text{Gap} \rightarrow \text{RQ1} \rightarrow H_1 \rightarrow \text{RQ2} \rightarrow H_2 $$

1. **Unknown marginal value of $Y_D$**
   $\downarrow$
2. **RQ1: Does $Y_D$ help beyond zero-shot?**
   $\downarrow$
3. **H1: $\Delta R > 0$**
   $\downarrow$
4. **RQ2: How much is that improvement worth?**
   $\downarrow$
5. **H2: $Y_D$ has an OD-equivalent value $p^*$**

---

## 5. Experimental Design Mapping

**Three main conditions:**
1. $X_{\text{urban}}, D_{ij} \rightarrow \hat{T}^{ZS}$
2. $X_{\text{urban}}, D_{ij}, Y_D \rightarrow \hat{T}^{YD}$
3. Calibration baselines: $X_{\text{urban}}, D_{ij}, p\% T^{GT} \rightarrow \hat{T}^{p\%OD}$
   *(Example: $p \in \{1, 5, 10, 20, 50\}\%$)*

**Answers:**
* For RQ1: Evaluate $\Delta R = R_{YD} - R_{ZS}$
* For RQ2: Find $p^*$

---

## Proposal Summary (Short Version)

**Gap**
> The marginal OD-reconstruction value of target-city distance-binned mobility information beyond zero-shot urban-context inference remains unclear.

**RQ1**
> Does target-city distance-binned mobility information improve OD reconstruction beyond zero-shot?

**H1**
> $\Delta R > 0$

**RQ2**
> If so, how much directly observed target-city OD information is that improvement equivalent to?

**H2**
> $R_{YD} \approx R_{p^*\%OD}$
