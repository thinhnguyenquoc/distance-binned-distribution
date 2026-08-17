## Dùng độ dài chuyến đi làm thông tin hiệu chỉnh
- Average/mean trip length để hiệu chỉnh tham số suy giảm khoảng cách của mô hình gravity, điển hình là phương pháp Hyman (Hyman, 1969).
- Observed trip-length distribution để hiệu chỉnh hoặc cân bằng mô hình gravity; tri-proportional balancing có thể buộc ma trận ước lượng khớp TLD quan sát (van der Zijpp & Heydecker).
- Median trip length để ước lượng tham số impedance; Merlin cho thấy median có thể ổn định hơn TLD trong một số mô hình một tham số (Merlin, 2020).

```
Thông tin độ dài chuyến đi có giúp hiệu chỉnh tham số hoặc hàm distance-decay của một spatial-interaction model hay không?
```
Chúng không trực tiếp kiểm tra marginal improvement trên một mô hình OD zero-shot đã được huấn luyện liên thành phố và bị đóng băng.
## Zero-shot/cross-city OD generation
- Deep Gravity sử dụng đặc trưng đô thị và khoảng cách để khái quát sang các khu vực chưa thấy (Simini et al., 2021).
- Cross-city prompt tuning cho zero-shot flow generation (Wang et al., 2026).
- neuroGravity chuyển mô hình sang các thành phố không quan sát và sử dụng đặc trưng dân số, cơ sở đô thị và không gian (Yang et al., 2026).
Một số nghiên cứu còn dùng distance distribution để đánh giá kết quả zero-shot. Tuy nhiên, distance distribution được dùng như metric hoặc quy luật học trong training, không phải một quan sát tổng hợp của chính thành phố mục tiêu được bổ sung tại inference.
## novelty
- Định lượng marginal reconstructive value của một target-city distance-binned mobility distribution đối với một zero-shot OD model đã đóng băng, trong khi urban context và pairwise geographic distance được giữ giống nhau.
## Gap
- Mặc dù thông tin về độ dài chuyến đi đã được sử dụng để hiệu chỉnh các mô hình tương tác không gian, và các mô hình gần đây có thể dự báo OD zero-shot từ bối cảnh đô thị và khoảng cách địa lý, vẫn chưa rõ liệu việc bổ sung phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu tại thời điểm suy luận có cải thiện khả năng tái tạo OD ở cấp độ cặp vùng so với zero-shot hay không.
## Narrative
- Prior studies demonstrate that mean, median, and full trip-length distributions can be used to calibrate the distance-decay component of spatial-interaction models. Separately, recent cross-city models can generate OD flows zero-shot from urban context and pairwise distance. However, these two research streams have not established whether a coarse target-city distance-binned distribution provides non-redundant, destination-resolved information beyond an otherwise identical zero-shot prediction. This unresolved marginal value constitutes the research gap.