## Dùng độ dài chuyến đi làm thông tin hiệu chỉnh
- Hyman (1969) thực tế là mục tiêu đi tìm tham số cho hàm decay sao cho mô hình sinh ra có average trip length giống với quan sát và không đề cập đến vấn đề zero-shot.
- Van der Zijpp và Heydecker mô tả tri-proportional balancing như một phương pháp có thể điều chỉnh prior matrix để đồng thời khớp trip ends và observed trip-length distribution. Tuy nhiên, thực nghiệm chính của họ đánh giá các thuật toán hiệu chỉnh gravity model từ traffic counts và observed OD measurements, thay vì đo marginal improvement do việc bổ sung TLD.
- Median trip time để ước lượng tham số impedance; Merlin cho thấy TLD method có thể có sai số tham số rất lớn, đặc biệt với power impedance function (Merlin, 2020). Thông qua việc sinh matrix từ Beta ground truth dùng gravity doubly constraints tạo các thông tin nén rồi sau đó ước lương lại Beta xem có sai lệch nhiều ko và theo Media trip time.
- Ait-Ali & Eliasson (2022) nghiên cứu marginal value của average travel distance. đo marginal value của thông tin khoảng cách đối với OD estimation.

```
Thông tin độ dài chuyến đi có giúp hiệu chỉnh tham số hàm distance-decay của một spatial-interaction model hay không?
```
Chúng không trực tiếp kiểm tra marginal improvement trên một mô hình OD zero-shot đã được huấn luyện liên thành phố và bị đóng băng.
## Zero-shot/cross-city OD generation
- Deep Gravity sử dụng đặc trưng đô thị và khoảng cách để khái quát sang các khu vực chưa thấy (Simini et al., 2021).
- GODDAG: target city không có OD labels nhưng mô hình không hoàn toàn frozen
- Cross-city prompt tuning cho zero-shot flow generation (Wang et al., 2026).
- neuroGravity chuyển mô hình sang các thành phố không quan sát và sử dụng đặc trưng dân số, cơ sở đô thị và không gian (Yang et al., 2026).
Một số nghiên cứu còn dùng distance distribution để đánh giá kết quả zero-shot. Tuy nhiên, distance distribution được dùng như metric hoặc quy luật học trong training, không phải một quan sát tổng hợp của chính thành phố mục tiêu được bổ sung tại inference.
## novelty
- Định lượng marginal reconstructive value của một target-city distance-binned mobility distribution đối với một zero-shot OD model đã đóng băng, trong khi urban context và pairwise geographic distance được giữ giống nhau.
## Gap
- Mặc dù thông tin về độ dài chuyến đi đã được sử dụng để hiệu chỉnh các mô hình tương tác không gian, và các mô hình gần đây có thể dự báo OD zero-shot từ bối cảnh đô thị và khoảng cách địa lý, vẫn chưa rõ liệu việc bổ sung phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu tại thời điểm suy luận có cải thiện khả năng tái tạo OD ở cấp độ cặp vùng so với zero-shot hay không.
## Narrative
- Prior studies demonstrate that mean, median, and full trip-length distributions can be used to calibrate the distance-decay component of spatial-interaction models. Separately, recent cross-city models can generate OD flows zero-shot from urban context and pairwise distance. However, these two research streams have not established whether a coarse target-city distance-binned distribution provides non-redundant, destination-resolved information beyond an otherwise identical zero-shot prediction. This unresolved marginal value constitutes the research gap.

## Phân biệt TLD và binned distribution
bài toán này nghe giống như việc một distance binned distribution không thể xác định được 1 TLD và là một họ TLD. Nhưng 1 TLD có thể xác định chính xác 1 binned distribution. Nên về cơ bản có thể nói là Binned distribution chứa ít thông tin chi tiết hơn so với TLD.

## thiết kế test
Thiết kế mạnh nhất sẽ là:

Oracle bins→Noisy/sampled bins→Real bins
	​
Tầng 1 — Oracle information-value test
Tầng 2 — Robustness test
Tầng 3 — Real-observation test - Meta

| Điều kiện         | Thông tin target city được cung cấp | Mục đích                          |
| ----------------- | ----------------------------------- | --------------------------------- |
| Zero-shot         | Không có (Y_D)                      | Baseline                          |
| Wrong-city bins   | (Y_D) của thành phố khác            | Kiểm tra tính đặc thù thành phố   |
| Oracle bins       | (B(T^{GT}))                         | Đo giá trị tối đa của các bin     |
| Noisy bins        | (Y_D^{oracle}+\epsilon)             | Đo robustness                     |
| Full TLD          | (f^{GT}(d))                         | Đo thông tin mất do binning       |
| Limited target OD | 5–10% (T^{GT})                      | So sánh với quan sát OD trực tiếp |

## maximum entropy

Khi không có bằng chứng để ưu tiên destination nào, ta chọn nghiệm đưa vào ít giả định bổ sung nhất. Tức là chọn giả thiếu các destination có xác xuất đến là như nhau. và điều này tương ứng là max entropy.

Entropy thấp: lưu lượng tập trung vào một số ít cặp OD.
Entropy cao: lưu lượng được phân tán trên nhiều cặp OD.

## held out city

| Cách sử dụng target urban information                                  | Có thay đổi mô hình? | Cách gọi phù hợp                                   |
| ---------------------------------------------------------------------- | -------------------: | -------------------------------------------------- |
| Đưa (X_{\text{target}},D_{\text{target}}) vào mô hình frozen để dự báo |                Không | Held-out-city zero-shot                            |
| Tính embedding bằng encoder đã khóa                                    |                Không | Held-out-city zero-shot                            |
| Chuẩn hóa dữ liệu bằng quy tắc cố định từ trước                        |                Không | Held-out-city zero-shot                            |
| Fine-tune encoder bằng (X_{\text{target}})                             |                   Có | Unsupervised target adaptation                     |
| Domain alignment bằng target urban data                                |                   Có | Transductive/domain-adaptation setting             |
| Prompt tuning bằng target urban data                                   |                   Có | Target-context adaptation                          |
| Huấn luyện bằng pseudo-label OD của target city                        |                   Có | Pseudo-label adaptation, không phải pure zero-shot |
| Sử dụng một phần target OD                                             |          Có giám sát | Few-shot target-city learning                      |
| Sử dụng toàn bộ target OD                                              |          Có giám sát | Supervised target-city training                    |
