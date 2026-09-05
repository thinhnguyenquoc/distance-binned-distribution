# Cải thiện tái tạo cường độ luồng OD của mô hình zero-shot giữ nguyên tham số bằng phân phối di chuyển theo khoảng cách của thành phố mục tiêu

## Tóm tắt

Ma trận nguồn–đích là đầu vào quan trọng cho phân tích giao thông và quy hoạch đô thị, nhưng dữ liệu chi tiết về cường độ luồng OD của thành phố mục tiêu thường khó thu thập. Các nghiên cứu sử dụng ngữ cảnh đô thị và khoảng cách địa lý đã phát triển các baseline cross-city zero-shot có khả năng dự báo luồng di chuyển mà không sử dụng dữ liệu quan sát về cường độ OD của thành phố mục tiêu. Nghiên cứu này xem xét liệu phân phối di chuyển theo khoảng cách của thành phố mục tiêu có thể cải thiện việc tái tạo cường độ luồng OD liên vùng trên tập hỗ trợ dương đã biết thông qua hiệu chỉnh kết quả của một baseline zero-shot có tham số được giữ nguyên hay không.

Phân phối di chuyển này được tổng hợp từ dữ liệu quan sát luồng của chính thành phố mục tiêu và chỉ cung cấp tỷ trọng khối lượng luồng theo các khoảng khoảng cách, không cung cấp cường độ của từng cặp OD cụ thể. Trong thí nghiệm chính, phương pháp được đánh giá bằng quy trình kiểm định chéo 5-fold trên các thành phố của Hoa Kỳ. Hiệu chỉnh ở cấp thành phố tạo ra mức cải thiện nhỏ nhưng nhất quán: CPC trung bình tăng 0.00354 (CI 95%: $[+0.0026, +0.0045]$), với 45/50 thành phố được cải thiện. Đồng thời, mức cải thiện cũng giảm khi độ phân giải hoặc chất lượng của phân phối quan sát dần suy giảm. Nghiên cứu chỉ đánh giá tái tạo cường độ trên tập các cặp OD liên vùng có luồng dương đã biết.

**Từ khóa:** ma trận nguồn–đích; tái tạo cường độ OD; phân phối di chuyển theo khoảng cách; zero-shot; học chuyển giao giữa các thành phố; quan sát tổng hợp; di chuyển không gian.

# Mục 1: Giới thiệu

Ma trận nguồn–đích (OD) mô tả cường độ di chuyển giữa các đơn vị không gian và là đầu vào quan trọng cho phân tích giao thông và quy hoạch đô thị. Tuy nhiên, dữ liệu OD chi tiết thường khó thu thập đầy đủ tại thành phố mục tiêu và có thể chịu hạn chế về độ phủ và tính đại diện [@gallotti2024distorted; @pappalardo2023future]. Luồng di chuyển cũng phụ thuộc vào bối cảnh đô thị và đặc trưng địa phương, nên các quy luật học được từ thành phố nguồn không nhất thiết chuyển giao hoàn toàn sang thành phố mục tiêu. Do đó, các mô hình chuyển giao giữa thành phố vẫn có thể mang sai lệch có hệ thống tại thành phố mục tiêu khi không có thông tin hiệu chỉnh địa phương [@yang2014limits].

Các mô hình mobility gần đây đã kết hợp ngữ cảnh đô thị và khoảng cách để dự báo luồng có khả năng chuyển giao giữa các thành phố [@simini2021deepgravity; @guo2025ugnn; @enaya2026transgm]. Tuy nhiên, một baseline cross-city giữ nguyên tham số chỉ suy luận thành phố mục tiêu từ các đặc trưng đầu vào sẵn có. Dù biết khoảng cách của từng cặp OD, mô hình không trực tiếp quan sát cách tổng khối lượng di chuyển của thành phố mục tiêu được phân bổ giữa các khoảng khoảng cách [@lenormand2016comparison; @verma2025distance]. Sự phân bổ luồng theo khoảng cách có thể khác nhau giữa các thành phố, nên một phân phối khoảng cách đặc thù của thành phố mục tiêu có thể chứa thông tin mà baseline cross-city chưa suy diễn đầy đủ.

Nghiên cứu này kiểm tra liệu phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu có cung cấp thông tin bổ sung cho một baseline cross-city đã huấn luyện hay không. Phân phối này chỉ mô tả tỷ trọng tổng luồng theo khoảng cách và được sử dụng tại thời điểm suy luận để hiệu chỉnh dự báo, trong khi toàn bộ tham số mô hình được giữ nguyên. Phép hiệu chỉnh được sử dụng như một công cụ thực nghiệm để định lượng giá trị thông tin bổ sung của phân phối di chuyển.

Nghiên cứu được tổ chức quanh hai câu hỏi: 
- Phân phối di chuyển theo khoảng cách của thành phố mục tiêu có cải thiện tái tạo cường độ OD so với baseline cross-city zero-shot giữ nguyên tham số hay không?
- Nếu có, mức cải thiện phụ thuộc như thế nào vào độ phân giải, chất lượng, thứ tự khoảng và tính đặc thù của quan sát mục tiêu?

Trong nghiên cứu này, phân phối được trích xuất từ luồng tham chiếu của chính thành phố mục tiêu và vì vậy được xem là quan sát oracle. Thiết lập này được sử dụng để kiểm tra giá trị thông tin của tín hiệu trước khi xem xét khả năng thu thập hoặc ước lượng nó từ nguồn độc lập.

Nghiên cứu được đánh giá bằng kiểm định chéo liên thành phố 5-fold trên 50 vùng đô thị Hoa Kỳ. Mỗi thành phố được đánh giá khi không tham gia huấn luyện, và toàn bộ tham số mô hình được giữ nguyên trước bước hiệu chỉnh.

Kết quả cho thấy phân phối tạo ra mức cải thiện nhỏ nhưng tương đối nhất quán, và lợi ích này phụ thuộc vào độ phân giải, chất lượng và tính đặc thù của quan sát mục tiêu.

Nghiên cứu có 3 đóng góp chính: 
- Định lượng giá trị thông tin bổ sung của phân phối trên một baseline cross-city giữ nguyên tham số.
- Xác định các điều kiện chi phối lợi ích của tín hiệu thông qua độ phân giải, nhiễu và các đối chứng target-specific.
- Phân tích cơ chế hiệu chỉnh và kiểm tra độ bền của kết quả trên nhiều khởi tạo và kiến trúc baseline.



# Mục 2: Nghiên cứu liên quan

## 2.1. Mô hình tương tác không gian và hiệu chỉnh dựa trên khoảng cách

Các mô hình tương tác không gian từ lâu đã biểu diễn luồng OD thông qua khả năng phát sinh, mức độ thu hút và lực cản không gian, trong đó khoảng cách hoặc chi phí di chuyển là thành phần cốt lõi của cấu trúc luồng [@wilson1971family; @ortuzar2011modelling].

Các phương pháp hiệu chỉnh cổ điển cho thấy thống kê tổng hợp về cự ly chuyến đi có thể được dùng để xác định tham số lực cản, chẳng hạn chiều dài chuyến đi trung bình trong Hyman (1969) hoặc trung vị thời gian di chuyển trong Merlin (2020). Các nghiên cứu so sánh cũng cho thấy quy luật suy giảm theo khoảng cách thay đổi giữa bộ dữ liệu và bối cảnh đô thị, thay vì tồn tại một dạng hàm cố định phù hợp cho mọi nơi. Mẫu hình suy giảm thực nghiệm có thể thay đổi theo phương thức, mục đích chuyến đi, mức độ đô thị hóa và điều kiện kinh tế–xã hội [@verma2025distance].

Những kết quả này cho thấy khoảng cách là một cấu trúc tổ chức quan trọng của luồng, đồng thời thông tin cự ly đặc thù của miền mục tiêu có thể bổ sung cho một quy luật suy giảm được chuyển giao chung. Nghiên cứu hiện tại kế thừa ý tưởng đó nhưng dùng toàn bộ phân phối theo khoảng thay vì ước lượng một tham số distance-decay đơn lẻ.

## 2.2. Sinh dữ liệu di chuyển và mô hình không gian dựa trên học máy

Các mô hình học máy gần đây như Deep Gravity và các kiến trúc neural có nhận thức địa lý cho thấy đặc trưng đô thị và khoảng cách có thể được học để dự báo luồng và chuyển giao giữa các khu vực [@simini2021deepgravity; @guo2025ugnn; @enaya2026transgm].

Khái quát hóa liên thành phố vẫn khó vì ánh xạ từ bối cảnh đô thị sang luồng không bất biến theo không gian. Một mô hình nguồn có thể biết khoảng cách của từng cặp nhưng vẫn không biết tỷ lệ thực nghiệm của tổng luồng tại thành phố mục tiêu được phân bổ giữa các dải cự ly. Nghiên cứu trước về khả năng dự báo luồng đi làm cho thấy việc thiếu dữ liệu hiệu chỉnh địa phương tạo ra giới hạn đáng kể đối với độ chính xác [@yang2014limits]. Hạn chế này không tự biến mất khi khoảng cách cặp được đưa vào đầu vào: khoảng cách cho mô hình biết hai vùng cách nhau bao xa, nhưng không cho biết tỷ lệ thực nghiệm của tổng luồng tại thành phố mục tiêu được phân bổ vào dải cự ly đó.

Vì vậy, nghiên cứu hiện tại giải quyết một vấn đề bổ sung cho hướng thiết kế kiến trúc neural. Khác với hướng cải thiện kiến trúc hoặc fine-tuning, nghiên cứu này giữ nguyên mô hình cross-city đã huấn luyện và kiểm tra riêng giá trị thông tin của một quan sát tổng hợp từ miền mục tiêu. Cách thiết kế này tách giá trị thông tin của quan sát mục tiêu khỏi những cải thiện có thể phát sinh do huấn luyện bổ sung, fine-tuning hoặc thay đổi kiến trúc.

## 2.3. Quan sát tổng hợp như một ràng buộc hiệu chỉnh

Quan sát tổng hợp nằm giữa hai cực: hoàn toàn không có thông tin tại thành phố mục tiêu và quan sát trực tiếp toàn bộ ma trận OD. Các ràng buộc cổ điển như tổng outflow, inflow hoặc moment chi phí đã được dùng để áp đặt tính nhất quán vĩ mô với số lượng quan sát ít hơn nhiều so với số ô OD.

Khác với các phương pháp chủ yếu hiệu chỉnh một hoặc vài tham số, nghiên cứu này sử dụng trực tiếp vector tỷ trọng luồng theo các khoảng khoảng cách. Điều này cho phép kiểm tra giá trị của tín hiệu ở nhiều mức độ phân giải khác nhau thông qua số lượng khoảng $K$. $Y_D$ khác với các biên origin/destination hoặc các cặp OD quan sát trực tiếp: nó chỉ ràng buộc cách tổng khối lượng được phân bổ giữa các dải cự ly và không xác định phân bổ giữa các cặp trong cùng một khoảng.

Tín hiệu tổng hợp trong nghiên cứu này không đồng nhất với tổng lượng chuyến đi theo điểm đi và điểm đến hoặc một mẫu các cặp OD được quan sát trực tiếp. Mỗi loại quan sát ràng buộc một khía cạnh khác nhau của ma trận luồng chưa biết. Phân phối di chuyển theo các khoảng khoảng cách chỉ ràng buộc tỷ lệ tổng khối lượng di chuyển được phân bổ vào từng khoảng cự ly; bản thân nó không xác định cặp origin–destination cụ thể nào phải nhận nhiều luồng hơn trong cùng một khoảng.


## 2.4. Khoảng trống nghiên cứu và vị trí của nghiên cứu hiện tại

Các hướng nghiên cứu trước đã cho thấy vai trò quan trọng của khoảng cách trong tương tác không gian, khả năng chuyển giao của các mô hình dự báo luồng và giá trị của các ràng buộc tổng hợp. Tuy nhiên, một câu hỏi vẫn chưa được kiểm tra trực tiếp: sau khi một mô hình cross-city đã học từ ngữ cảnh đô thị và khoảng cách giữa các cặp vùng, phân phối di chuyển theo khoảng cách của chính thành phố mục tiêu còn cung cấp thêm bao nhiêu giá trị, và giá trị đó duy trì trong những điều kiện quan sát nào?

Trong thiết kế của nghiên cứu này, phân phối di chuyển theo khoảng cách của thành phố mục tiêu được sử dụng như tín hiệu target-specific duy nhất đưa vào sau khi mô hình đã huấn luyện xong; backbone được giữ nguyên và không có cập nhật trọng số. Cách thiết kế này cho phép tách giá trị thông tin của quan sát mục tiêu khỏi những cải thiện có thể phát sinh do fine-tuning hoặc huấn luyện bổ sung.

Các nghiên cứu hiện có chưa trực tiếp kiểm tra giá trị cải thiện biên của tín hiệu này trên một baseline cross-city đã được huấn luyện và giữ nguyên tham số. Vì vậy, nghiên cứu tập trung vào việc định lượng giá trị bổ sung của phân phối di chuyển theo khoảng cách của thành phố mục tiêu và xác định những điều kiện quan sát chi phối mức cải thiện đó.


# Mục 3: Nguồn dữ liệu, đơn vị không gian và phương pháp luận



## 3.1. Ký hiệu và dữ liệu đầu vào

Gọi $c$ là một thành phố và $\mathcal{V}_c$ là tập các vùng đơn vị phân chia thành phố đó. Mỗi cặp có thứ tự $(i,j)$ với $i,j \in \mathcal{V}_c$ biểu diễn một cặp nguồn–đích (OD). 

### Bảng 1: Ký hiệu cốt lõi, nguồn dữ liệu và trạng thái sẵn có của thông tin

| Ký hiệu | Mô tả toán học | Nguồn / Vai trò |
| :--- | :--- | :--- |
| $c$ | Chỉ số thành phố ($c \in \{1, \dots, C\}$) | Mã định danh thành phố ($C = 50$) |
| $i, j$ | Chỉ số vùng xuất phát (origin) và vùng đích (destination) | Đơn vị không gian cơ sở |
| $t_{c,ij}$ | Cường độ luồng di chuyển quan sát được ($t_{c,ij} \ge 1$) | Dữ liệu tham chiếu (ground truth) |
| $d_{c,ij}$ | Khoảng cách giữa tâm của vùng $i$ và vùng $j$ (km) | Tính từ tọa độ tâm (Haversine) |
| $\Omega_c$ | Tập hỗ trợ liên vùng dương đã biết; xem định nghĩa đầy đủ tại Mục 3.2. | Giả định support đã biết |
| $I_b$ | Khoảng khoảng cách thứ $b$ ($b = 1, \dots, K$) | Phân vị khoảng cách |
| $K$ | Số lượng khoảng khoảng cách ($K = 8$ ở thiết lập chính) | Cấu hình thực nghiệm cố định |
| $Y_{c,b}$ | Tỷ trọng luồng di chuyển mục tiêu trong khoảng $b$ ($\sum_{b=1}^K Y_{c,b} = 1$) | Dữ liệu đầu vào hiệu chỉnh oracle |
| $\hat{t}_{c,ij}^{(0)}$ | Dự báo cường độ luồng của baseline cross-city zero-shot (điều kiện $M_0$) | Đầu ra baseline giữ nguyên tham số |
| $\hat{t}_{c,ij}^{(1)}$ | Dự báo cường độ luồng sau hiệu chỉnh tại thời điểm suy luận (điều kiện $M_1$) | Đầu ra sau hiệu chỉnh |
| $M_0, M_1$ | Tên hai điều kiện thực nghiệm (baseline zero-shot giữ nguyên tham số và dự báo sau hiệu chỉnh) | Điều kiện thực nghiệm đối chứng |



## 3.2. Nguồn dữ liệu và biểu diễn không gian

Phạm vi đánh giá được giới hạn trên tập hỗ trợ liên vùng dương đã biết:

$$
\Omega_c = \left\{(i,j) \in \mathcal{V}_c \times \mathcal{V}_c : t_{c,ij} \ge 1,\ i \neq j,\ d_{c,ij} > 0\right\}.
$$

Trong toàn bài, các cặp ngoài $\Omega_c$ được xem là chưa biết và không thuộc phạm vi đánh giá.


## 3.3. Đơn vị không gian và cấu hình chuẩn cấp thành phố
Các thử nghiệm chính sử dụng một phân phối di chuyển theo khoảng cách duy nhất ở cấp thành phố. Tỷ trọng luồng di chuyển mục tiêu rơi vào khoảng khoảng cách thứ $b$ ($I_b = [a_{b-1}, a_b)$) được định nghĩa là:

$$
Y_{c,b} = \frac{\sum_{(i,j) \in \Omega_c} t_{c,ij} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_c} t_{c,ij}} 
$$

Các tỷ trọng được chuẩn hóa để: $\sum_{b=1}^K Y_{c,b} = 1$.

$Y_{D,c}$ được tổng hợp từ luồng ground-truth của thành phố mục tiêu và được sử dụng như một quan sát oracle tại thời điểm hiệu chỉnh. Một biến thể thăm dò sử dụng phân phối theo origin-county được đánh giá trên các vùng đô thị multi-county, thiết lập và giới hạn của phân tích này được trình bày trong Phụ lục S7.


## 3.4. Cấu trúc mô hình và hiệu chỉnh tại thời điểm suy luận

### 3.4.1. Giao diện dự báo baseline chung

Ba baseline được đánh giá gồm Urban GNN, Pairwise Node MLP và Gravity hai tham số. Mỗi baseline được huấn luyện hoặc ước lượng chỉ trên các thành phố nguồn và tạo dự báo zero-shot $\hat{t}^{(0)}_{c,ij}$ trên thành phố mục tiêu mà không sử dụng $Y_D$. Cùng một toán tử hiệu chỉnh được áp dụng cho cả ba baseline.

Cùng một quy tắc hiệu chỉnh theo khoảng cách được áp dụng cho dự báo ban đầu của cả ba mô hình. Urban GNN là mô hình chính, Pairwise Node MLP và Gravity hai tham số được sử dụng để đánh giá mức độ phụ thuộc của hiệu quả hiệu chỉnh vào kiến trúc baseline.

### 3.4.2. Mô hình neural chính: Urban GNN kết hợp tiên nghiệm Gravity

Urban GNN là baseline chính của nghiên cứu. Mô hình mã hóa các đặc trưng bối cảnh đô thị của từng tract thông qua một đồ thị không gian, sau đó kết hợp embedding của origin và destination với khoảng cách cặp và một gravity prior để dự báo cường độ luồng OD dương.

Mô hình được huấn luyện trên các thành phố nguồn của từng fold và toàn bộ tham số được giữ cố định khi suy luận trên thành phố mục tiêu.

### 3.4.3. Mô hình tham số cổ điển: Gravity hai tham số

Để kiểm tra liệu hiệu quả hiệu chỉnh có phụ thuộc vào backbone neural hay không, chúng tôi sử dụng thêm một baseline gravity dạng lũy thừa:

$$
\hat{t}^{(0,\mathrm{grav})}_{c,ij} = \exp(G) \frac{P_{c,i} P_{c,j}}{d_{c,ij}^{\alpha}}, \qquad (i,j) \in \Omega_c.
$$

Trong đó, $G$ là hệ số quy mô toàn cục và $\alpha > 0$ là số mũ suy giảm theo khoảng cách. Để bảo đảm ổn định số học, dân số tract được chặn dưới tại

$$
P_{c,i} = \max(\operatorname{pop}_{c,i}, 1.0), \qquad P_{c,j} = \max(\operatorname{pop}_{c,j}, 1.0),
$$

và khoảng cách được chặn dưới tại

$$
d_{c,ij} = \max(\mathrm{dist}_{c,ij}, 0.1\,\text{km}).
$$

Hai tham số $(G, \alpha)$ được ước lượng bằng pooled log-linear ordinary least squares chỉ trên các thành phố huấn luyện của từng fold và được giữ cố định khi suy luận trên thành phố kiểm tra. Dự báo gravity sau đó được đưa qua cùng toán tử hiệu chỉnh bằng $Y_D$ như các baseline khác.

### 3.4.4. Mô hình bóc tách: Pairwise Node MLP

Để kiểm tra liệu hiệu quả của $Y_D$ có phụ thuộc riêng vào cơ chế truyền thông điệp trên đồ thị hay không, chúng tôi sử dụng thêm một Pairwise Node MLP không có graph convolution. Mô hình sử dụng cùng đặc trưng tract, cùng gravity prior và cùng decoder cặp OD như baseline neural chính, nhưng mỗi tract được mã hóa độc lập trước khi dự báo luồng.

### 3.4.5. Mục tiêu huấn luyện dưới quan sát partial OD

Do benchmark chỉ chứa các luồng dương, hai neural baseline được huấn luyện bằng Zero-Truncated Negative Binomial likelihood.

$$
p_+(t \mid \mu, \phi) = \frac{p_{\mathrm{NB}}(t \mid \mu, \phi)}{1 - p_{\mathrm{NB}}(0 \mid \mu, \phi)}, \qquad \mathcal{L}_c = -\frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j) \in \Omega_c} \log p_+(t_{c,ij} \mid \mu_{c,ij}, \phi).
$$

Loss được lấy trung bình trên các cặp $(i,j) \in \Omega_c$ của từng thành phố để tránh các đô thị có nhiều cặp OD chi phối quá trình tối ưu.

### 3.4.6. Cấu hình huấn luyện và lựa chọn checkpoint

Hai neural baseline sử dụng cùng protocol huấn luyện, chọn checkpoint theo CPC validation và được lặp trên ba model seeds. Sau khi chọn checkpoint, toàn bộ tham số được giữ nguyên trên target cities. Chi tiết siêu tham số huấn luyện được cung cấp trong Phụ lục.

### 3.4.7. Toán tử hiệu chỉnh khoảng cách tại thời điểm suy luận

Cho bất kỳ mô hình baseline giữ nguyên tham số nào tạo ra dự báo ban đầu $\hat{t}_{c,ij}^{(0)}$ trên $\Omega_c$, phân phối khoảng cách ngầm định bởi baseline trong khoảng thứ $b$ là:

$$
\widehat{Y}_{c,b}^{(0)} = \frac{\sum_{(i,j) \in \Omega_c} \hat{t}_{c,ij}^{(0)} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_c} \hat{t}_{c,ij}^{(0)}}.
$$

Toán tử hiệu chỉnh giải tích tái phân bổ khối lượng luồng theo công thức nghiệm đóng:

$$
\hat{t}_{c,ij}^{(1)} = \hat{t}_{c,ij}^{(0)} \frac{Y_{c,b(i,j)}}{\widehat{Y}_{c,b(i,j)}^{(0)}}.
$$

Trong đó, $b(i,j)$ là khoảng chứa $d_{c,ij}$. Mọi cặp OD trong cùng một khoảng được nhân với cùng một hệ số. Phép hiệu chỉnh không cập nhật tham số mô hình. Thiết lập chính cố định $q = 1$; dạng tổng quát $q \in [0, 1]$ được trình bày trong Phụ lục S2. Do các hệ số hiệu chỉnh dương và không đổi trong mỗi khoảng, toán tử bảo toàn tập hỗ trợ, thứ hạng nội khoảng và tổng khối lượng dự báo; các chứng minh được trình bày trong Phụ lục S3.

![Hình 1](figures/fig1_oracle_calibration_framework.png)
**Hình 1. Framework hiệu chỉnh oracle có điều kiện theo support.** Baseline cross-city tạo dự báo $\widehat{\mathbf{T}}_c^{(0)}$ trên thành phố mục tiêu; $Y_D$ oracle được dùng để tái phân bổ khối lượng giữa các khoảng khoảng cách và tạo $\widehat{\mathbf{T}}_c^{(1)}$ mà không cập nhật tham số mô hình.



## 3.5. Giao thức đánh giá cross-city và suy luận thống kê

### 3.5.1. Giao thức kiểm định chéo liên thành phố 5-fold

Nghiên cứu áp dụng giao thức kiểm định chéo liên thành phố 5-fold trên 50 vùng đô thị Hoa Kỳ (mỗi fold gồm 35 thành phố huấn luyện, 5 thành phố validation và 10 thành phố kiểm tra). Đơn vị phân chia fold là toàn bộ thành phố; không có bất kỳ cặp OD hoặc tract nào của cùng một thành phố bị phân tán giữa tập huấn luyện và tập kiểm tra.

### 3.5.2. Thước đo đánh giá và so sánh mô hình

Thước đo định lượng chính để đánh giá khả năng tái tạo luồng di chuyển zero-shot là Common Part of Commuters (CPC) [@lenormand2016comparison], được tính trên tập hỗ trợ liên vùng dương $\Omega_c$:

$$
\operatorname{CPC}_c(\hat{t}) = \frac{2 \sum_{(i,j) \in \Omega_c} \min(t_{c,ij}, \hat{t}_{c,ij})}{\sum_{(i,j) \in \Omega_c} t_{c,ij} + \sum_{(i,j) \in \Omega_c} \hat{t}_{c,ij}}.
$$

CPC nằm trong $[0, 1]$, với giá trị lớn hơn biểu thị mức chồng lấp lớn hơn giữa cường độ dự báo và quan sát.

Các thước đo sai số và xếp hạng bổ sung được báo cáo như kiểm tra độ bền; định nghĩa đầy đủ được trình bày trong Phụ lục S4.

Bên cạnh đó, phân phối khoảng cách gộp sau hiệu chỉnh được đối chiếu với $Y_D$ như một chẩn đoán cơ chế nội bộ nhằm xác nhận thuật toán đã tái phân bổ khối lượng đúng thiết kế. Cả ba họ mô hình (GNN, MLP, Gravity) được so sánh trên cùng tập hỗ trợ $\Omega_c$ theo CPC baseline.

### 3.5.3. Phân tích thống kê và lượng hóa độ bất định

Đối với mỗi thành phố, mức cải thiện được tính từ chênh lệch CPC giữa dự báo sau hiệu chỉnh và baseline, sau đó lấy trung bình qua các model seeds và macro-average trên toàn bộ 50 thành phố.

Khoảng tin cậy 95% được ước lượng bằng paired nonparametric bootstrap ở cấp thành phố, phân tầng theo fold. Ý nghĩa thống kê của các chênh lệch ghép cặp được đánh giá bằng kiểm định Wilcoxon signed-rank. Tỷ lệ thành phố có $\Delta\mathrm{CPC} > 0$ được báo cáo như một thống kê mô tả bổ sung.

### 3.5.4. Các phân tích độ bền và chẩn đoán

Ngoài thí nghiệm chính, chúng tôi kiểm tra liệu hiệu quả của $Y_D$ có phụ thuộc vào độ phân giải khoảng cách, chất lượng quan sát, thứ tự các khoảng, tính đặc thù của thành phố mục tiêu, khởi tạo mô hình và kiến trúc baseline hay không. Các thiết lập cụ thể được trình bày cùng kết quả tương ứng.

# 4. Kết quả thực nghiệm



## 4.1. Việc sử dụng $Y_D$ có cải thiện tái tạo OD so với baseline zero-shot giữ nguyên tham số hay không?

Trong thí nghiệm chính sử dụng Urban GNN làm baseline chính, CPC liên vùng trung bình trên 50 thành phố tăng từ 0.71281 ở baseline $M_0$ lên 0.71635 sau hiệu chỉnh $M_1$, tương ứng với $\Delta\mathrm{CPC}=+0.00354$ và khoảng tin cậy bootstrap 95% $[+0.0026,+0.0045]$. Trung vị $\Delta\mathrm{CPC}$ là $+0.00195$, với 45/50 thành phố có mức thay đổi dương. Kiểm định Wilcoxon signed-rank ghép cặp cho $p=1.93\times10^{-9}$. Mức tăng tương đương khoảng 0.5% CPC của baseline và có năm thành phố suy giảm, cho thấy hiệu quả có quy mô nhỏ và không xuất hiện ở mọi thành phố.


![Hình 2](figures/fig2_main_per_city.png)
**Hình 2: Mức cải thiện CPC liên vùng theo từng thành phố từ hiệu chỉnh khoảng cách mục tiêu.** 

Biểu đồ cột thể hiện mức thay đổi hiệu năng theo từng thành phố $\Delta\text{CPC}_c = \text{CPC}(M_{1,c}) - \text{CPC}(M_{0,c})$ trên $N=50$ thành phố kiểm tra, xếp từ thấp đến cao. Đường nét đứt màu xanh lá thể hiện mức cải thiện trung bình ($+0.00354$) và đường chấm màu cam thể hiện trung vị ($+0.00195$). Tổng cộng có 45/50 thành phố (90.0%) đạt mức tăng dương, với khoảng tin cậy 95% phân tầng theo fold là $[+0.0026, +0.0045]$.



### Bảng 2: Benchmark chính với Urban GNN (N=50,K=8)

| Điều kiện thực nghiệm | CPC liên vùng TB | Trung vị CPC | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% (Phân tầng) | Tỷ lệ thành phố thắng | Wilcoxon $p$ (Hai phía) |
|---|---|---|---|---|---|---|
| **Baseline zero-shot giữ nguyên tham số ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **Dự báo sau hiệu chỉnh ($M_1$)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | **$[+0.0026, +0.0045]$** | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ |




## 4.2. Mức cải thiện có thực sự đặc thù theo thành phố mục tiêu và có ý nghĩa cấu trúc hay không?

Để phân biệt thông tin đặc thù của thành phố mục tiêu với ảnh hưởng chung của việc thay đổi dự báo, chúng tôi so sánh $Y_D$ chính xác của target với các phân phối donor được khớp để tạo ra cùng mức thay đổi phân phối. $Y_D$ của target tạo ra mức tăng trung bình $\Delta\mathrm{CPC}=+0.003539$, trong khi các donor đã khớp chỉ tạo ra $-0.000091$. Chênh lệch giữa hai điều kiện là $+0.003630$, với khoảng tin cậy 95% $[+0.00287,+0.00445]$ và kiểm định Wilcoxon một phía $p=2.19\times10^{-11}$. Vì vậy, cùng một mức độ điều chỉnh nhưng sử dụng phân phối của thành phố khác không tái tạo được lợi ích của thông tin target.

Để kiểm tra liệu một phân phối khoảng cách chung có đủ để hiệu chỉnh baseline hay không, chúng tôi sử dụng phân phối trung bình của các thành phố huấn luyện, được khớp theo cùng mức độ thay đổi phân phối. Điều kiện này tạo ra $\Delta\mathrm{CPC}=+0.000914$, thấp hơn mức $+0.003539$ khi sử dụng $Y_D$ của target. Chênh lệch giữa hai điều kiện là $+0.002626$, với khoảng tin cậy 95% $[+0.00197,+0.00336]$ và kiểm định Wilcoxon một phía $p=4.03\times10^{-11}$, cho thấy phân phối trung bình không tái tạo được lợi ích của thông tin đặc thù theo thành phố.

Để kiểm tra vai trò của thứ tự khoảng cách, chúng tôi hoán vị các thành phần trong $Y_D$ của target. Phép biến đổi này giữ nguyên các giá trị tỷ trọng nhưng phá vỡ sự tương ứng giữa mỗi tỷ trọng và khoảng cách của nó. Hiệu chỉnh bằng phân phối đã hoán vị làm CPC giảm trung bình $-0.006964$, trái với mức tăng $+0.003539$ khi sử dụng đúng thứ tự. Kết quả cho thấy lợi ích của $Y_D$ phụ thuộc vào sự liên kết chính xác giữa tỷ trọng luồng và các khoảng cách tương ứng.



![Hình 3](figures/fig5_structural_validity_placebo.png)
**Hình 3: Các đối chứng placebo khớp liều lượng công bằng.** 

So sánh mức tăng tái tạo trung bình $\Delta\mathrm{CPC}$ trên $N=50$ thành phố kiểm tra dưới 3 điều kiện: (1) Phân phối mục tiêu thực sự ($Y_D$, $+0.00354$, $p < 10^{-8}$); (2) Đối chứng donor từ thành phố khác đã khớp liều lượng ($-0.00009$, không có ý nghĩa); và (3) Hoán vị các khoảng khoảng cách ($-0.00696$, $p < 10^{-14}$). Thanh sai số biểu diễn khoảng tin cậy 95% bootstrap phân tầng.



### Bảng 3: Tính đặc thù mục tiêu và các đối chứng Placebo ($N=50$)

| Experimental Condition | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% (Phân tầng) | Benefit vs $M_0$ ($p_{\text{2-sided}}$) | Specificity Gain vs Placebo | Specificity 95% CI | Target vs Placebo ($p_{\text{1-sided}}$) | Target-specific win rate ($\text{Target } Y_D > \text{Placebo}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Oracle Target $Y_D$ (Upper Bound)** | **$+0.003539$** | $[+0.00260, +0.00450]$ | $1.93 \times 10^{-9}$ | — | — | — | **45 / 50 (vs $M_0$)** |
| **2. Dose-Matched Training Donors ($B_{\text{draw}}=1000$)** | **$-0.000091$** | $[-0.00089, +0.00071]$ | $0.4097$ (n.s.) | **$+0.003630$** | $[+0.00287, +0.00445]$ | $\mathbf{2.19 \times 10^{-11}}$ | **46 / 50 (92.0%)** |
| **3. Dose-Matched Fold Train-Mean $Y_D$** | **$+0.000914$** | $[+0.00001, +0.00186]$ | $0.4319$ (n.s.) | **$+0.002626$** | $[+0.00197, +0.00336]$ | $\mathbf{4.03 \times 10^{-11}}$ | **47 / 50 (94.0%)** |
| **4. Permuted Target $Y_D$ ($B_{\text{draw}}=1000$ Permutations)** | **$-0.006964$** | $[-0.00914, -0.00512]$ | $1.78 \times 10^{-15}$ | **$+0.010504$** | $[+0.00843, +0.01279]$ | $1.78 \times 10^{-15}$ | **49 / 50 (98.0%)** |




## 4.3. Giá trị bổ sung của $Y_D$ phụ thuộc như thế nào vào độ phân giải và chất lượng quan sát?

Chúng tôi đánh giá sự thay đổi của $\Delta\mathrm{CPC}$ theo số khoảng khoảng cách $K$ và theo mức sai số Total Variation được bổ sung vào $Y_D$, lần lượt đại diện cho độ phân giải và chất lượng của quan sát.



### 4.3.1. Độ phân giải khoảng cách cao hơn cung cấp ràng buộc giàu thông tin hơn

$\Delta\mathrm{CPC}$ tăng đơn điệu từ $+0.00098$ tại $K=2$ lên $+0.00639$ tại $K=20$, với mức $+0.00354$ ở cấu hình chính $K=8$. Khoảng tin cậy 95% nằm phía trên 0 tại tất cả các giá trị $K$ được kiểm tra, trong khi số thành phố cải thiện tăng từ 39/50 tại $K=2$ lên 46/50 tại $K=20$. Mức cải thiện tăng theo độ phân giải khoảng cách trong phạm vi các giá trị $K$ được đánh giá, với tốc độ tăng có xu hướng chậm lại ở các cấu hình $K$ lớn hơn.

### Bảng 4: Độ mở rộng của độ phân giải thông tin qua các khoảng khoảng cách

| Độ phân giải ($K$) | CPC liên vùng TB | Trung vị CPC | $\Delta\text{CPC}$ trung bình | Trung vị $\Delta\text{CPC}$ | Khoảng tin cậy 95% (Phân tầng) | Tỷ lệ thành phố thắng |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **$K = 2$** | $0.71379 \pm 0.04441$ | $0.71665$ | **$+0.00098$** | $+0.00034$ | $[+0.00052, +0.00151]$ | **39 / 50 (78.0%)** |
| **$K = 4$** | $0.71479 \pm 0.04439$ | $0.71720$ | **$+0.00198$** | $+0.00088$ | $[+0.00125, +0.00279]$ | **39 / 50 (78.0%)** |
| **$K = 6$** | $0.71570 \pm 0.04445$ | $0.71784$ | **$+0.00289$** | $+0.00152$ | $[+0.00201, +0.00384]$ | **44 / 50 (88.0%)** |
| **$K = 8$ (Anchor)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | $+0.00195$ | $[+0.00262, +0.00447]$ | **45 / 50 (90.0%)** |
| **$K = 10$** | $0.71694 \pm 0.04450$ | $0.72007$ | **$+0.00413$** | $+0.00235$ | $[+0.00311, +0.00514]$ | **45 / 50 (90.0%)** |
| **$K = 12$** | $0.71761 \pm 0.04453$ | $0.72060$ | **$+0.00480$** | $+0.00288$ | $[+0.00372, +0.00590]$ | **46 / 50 (92.0%)** |
| **$K = 14$** | $0.71819 \pm 0.04456$ | $0.72145$ | **$+0.00538$** | $+0.00373$ | $[+0.00424, +0.00654]$ | **45 / 50 (90.0%)** |
| **$K = 16$** | $0.71855 \pm 0.04458$ | $0.72205$ | **$+0.00574$** | $+0.00433$ | $[+0.00455, +0.00694]$ | **46 / 50 (92.0%)** |
| **$K = 18$** | $0.71884 \pm 0.04460$ | $0.72230$ | **$+0.00603$** | $+0.00458$ | $[+0.00480, +0.00726]$ | **47 / 50 (94.0%)** |
| **$K = 20$** | $0.71920 \pm 0.04462$ | $0.72266$ | **$+0.00639$** | $+0.00494$ | $[+0.00508, +0.00769]$ | **46 / 50 (92.0%)** | 




![Hình 4](figures/fig3_resolution_sensitivity.png)
**Hình 4 | Phân tích độ nhạy của độ phân giải thông tin ($K$).** Mức tăng CPC liên vùng trung bình $\Delta\text{CPC}$ tăng đơn điệu từ $K=2$ ($+0.00098$) lên $K=20$ ($+0.00639$). Dải bóng mờ biểu diễn khoảng tin cậy 95% bootstrap phân tầng.

Trong một phân tích thăm dò trên 11 vùng đô thị trải rộng qua nhiều county, hiệu chỉnh cấp county cải thiện so với hiệu chỉnh cấp thành phố tại 9/11 trường hợp. Tuy nhiên, mức tăng bổ sung pooled trên toàn bộ 50 vùng đô thị chỉ là $\Delta\mathrm{CPC}_{\mathrm{res}} = +0.00014$, vì 39 vùng single-county tạo ra hai phân hoạch tương đương về mặt toán học. Do đó, kết quả này không được xem là bằng chứng tổng quát rằng tăng độ phân giải không gian sẽ cải thiện hiệu năng; chi tiết được trình bày trong Phụ lục S7.



### 4.3.2. Nhiễu quan sát tổng hợp làm giảm giá trị bổ sung của $Y_D$

Độ nhạy đối với chất lượng quan sát được đánh giá bằng cách gây nhiễu $Y_D$ của thành phố mục tiêu tại các mức sai số Total Variation $\epsilon\in[0.00,0.05]$, trong khi giữ nguyên baseline, tập thành phố đánh giá và toán tử hiệu chỉnh.


![Hình 5](figures/fig4_noise_dose_response.png)
**Hình 5 | Đường đáp ứng theo mức nhiễu Total Variation (TV).** Mức cải thiện sau hiệu chỉnh, $\Delta\mathrm{CPC}$, giảm khi mức nhiễu TV tăng từ $\epsilon=0.00$ đến $\epsilon=0.05$. Đường ngang tại $\Delta\mathrm{CPC}=0$ biểu thị mức hiệu năng tương đương baseline zero-shot $M_0$. Đường đứt nét màu đỏ thẳng đứng biểu thị điểm giao cắt thực nghiệm $\epsilon_{\mathrm{cross}}\approx4.44\%$, tại đó lợi ích trung bình của hiệu chỉnh xấp xỉ bằng 0. Dải bóng mờ biểu diễn khoảng tin cậy bootstrap 95%. Giá trị $\epsilon_{\mathrm{cross}}$ chỉ áp dụng cho benchmark và cơ chế gây nhiễu được sử dụng trong nghiên cứu này.

$\Delta\mathrm{CPC}$ giảm đơn điệu từ $+0.00354$ khi không có nhiễu xuống $+0.00070$ tại sai số TV 4% và $-0.00087$ tại 5%. Trên 1.000 hướng nhiễu, điểm giao cắt trung bình với baseline được ước lượng tại $\epsilon_{\mathrm{cross}}=4.44\%$, với khoảng tin cậy 95% $[4.16\%, 4.77\%]$. Đây là ngưỡng thực nghiệm riêng cho benchmark và cơ chế gây nhiễu đã sử dụng, không phải mức dung sai áp dụng chung cho dữ liệu thực tế.

## 4.4. Kết quả có bền vững trước các lựa chọn huấn luyện và mô hình hóa hay không?

Các kết quả trước cho thấy $Y_D$ cung cấp thông tin bổ sung cho dự báo zero-shot, nhưng mức cải thiện phụ thuộc vào độ phân giải, chất lượng quan sát và tính đặc thù của thành phố mục tiêu. Để đánh giá liệu kết quả này có ổn định trước biến thiên ngẫu nhiên trong huấn luyện và lựa chọn mô hình hay không, chúng tôi thực hiện phân tích trên nhiều model seeds và các kiến trúc baseline khác nhau.

### 4.4.1. Tính ổn định qua các lần khởi tạo mô hình độc lập

Để kiểm tra liệu hiệu quả của $Y_D$ có phụ thuộc vào một lần khởi tạo mô hình cụ thể hay không, chúng tôi lặp lại cùng giao thức trên ba model seeds độc lập.

Mức cải thiện trung bình vẫn dương ở cả ba seed, với $\Delta\mathrm{CPC}$ dao động từ khoảng $+0.0031$ đến $+0.0043$. Kết quả này cho thấy hiệu ứng quan sát được không chỉ xuất hiện ở một trạng thái khởi tạo duy nhất.


### 4.4.2. Hiệu quả trên các kiến trúc neural và mô hình Gravity cổ điển

Mức cải thiện do $Y_D$ xuất hiện trên cả hai neural backbone được đánh giá. Urban GNN đạt $\Delta\mathrm{CPC}=+0.00354$, trong khi Node MLP đạt $+0.00329$. Với Gravity hai tham số, mức tăng nhỏ hơn đáng kể và không cho thấy cùng mức độ ổn định. Kết quả này cho thấy hiệu ứng không chỉ phụ thuộc vào riêng kiến trúc Urban GNN, nhưng cũng không mở rộng đồng đều sang mọi họ mô hình.



### Bảng 5: Tính tổng quát trên các kiến trúc backbone ($N=50$ thành phố, $K=8$ khoảng)

| Kiến trúc backbone | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% Bootstrap | Tỷ lệ thành phố thắng |
|:---|:---:|:---:|:---:|
| **Urban GNN (Truyền thông điệp)**  | **$+0.00354$** | $[+0.0026, +0.0045]$ | **45 / 50 (90.0%)** |
| **Node MLP (Không truyền thông điệp đồ thị)**  | **$+0.00329$** | $[+0.0025, +0.0042]$ | **47 / 50 (94.0%)** | 
| **Mô hình Gravity 2 tham số cổ điển** | $+0.00084$ | $[+0.0002, +0.0016]$ | 22 / 50 (44.0%) | 

*Ghi chú: Tất cả mô hình được đánh giá theo cùng kiểm định chéo 5-fold ($N=50$ thành phố kiểm tra; hai mô hình neural được tính trung bình qua 3 seeds). Mô hình Gravity dạng log-linear được ước lượng bằng pooled ordinary least squares (OLS) trên dữ liệu các thành phố huấn luyện của từng fold, tuyệt đối không sử dụng dữ liệu luồng của thành phố kiểm tra.*


### 4.4.3 Tổng hợp về độ bền vững và tính ổn định của hiệu chỉnh

Mức tăng do hiệu chỉnh được tái hiện qua nhiều model seeds độc lập và trên cả hai neural backbone đã đánh giá là Urban GNN và Node MLP. Gravity baseline cổ điển chỉ cho mức thay đổi nhỏ, không có ý nghĩa thống kê; vì vậy bằng chứng kiến trúc chỉ hỗ trợ robustness trên hai neural backbone đã kiểm tra, không mở rộng cho mọi họ mô hình. Phân tích độ nhạy theo độ phân giải khoảng cách sử dụng pair-weighted quantile bins được xây dựng hoàn toàn từ các thành phố huấn luyện. Tổng hợp lại, kết quả chính không phải hệ quả riêng của một lần khởi tạo tham số hoặc chỉ của kiến trúc Urban GNN.


## 4.5. Mức sai lệch phân phối khoảng cách của baseline có liên hệ mạnh với mức cải thiện hiệu chỉnh theo thành phố

Sai lệch phân phối khoảng cách ban đầu của baseline có liên hệ mạnh với mức cải thiện sau hiệu chỉnh. Sau khi kiểm soát độ chính xác baseline và quy mô đô thị, tương quan từng phần vẫn đạt $r_{\mathrm{partial}}=+0.795$. Kết quả này phù hợp với cơ chế của phương pháp: $Y_D$ hữu ích nhất tại những thành phố mà baseline phân bổ sai khối lượng giữa các khoảng khoảng cách. Đây là bằng chứng liên hệ quan sát, không phải bằng chứng nhân quả.


![Hình 6](figures/fig6_mechanistic_dpre.png)
**Hình 6 | Phân tích cơ chế giải thích sai lệch phân phối khoảng cách ban đầu ($d_{\text{pre}}$).** Tương quan giữa sai số Total Variation ban đầu của baseline $d_{\text{pre}} = \text{TV}(\widehat{Y}_D^{(0)}, Y_D^{\text{GT}})$ và mức cải thiện $\Delta\mathrm{CPC}$ tại từng thành phố. Hệ số tương quan từng phần sau khi kiểm soát độ chính xác ban đầu và quy mô đô thị đạt $r_{\mathrm{partial}} = +0.7951$ ($p = 5.35 \times 10^{-12}$), cho thấy các thành phố mà mô hình cơ sở ước lượng sai lệch nhiều nhất về cơ cấu cự ly sẽ nhận được lợi ích lớn nhất từ phép hiệu chỉnh.



*Ghi chú: Đánh giá trên toàn bộ $N=50$ thành phố kiểm tra. $d_{\mathrm{pre}} = \operatorname{TV}(\widehat{Y}_D^{(0)}, Y_D^{\mathrm{GT}})$ measures the Total Variation error between the zero-shot baseline's distance allocation and ground truth. Multivariate OLS serves as an observational diagnostic for linear association with performance gain heterogeneity rather than a causal model. Significance: *** $p < 0.001$.*


# Mục 5: Thảo luận chuyên sâu

Trong phần này, chúng tôi đặt các phát hiện của nghiên cứu vào bức tranh tổng thể của các nghiên cứu về mô hình hóa di chuyển con người và học chuyển giao không gian [@barbosa2018humanmobility; @enaya2026transgm; @lenormand2016comparison; @simini2021deepgravity]. Chúng tôi phân tích các cơ chế lý thuyết giải thích giá trị thông tin của phân phối cự ly tổng hợp, đánh giá độ phân giải quan sát và độ nhạy đối với nhiễu tổng hợp có kiểm soát, thảo luận ý nghĩa phương pháp luận và thực tiễn cho phân tích đô thị khan hiếm dữ liệu, đồng thời nêu rõ các hạn chế chính và định hướng nghiên cứu tiếp theo.


### 5.1. Giá trị thông tin bổ sung của $Y_D$

Kết quả cho thấy phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu cung cấp một lượng thông tin bổ sung nhỏ nhưng tương đối nhất quán cho baseline cross-city giữ nguyên tham số. Điều này cho thấy khoảng cách cặp và bối cảnh đô thị mà mô hình đã quan sát chưa đủ để suy diễn hoàn toàn cách tổng khối lượng di chuyển của từng thành phố được phân bổ theo cự ly.

Quan trọng hơn, trong thiết kế thực nghiệm này lợi ích này xuất hiện mà không cần cập nhật tham số mô hình, nên có thể được diễn giải như giá trị thông tin riêng của quan sát tổng hợp $Y_D$, thay vì hiệu ứng của fine-tuning hoặc huấn luyện bổ sung.

### 5.2. Cơ chế hiệu chỉnh: tái phân bổ khối lượng giữa các khoảng khoảng cách

Toán tử hiệu chỉnh sử dụng $Y_D$ để điều chỉnh tổng khối lượng dự báo giữa các khoảng khoảng cách, trong khi giữ nguyên tỷ lệ tương đối giữa các cặp OD nằm trong cùng một khoảng. Vì vậy, $Y_D$ chỉ có thể sửa sai lệch ở cấu trúc cự ly vĩ mô mà baseline chưa mô hình hóa đúng, chứ không tái sắp xếp chi tiết các cặp OD trong cùng một bin.

Phù hợp với cơ chế này, các thành phố có sai lệch phân phối khoảng cách ban đầu lớn hơn thường nhận được mức cải thiện lớn hơn sau hiệu chỉnh. Mối liên hệ này hỗ trợ cách diễn giải cơ chế của phương pháp, nhưng không được xem là bằng chứng nhân quả.

Sự khác biệt về mức cải thiện giữa các thành phố vì vậy có thể được hiểu một phần qua mức sai lệch phân phối khoảng cách ban đầu của baseline.

### 5.3. Các điều kiện chi phối giá trị của Y_D
	​
Kết quả theo $K$ cho thấy giá trị của $Y_D$ tăng khi ràng buộc khoảng cách trở nên chi tiết hơn, nhưng lợi ích biên giảm ở các phân hoạch mịn hơn. Điều này gợi ý rằng phần lớn thông tin hữu ích nằm ở cấu trúc phân bổ cự ly tương đối thô, trong khi tăng thêm độ chi tiết chỉ mang lại lợi ích bổ sung nhỏ hơn.

Giá trị của $Y_D$ phụ thuộc vào cả cấu trúc và độ chính xác của quan sát. Khi thứ tự các khoảng khoảng cách bị phá vỡ, lợi ích của hiệu chỉnh không còn được duy trì. Lợi ích này phụ thuộc vào việc các tỷ trọng được gắn đúng với các khoảng khoảng cách của thành phố mục tiêu và được quan sát với độ chính xác đủ cao.

Các đối chứng donor cho thấy lợi ích của hiệu chỉnh không xuất hiện chỉ vì mô hình được cung cấp thêm một phân phối khoảng cách bất kỳ. Phân phối của thành phố khác hoặc phân phối trung bình từ tập huấn luyện không tái tạo được mức cải thiện đạt được khi sử dụng $Y_D$ của đúng thành phố mục tiêu.

Kết quả này cho thấy phần thông tin hữu ích trong $Y_D$ mang tính đặc thù theo thành phố, thay vì chỉ phản ánh một prior suy giảm theo khoảng cách chung có thể chuyển trực tiếp giữa các đô thị.


## 5.6. Ý nghĩa phương pháp luận và phạm vi ứng dụng

Các mô hình như Deep Gravity và UGNN cho thấy neural networks có thể kết hợp nhiều dạng thông tin địa lý để học các quy luật mobility có khả năng chuyển giao [@simini2021deepgravity; @guo2025ugnn]. Tuy nhiên, các mô hình này vẫn cần OD observations từ các khu vực nguồn để huấn luyện. Đóng góp của nghiên cứu hiện tại không phải loại bỏ nhu cầu về OD training data, mà là cho thấy một mô hình nguồn đã huấn luyện có thể được điều chỉnh tại inference time bằng một quan sát tổng hợp của thành phố mục tiêu mà không cần cập nhật tham số.

Về mặt phương pháp, kết quả cho thấy một ràng buộc tổng hợp chính xác tại miền mục tiêu có thể điều chỉnh mô hình cross-city có tham số được giữ nguyên ở thời điểm suy luận mà không cần fine-tuning tham số hoặc huấn luyện lại end-to-end. Kết quả không chứng minh tính khả thi triển khai, mà chỉ xác lập rằng nếu một quan sát tổng hợp đủ chính xác tồn tại, nó có thể chứa thông tin bổ sung hữu ích. Do phép hiệu chỉnh chỉ tái phân bổ cường độ trên tập hỗ trợ đã biết, kết quả không mở rộng sang bài toán phát hiện các liên kết OD chưa quan sát.


## 5.7. Các giới hạn của nghiên cứu

Mobility datasets có thể chứa sai lệch về độ phủ, tính đại diện và quy trình tiền xử lý [@gallotti2024distorted; @pappalardo2023future]. Ngoài ra, giảm độ phân giải hoặc tổng hợp dữ liệu không tự động tạo ra bảo đảm quyền riêng tư. Mobility traces vẫn có thể chứa thông tin nhận dạng đáng kể sau khi được làm thô [@demontjoye2013unique], và việc cung cấp bảo đảm differential privacy ở cấp người dùng cho dữ liệu vị trí tổng hợp vẫn gặp nhiều khó khăn thực tế [@houssiau2022differential]. Nghiên cứu hiện tại không thực hiện privacy analysis đối với $Y_D$; vì vậy, $Y_D$ chỉ nên được gọi là một quan sát tổng hợp có số chiều thấp, không phải một cơ chế privacy-preserving đã được chứng minh.

Phân tích county-level chỉ mang tính thăm dò. Chỉ 11 vùng đô thị trong benchmark tạo ra phân hoạch multi-county thực sự, trong khi 39 trường hợp còn lại tương đương với hiệu chỉnh cấp thành phố. Hơn nữa, county là ranh giới hành chính và có thể không phản ánh đúng các vùng di chuyển chức năng. Vì vậy, kết quả này không hỗ trợ một claim tổng quát về lợi ích của độ phân giải không gian chi tiết hơn.

$Y_D$ được trích từ chính target ground-truth OD, nên chưa chứng minh hiệu quả với quan sát được thu thập độc lập.

Evaluation chỉ diễn ra trên known positive support, không xử lý zero flows hoặc link discovery.

## 5.8. Các định hướng nghiên cứu tương lai

Một hướng phát triển tự nhiên là kết hợp $Y_D$ với các ràng buộc tổng hợp khác, chẳng hạn tổng outflow theo origin hoặc tổng inflow theo destination. Các mô hình spatial interaction cổ điển cung cấp nền tảng cho việc áp dụng đồng thời các ràng buộc sản sinh, thu hút và impedance [@wilson1971family; @ortuzar2011modelling]. Các hướng nghiên cứu gần đây cũng nhấn mạnh giá trị của việc kết hợp mechanistic mobility models với các phương pháp học máy có khả năng mở rộng và diễn giải [@pappalardo2023future]. Future work có thể đánh giá các nguồn quan sát tổng hợp độc lập, đơn vị địa lý, điều kiện truy cập và mức độ phù hợp được xác lập—nhưng nghiên cứu hiện tại chưa sử dụng telemetry bên ngoài.

# 6. Kết luận

Nghiên cứu này xem xét liệu một quan sát tổng hợp có số chiều thấp—phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu, có thể cải thiện kết quả tái tạo cường độ luồng OD liên vùng trên tập hỗ trợ dương đã biết so với một baseline cross-city zero-shot có tham số được giữ nguyên hay không. Trong thiết lập này, baseline ($M_0$) có toàn bộ tham số được giữ nguyên và chỉ sử dụng bối cảnh đô thị cùng khoảng cách địa lý giữa các cặp vùng. Thông tin về cường độ luồng của thành phố mục tiêu chỉ đi vào $M_1$ dưới dạng phân phối di chuyển theo khoảng cách tổng hợp $Y_D$ tại thời điểm suy luận mà không đòi hỏi bất kỳ sự huấn luyện lại hay cập nhật tham số nào.

Kết quả thực nghiệm trên 50 thành phố Hoa Kỳ cho thấy việc hiệu chỉnh bằng $Y_D$ tạo ra mức cải thiện CPC trung bình $+0.00354$ (khoảng tin cậy bootstrap 95%: $[+0.0026, +0.0045]$, trung vị $+0.00195$, kiểm định Wilcoxon ghép cặp $W = 83.0, p = 1.93 \times 10^{-9}$), với 45 trong 50 thành phố có kết quả tốt hơn baseline (tỷ lệ thắng 90.0%). Những kết quả này trả lời tích cực cho câu hỏi nghiên cứu chính: phân phối khoảng cách theo khoảng của thành phố mục tiêu chứa thông tin bổ sung mà baseline zero-shot chưa phản ánh đầy đủ trong benchmark này.

Các thí nghiệm chẩn đoán và kiểm tra độ bền vững làm rõ các điều kiện chi phối giá trị thông tin này. Trên các độ phân giải đã kiểm tra, mức cải thiện tăng khi độ phân giải $K$ tăng trong phạm vi các cấu hình đã đánh giá, với xu hướng lợi ích biên giảm dần. Trong thiết kế nhiễu Total Variation tổng hợp của nghiên cứu, mức tăng trung bình đi qua 0 gần $\epsilon_{\text{cross}}\approx4.44\%$ sai số TV; đây là điểm giao cắt thực nghiệm riêng cho benchmark, không phải bảo đảm dung sai phổ quát. Hoán vị sai thứ tự các bin làm giảm độ chính xác ($\Delta\mathrm{CPC}=-0.00696$, $p<10^{-14}$), còn donor placebo từ thành phố khác được khớp liều lượng không tái tạo mức tăng của target ($\Delta\mathrm{CPC}=-0.000091$, $p=0.4097$). Tổng hợp các kiểm tra này hỗ trợ cách diễn giải rằng lợi ích quan sát được phụ thuộc vào thông tin khoảng cách đúng thứ tự và đặc thù của thành phố mục tiêu trong các điều kiện đã đánh giá.

Về mặt phương pháp, nghiên cứu cung cấp bằng chứng thực nghiệm rằng một quan sát tổng hợp có số chiều thấp có thể hiệu chỉnh một mô hình neural cross-city có tham số được giữ nguyên tại thời điểm suy luận mà không cần fine-tuning. Về mặt cơ chế, toán tử hiệu chỉnh sử dụng $Y_D$ để tái phân bổ khối lượng luồng dự báo giữa các khoảng khoảng cách và bảo toàn thứ hạng nội khoảng. Sai lệch phân bổ khoảng cách của baseline có liên hệ mạnh với mức tăng sau hiệu chỉnh ($r_{\text{partial}}=+0.7951$, $p=5.35\times10^{-12}$); mẫu hình này phù hợp với cơ chế trên nhưng chưa đủ để thiết lập quan hệ nhân quả. Vì vậy, $Y_D$ là một ràng buộc vĩ mô bổ sung chứ không phải sự thay thế độc lập cho ma trận OD chi tiết.

Mặc dù mức cải thiện xuất hiện tại 90% số thành phố được đánh giá, độ lớn tuyệt đối vẫn khiêm tốn và thay đổi theo mức sai lệch ban đầu của baseline. Do đó, phương pháp nên được hiểu như một bước hậu xử lý nhẹ, không phải sự thay thế cho các cuộc khảo sát giao thông toàn diện.

Tóm lại, phân phối di chuyển theo nhóm khoảng cách của thành phố mục tiêu cung cấp một ràng buộc tổng hợp minh bạch về mặt toán học và tạo ra mức cải thiện nhỏ nhưng tương đối nhất quán so với baseline zero-shot giữ nguyên tham số trong benchmark này. Kết luận này chỉ áp dụng cho tái tạo cường độ trên tập hỗ trợ dương đã biết với $Y_D$ oracle; nó không mở rộng sang link discovery, full-matrix reconstruction hay triển khai với quan sát tổng hợp được thu thập độc lập.



# Mục 7: Tuyên bố về khả năng truy cập dữ liệu và mã nguồn

Bố sung sau

# Mục 8: Các tuyên bố và cam kết khoa học
Bổ sung sau

# Mục 9: Tài liệu tham khảo




1. **Barbosa, H., Barthelemy, M., Ghoshal, G., James, C. R., Lenormand, M., Louail, T., Menezes, R., Ramasco, J. J., Simini, F., & Tomasini, M.** (2018). Human mobility: Models and applications. *Physics Reports*, 734, 1–74. [https://doi.org/10.1016/j.physrep.2018.01.001](https://doi.org/10.1016/j.physrep.2018.01.001)

2. **de Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., & Blondel, V. D.** (2013). Unique in the crowd: The privacy bounds of human mobility. *Scientific Reports*, 3, 1376. [https://doi.org/10.1038/srep01376](https://doi.org/10.1038/srep01376)

3. **Enaya, A., Zhong, C., Batty, M., Morphet, R., & Lopane, F. D.** (2026). TransGM: Transferable gravity models for cross-city policy transfer. *Computers, Environment and Urban Systems*, 128, 102455. [https://doi.org/10.1016/j.compenvurbsys.2026.102455](https://doi.org/10.1016/j.compenvurbsys.2026.102455)

4. **GADM.** (n.d.). *GADM database of global administrative areas (Version 4.1)* [Data set]. Retrieved September 2, 2026, from [https://gadm.org/data.html](https://gadm.org/data.html)

5. **Gallotti, R., Maniscalco, D., Barthelemy, M., & De Domenico, M.** (2024). Distorted insights from human mobility data. *Communications Physics*, 7, 421. [https://doi.org/10.1038/s42005-024-01909-x](https://doi.org/10.1038/s42005-024-01909-x)

6. **Grogger, J. T., & Carson, R. T.** (1991). Models for truncated counts. *Journal of Applied Econometrics*, 6(3), 225–238. [https://doi.org/10.1002/jae.3950060302](https://doi.org/10.1002/jae.3950060302)

7. **Guo, J., Bai, S., Li, X., Xian, K., Liu, E., Ding, W., & Ma, X.** (2025). A universal geography neural network for mobility flow prediction in planning scenarios. *Computer-Aided Civil and Infrastructure Engineering*, 40, 5769–5789. [https://doi.org/10.1111/mice.13398](https://doi.org/10.1111/mice.13398)

8. **Hilbe, J. M.** (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press.

9. **Houssiau, F., Rocher, L., & de Montjoye, Y.-A.** (2022). On the difficulty of achieving differential privacy in practice: User-level guarantees in aggregate location data. *Nature Communications*, 13, 29. [https://doi.org/10.1038/s41467-021-27566-0](https://doi.org/10.1038/s41467-021-27566-0)

10. **Hyman, G. M.** (1969). The calibration of trip distribution models. *Environment and Planning A*, 1(1), 105–112. [https://doi.org/10.1068/a010105](https://doi.org/10.1068/a010105)

11. **Lenormand, M., Bassolas, A., & Ramasco, J. J.** (2016). Systematic comparison of trip distribution laws and models. *Journal of Transport Geography*, 51, 158–169. [https://doi.org/10.1016/j.jtrangeo.2015.12.008](https://doi.org/10.1016/j.jtrangeo.2015.12.008)

12. **Merlin, L. A.** (2020). A new method using medians to calibrate single-parameter spatial interaction models. *Journal of Transport and Land Use*, 13(1), 49–70. [https://doi.org/10.5198/jtlu.2020.1614](https://doi.org/10.5198/jtlu.2020.1614)

13. **Ortúzar, J. de D., & Willumsen, L. G.** (2011). *Modelling transport* (4th ed.). John Wiley & Sons. [https://doi.org/10.1002/9781119993308](https://doi.org/10.1002/9781119993308)

14. **Pappalardo, L., Manley, E., Sekara, V., & Alessandretti, L.** (2023). Future directions in human mobility science. *Nature Computational Science*, 3, 588–600. [https://doi.org/10.1038/s43588-023-00469-4](https://doi.org/10.1038/s43588-023-00469-4)

15. **Simini, F., Barlacchi, G., Luca, M., & Pappalardo, L.** (2021). A Deep Gravity model for mobility flows generation. *Nature Communications*, 12, 6576. [https://doi.org/10.1038/s41467-021-26752-4](https://doi.org/10.1038/s41467-021-26752-4)

16. **Verma, R., & Ukkusuri, S. V.** (2025). What determines travel time and distance decay in spatial interaction and accessibility? *Journal of Transport Geography*, 122, 104061. [https://doi.org/10.1016/j.jtrangeo.2024.104061](https://doi.org/10.1016/j.jtrangeo.2024.104061)

17. **Wilson, A. G.** (1971). A family of spatial interaction models, and associated developments. *Environment and Planning A*, 3(1), 1–32. [https://doi.org/10.1068/a030001](https://doi.org/10.1068/a030001)

18. **Yang, Y., Herrera, C., Eagle, N., & González, M. C.** (2014). Limits of predictability in commuting flows in the absence of data for calibration. *Scientific Reports*, 4, 5662. [https://doi.org/10.1038/srep05662](https://doi.org/10.1038/srep05662)

19. **Efron, B., & Tibshirani, R. J.** (1993). *An introduction to the bootstrap*. Chapman & Hall.

20. **Holm, S.** (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. [https://www.jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)

21. **Loshchilov, I., & Hutter, F.** (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*. [https://openreview.net/forum?id=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7)

22. **Wilcoxon, F.** (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83. [https://doi.org/10.2307/3001968](https://doi.org/10.2307/3001968)



# Phụ lục phương pháp bổ sung (Supplementary Methods)



## S1. Chi tiết kiến trúc mạng neural GNN và ổn định số học

### S1.1. Các lớp tensor của Urban GNN Encoder
Mạng Urban GNN ánh xạ vector đặc trưng đô thị 26 chiều $\mathbf{x}_{c,i} \in \mathbb{R}^{26}$ và cấu trúc đồ thị bán kính không gian $\mathcal{G}_c = (\mathcal{V}_c, \mathcal{E}_c)$ thành biểu diễn ẩn 64 chiều $\mathbf{h}_{c,i} \in \mathbb{R}^{64}$:

1. **Chiếu nút ban đầu**:
$$
\mathbf{h}_{c,i}^{(0)} = \operatorname{Dropout}\bigl(\operatorname{ReLU}\bigl(\operatorname{LayerNorm}(\mathbf{W}_{\mathrm{in}}\mathbf{x}_{c,i}+\mathbf{b}_{\mathrm{in}})\bigr)\bigr).
$$

2. **Thông điệp điều kiện hóa theo khoảng cách**:
$$
\mathbf{m}_{ji}^{(\ell)} = \mathbf{W}_{\mathrm{msg}}^{(\ell)} [\mathbf{h}_{c,j}^{(\ell-1)} \mathbin{\Vert} \log(1+d_{c,ji})] + \mathbf{b}_{\mathrm{msg}}^{(\ell)}.
$$

3. **Tổng hợp thông điệp**:
$$
\mathbf{a}_{c,i}^{(\ell)} = \frac{1}{\max(\deg(i),1)} \sum_{j\in\mathcal{N}(i)} \mathbf{m}_{ji}^{(\ell)}.
$$

4. **Biến đổi trạng thái nút**:
$$
\widetilde{\mathbf{h}}_{c,i}^{(\ell)} = \operatorname{LayerNorm}\bigl(\operatorname{ReLU}\bigl(\mathbf{a}_{c,i}^{(\ell)} + \mathbf{W}_{\mathrm{self}}^{(\ell)} \mathbf{h}_{c,i}^{(\ell-1)} + \mathbf{b}_{\mathrm{self}}^{(\ell)}\bigr)\bigr).
$$

5. **Cập nhật residual**:
$$
\mathbf{h}_{c,i}^{(\ell)} = \mathbf{h}_{c,i}^{(\ell-1)} + \operatorname{Dropout}\bigl(\widetilde{\mathbf{h}}_{c,i}^{(\ell)}\bigr).
$$

6. **Chiếu đầu ra**:
$$
\mathbf{h}_{c,i} = \mathbf{W}_{\mathrm{out}}\mathbf{h}_{c,i}^{(2)} + \mathbf{b}_{\mathrm{out}} \in \mathbb{R}^{64}.
$$

### S1.2. Ổn định số học và gradient clipping

Trong quá trình huấn luyện, log-likelihood của ZTNB được tính toán thông qua hàm `torch.lgamma`. Để ngăn hiện tượng tràn số hoặc biến mất gradient:

* Tham số trung bình cơ sở được chặn dưới: $\mu_{c,ij} = \operatorname{softplus}(\log T_{c,ij}^{\mathrm{grav}} + \operatorname{residual}_{c,ij}) + 10^{-4}$.
* Tham số phân tán được chặn trong không gian log: $\log \phi_{\mathrm{safe}} = \operatorname{clamp}(\log \phi, \text{min}=-10.0, \text{max}=10.0)$, sau đó $\phi = \exp(\log \phi_{\mathrm{safe}})$.
* Hằng số ổn định $\epsilon = 10^{-8}$ được cộng vào $\mu$ và $\phi$ trong các số hạng logarit; xác suất tại 0 được chuẩn hóa số học qua $\log(1 - P_{\mathrm{NB}}(0)) = \operatorname{log1p}(-\exp(\log P_{\mathrm{NB}}(0)))$ với chặn trên $1.0 - 10^{-7}$. Khi suy luận kỳ vọng điều kiện, mẫu số $1 - P_{\mathrm{NB}}(0)$ được chặn dưới bằng $10^{-6}$.
* Gradient của toàn bộ tham số mô hình được cắt theo chuẩn Euclid tối đa: $\|\mathbf{g}\|_2 \le 5.0$ thông qua `torch.nn.utils.clip_grad_norm_`.



## S2. Dạng tổng quát của toán tử hiệu chỉnh giải tích ($q \in [0, 1]$)

Tham số cường độ hiệu chỉnh $q \in [0, 1]$ điều khiển mức độ can thiệp của thông tin khoảng cách mục tiêu:
* $q = 0$: giữ nguyên dự báo ban đầu của baseline ($\widehat{t}^{(1)} \equiv \widehat{t}^{(0)}$);
* $q = 1$: khớp đầy đủ tỷ trọng luồng theo từng khoảng khoảng cách;
* Nghiên cứu chính cố định $q = 1$.

Ở cấu hình chính $K=8$, tất cả các khoảng đều hoạt động trên 50 thành phố đánh giá.

Quy trình hiệu chỉnh tổng quát được thực hiện qua các bước:

### S2.1. Tập các khoảng hoạt động
Tập các khoảng cự ly có dự báo baseline dương được xác định bởi:
$$
A_c = \{ b \in \{1, \dots, K\} : \widehat{Y}_{c,b}^{(0)} > 0 \}.
$$

### S2.2. Phân phối mục tiêu điều kiện trên các khoảng hoạt động
Tỷ trọng mục tiêu được điều kiện hóa trên các khoảng hoạt động theo:
$$
p_{c,b}^{\mathrm{cond}} = \frac{Y_{c,b} \mathbf{1}(b \in A_c)}{\sum_{r \in A_c} Y_{c,r}}.
$$
Việc điều kiện hóa bảo đảm tổng tỷ trọng trên các khoảng hoạt động bằng 1.

### S2.3. Trọng số hiệu chỉnh mềm
Với mỗi khoảng hoạt động $b \in A_c$, tỷ lệ co giãn mềm được tính theo:
$$
w_{c,b}(q) = \biggl( \frac{p_{c,b}^{\mathrm{cond}}}{\widehat{Y}_{c,b}^{(0)}} \biggr)^q, \qquad b \in A_c.
$$

### S2.4. Hệ số chuẩn hóa và hệ số co giãn
Hệ số chuẩn hóa bảo toàn tổng khối lượng và hệ số co giãn tương ứng là:
$$
Z_c(q) = \sum_{r \in A_c} \widehat{Y}_{c,r}^{(0)} w_{c,r}(q), \qquad s_{c,b}(q) = \frac{w_{c,b}(q)}{Z_c(q)}.
$$

### S2.5. Dự báo sau hiệu chỉnh
Cường độ luồng dự báo sau hiệu chỉnh cho cặp $(i,j)$ được xác định bởi:
$$
\widehat{t}_{c,ij}^{(1)} = s_{c,b(i,j)}(q) \widehat{t}_{c,ij}^{(0)},
$$
trong đó $b(i,j)$ là khoảng cự ly chứa cặp $(i,j)$.

### S2.6. Trường hợp chính $q = 1$
Khi tất cả các khoảng khoảng cách đều hoạt động:
$$
A_c = \{1, \dots, K\},
$$
ta có:
$$
p_{c,b}^{\mathrm{cond}} = Y_{c,b}, \qquad Z_c(1) = 1, \qquad s_{c,b}(1) = \frac{Y_{c,b}}{\widehat{Y}_{c,b}^{(0)}}.
$$
Khi đó, dạng tổng quát thu về đúng toán tử hiệu chỉnh rút gọn được sử dụng trong thân bài.



## S3. Chứng minh giải tích các đặc tính bất biến

### S3.1. Bảo toàn tập hỗ trợ
Vì $s_{c,b}(q) > 0$ trên mọi khoảng hoạt động, một dự báo dương trước hiệu chỉnh vẫn dương sau hiệu chỉnh. Toán tử chỉ hoạt động trên $\Omega_c$, nên không tạo thêm liên kết bên ngoài tập hỗ trợ đã biết:
$$
\widehat{t}_{c,ij}^{(1)} > 0 \quad \Longleftrightarrow \quad \widehat{t}_{c,ij}^{(0)} > 0, \qquad (i,j) \in \Omega_c.
$$

### S3.2. Bảo toàn thứ hạng nội khoảng
Với hai cặp $(i,j)$ và $(u,v)$ cùng thuộc khoảng $b$, ta có:
$$
\frac{\widehat{t}_{c,ij}^{(1)}}{\widehat{t}_{c,uv}^{(1)}} = \frac{s_{c,b}(q) \widehat{t}_{c,ij}^{(0)}}{s_{c,b}(q) \widehat{t}_{c,uv}^{(0)}} = \frac{\widehat{t}_{c,ij}^{(0)}}{\widehat{t}_{c,uv}^{(0)}}.
$$
Do đó, thứ tự tương đối của các cặp trong cùng một khoảng không thay đổi ($\tau = 1$).

### S3.3. Bảo toàn tổng khối lượng dự báo
Gọi $S_c^{(0)}$ là tổng khối lượng dự báo của baseline:
$$
S_c^{(0)} = \sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}^{(0)}.
$$
Tổng khối lượng luồng sau hiệu chỉnh thỏa mãn:
$$
\begin{aligned}
\sum_{(i,j) \in \Omega_c} \widehat{t}_{c,ij}^{(1)} &= S_c^{(0)} \sum_{b \in A_c} \widehat{Y}_{c,b}^{(0)} s_{c,b}(q) \\
&= \frac{S_c^{(0)}}{Z_c(q)} \sum_{b \in A_c} \widehat{Y}_{c,b}^{(0)} w_{c,b}(q) \\
&= S_c^{(0)}.
\end{aligned}
$$
Vì $S_c^{(0)}$ chính là tổng khối lượng dự báo trước hiệu chỉnh, toán tử bảo toàn tổng khối lượng dự báo của baseline.



## S4. Định nghĩa toán học các thước đo sai số phụ

Tất cả các thước đo sai số phụ được tính trên cùng tập hỗ trợ liên vùng dương đã biết $\Omega_c$. CPC vẫn là thước đo chính; các metric dưới đây chỉ phục vụ kiểm tra độ bền của kết quả.

1. **Sai số tuyệt đối trung bình (MAE)**:
$$
\operatorname{MAE}_c = \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} \lvert t_{c,ij} - \widehat{t}_{c,ij} \rvert.
$$

2. **Căn bậc hai sai số bình phương trung bình (RMSE)**:
$$
\operatorname{RMSE}_c = \sqrt{ \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} \bigl( t_{c,ij} - \widehat{t}_{c,ij} \bigr)^2 }.
$$

3. **RMSE chuẩn hóa (NRMSE)**:
$$
\overline{t}_c = \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} t_{c,ij}, \qquad \operatorname{NRMSE}_c = \frac{\operatorname{RMSE}_c}{\overline{t}_c}.
$$

4. **RMSE trên thang log ($\operatorname{RMSE}_{\log 1p}$)**:
$$
\operatorname{RMSE}_{\log 1p,c} = \sqrt{ \frac{1}{\lvert\Omega_c\rvert} \sum_{(i,j)\in\Omega_c} \bigl[ \log(1+t_{c,ij}) - \log(1+\widehat{t}_{c,ij}) \bigr]^2 }.
$$

5. Hệ số tương quan hạng Spearman $\rho_{\mathrm{Spearman},c}$: đo mức độ tương quan đơn điệu giữa các cường độ quan sát và dự báo trên $\Omega_c$. Giá trị lớn hơn biểu thị thứ hạng phù hợp hơn.

6. **Sai số tương đối tổng luồng ($\operatorname{RelError}$)**:
$$
\operatorname{RelError}_c = \frac{ \left\lvert \sum_{(i,j)\in\Omega_c} \widehat{t}_{c,ij} - \sum_{(i,j)\in\Omega_c} t_{c,ij} \right\rvert }{ \sum_{(i,j)\in\Omega_c} t_{c,ij} }.
$$



## S5. Giao thức Bootstrap phân tầng theo fold và kiểm định thống kê

1. **Paired Nonparametric Bootstrap phân tầng theo fold**:
   * Đơn vị lấy mẫu lại là thành phố.
   * Lấy mẫu có hoàn lại riêng trong từng fold.
   * Mỗi fold lấy lại 10 thành phố từ 10 thành phố kiểm tra ban đầu.
   * $M_0$ và $M_1$ luôn được giữ ghép cặp.
   * Không lấy mẫu độc lập các cặp OD.
   * Gọi $\mathcal{C}^{*(r)}$ là multiset gồm 50 thành phố được lấy lại ở bootstrap replicate $r$ ($r = 1, \dots, B$ với $B = 10{,}000$ và $C = 50$):
$$
\overline{\Delta}^{*(r)} = \frac{1}{C} \sum_{c\in\mathcal{C}^{*(r)}} \Delta_c, \qquad r = 1, \dots, B.
$$

2. **Khoảng tin cậy percentile 95%**:
$$
\mathrm{CI}_{95\%} = \bigl[ Q_{0.025}\bigl(\overline{\Delta}^*\bigr), Q_{0.975}\bigl(\overline{\Delta}^*\bigr) \bigr].
$$

3. **Kiểm định Wilcoxon signed-rank hai phía**:
   Kiểm định giả thuyết ghép cặp hai phía trên 50 hiệu số cấp thành phố:
$$
H_0: \operatorname{median}(\Delta_c) = 0, \qquad H_1: \operatorname{median}(\Delta_c) \neq 0.
$$

4. **Hiệu chỉnh Holm–Bonferroni**:
   Đối với họ gồm $M$ giả thuyết:
$$
p_{(k)} \leq \frac{\alpha}{M - k + 1}, \qquad k = 1, \dots, M.
$$
   Các $p$-value được sắp xếp tăng dần. Quy trình step-down dừng tại giả thuyết đầu tiên không thỏa điều kiện bác bỏ.



## S6. Chi tiết kỹ thuật các stress-test độ bền

1. **Tổng hợp nhiễu Total Variation (TV Noise Bisection)**:
   * Áp dụng trên các bin hoạt động có $p_b > 0$ ($b = 1, \dots, K_{\mathrm{act}}$). Vector nhiễu chuẩn hóa được trừ kỳ vọng (centered) theo đúng triển khai trong mã nguồn:
$$
z_b^{\mathrm{ctr}} = z_b - \frac{1}{K_{\mathrm{act}}} \sum_{r=1}^{K_{\mathrm{act}}} z_r, \qquad z_b \sim \mathcal{N}(0, 1).
$$
   * Tỷ trọng nhiễu thu được qua exponential tilting:
$$
p_b(\sigma) = \frac{\exp\bigl(\log p_b + \sigma z_b^{\mathrm{ctr}}\bigr)}{\sum_{r=1}^{K_{\mathrm{act}}} \exp\bigl(\log p_r + \sigma z_r^{\mathrm{ctr}}\bigr)}.
$$
   * Hệ số co giãn $\sigma$ được giải bằng phương pháp chia đôi (bisection solver) để khoảng cách Total Variation đạt đúng mức quy định $\epsilon$:
$$
\operatorname{TV}\bigl(p(\sigma), p\bigr) = \frac{1}{2} \sum_{b=1}^{K_{\mathrm{act}}} \lvert p_b(\sigma) - p_b \rvert = \epsilon.
$$
   Sau công thức, hệ số co giãn $\sigma$ được tìm bằng phương pháp chia đôi (bisection solver) để đạt mức TV yêu cầu $\epsilon$.

2. **Đối chứng Donor Placebo**:

   Training-Mean Donor: Trước hết, phân phối khoảng cách trung bình $\bar{Y}_{D,\mathrm{train}}$ được tính từ toàn bộ các thành phố huấn luyện trong cùng fold. Để bảo đảm so sánh công bằng với điều kiện target-specific, log-ratio giữa $\bar{Y}_{D,\mathrm{train}}$ và phân phối khoảng cách của baseline được centered và sau đó co giãn để có cùng cường độ can thiệp $D_T$ với target $Y_D$:

$$
\tilde{\mathbf{r}}_M^{*} = \tilde{\mathbf{r}}_M \frac{D_T}{D_M},
$$

   trong đó $D_M = \|\tilde{\mathbf{r}}_M\|_2$ là độ lớn can thiệp ban đầu của Training-Mean. Vector đã dose-match $\tilde{\mathbf{r}}_M^{*}$ sau đó được dùng để xây dựng phân phối hiệu chỉnh theo cùng quy trình như các placebo khác.



## S7. Phân tích thăm dò về độ phân giải không gian cấp county

### S7.1. Thiết lập

Phân tích thăm dò này kiểm tra xem việc cung cấp quan sát khoảng cách tổng hợp ở độ phân giải không gian chi tiết hơn cấp thành phố—cụ thể là nhóm theo đơn vị hành chính cấp hạt (county)—có mang lại thông tin bổ sung hay không.

Ranh giới county được lấy từ Database of Global Administrative Areas, phiên bản 4.1 (GADM 4.1) [@gadm41]. Mỗi tract được gán vào county bao quanh tương ứng thông qua phép nối điểm trong đa giác (point-in-polygon) giữa tọa độ tâm tract và polygon của county. Trường hợp tâm tract nằm trên ranh giới polygon hoặc gần bờ biển, quy trình sử dụng phép gán polygon gần nhất trong hệ tọa độ EPSG:5070 với ngưỡng khoảng cách tối đa 5 km. Mỗi tract được gán duy nhất vào một county. GADM chỉ được sử dụng nghiêm ngặt cho bước phân nhóm không gian này, không phải nguồn của tọa độ tâm, đặc trưng đô thị hay luồng OD.

Gọi $g(i)$ là county được gán cho tract $i$. Các cặp OD được nhóm theo **county của điểm xuất phát (origin tract)**:
$$
\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c : g(i) = \ell\right\}.
$$
Tract đích $j$ có thể thuộc cùng county hoặc county khác trong vùng đô thị. Phân phối khoảng cách của nhóm origin-county $\ell$ được định nghĩa:
$$
Y_{c,\ell,b} = \frac{\sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij} \mathbf{1}(d_{c,ij} \in I_b)}{\sum_{(i,j) \in \Omega_{c,\ell}^+} t_{c,ij}}, \qquad \sum_{b=1}^K Y_{c,\ell,b} = 1.
$$
Vì dữ liệu đầu vào giới hạn trong tập tract của vùng đô thị do phòng thí nghiệm cung cấp, $\mathbf{Y}_{c,\ell}$ mô tả phân phối khoảng cách xuất phát từ các tract thuộc county $\ell$ trong vùng đô thị đó, không đại diện cho toàn bộ di chuyển trên toàn địa bàn county ngoài phạm vi nghiên cứu.

Mỗi phân phối $\mathbf{Y}_{c,\ell}$ được sử dụng để hiệu chỉnh các cặp OD có origin tract thuộc county $\ell$. Sau đó, các dự báo đã hiệu chỉnh từ toàn bộ các nhóm county được tập hợp lại thành dự báo hoàn chỉnh cho vùng đô thị:
$$
\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \hat{t}_{c,ij}^{\mathrm{county}} : (i,j) \in \Omega_{c,\ell}^+ \right\},
$$
trong đó $\mathcal{G}_c$ là tập hợp các county xuất hiện trong tập dữ liệu của vùng đô thị $c$.

Quan trọng là việc chuyển độ phân giải quan sát từ cấp thành phố sang cấp county không làm thay đổi phạm vi đánh giá: mô hình vẫn tái tạo và được đánh giá trên toàn bộ tập hỗ trợ luồng dương $\Omega_c$ của vùng đô thị mục tiêu; chỉ có tín hiệu giám sát tổng hợp trong bước hiệu chỉnh trở nên chi tiết hơn theo không gian.

Trong số 50 vùng đô thị của benchmark, có đúng 39 vùng single-county (nơi toàn bộ các tract thuộc cùng một county duy nhất, do đó $\lvert\mathcal{G}_c\rvert = 1$). Với 39 vùng này, phân hoạch theo county hoàn toàn trùng khớp với phân hoạch cấp thành phố, dẫn đến $M1_{\mathrm{county}} \equiv M1_{\mathrm{city}}$ và $\Delta\mathrm{CPC}_{\mathrm{res},c} = 0$ về mặt toán học. Chỉ có 11 vùng đô thị trải rộng qua từ 2 đến 7 county tạo ra phân hoạch mới thực sự.

### S7.2. Kết quả

Trên toàn bộ 50 vùng đô thị, mức tăng bổ sung pooled từ hiệu chỉnh cấp county so với hiệu chỉnh cấp thành phố là rất nhỏ:

$$
\Delta\mathrm{CPC}_{\mathrm{res}} = +0.00014, \quad \text{95\% CI } [+0.00002,\,+0.00028], \quad \text{Wilcoxon } p = 0.0064.
$$


Mức tăng pooled khiêm tốn này chịu chi phối bởi 39 vùng single-county có mức tăng bằng 0 tuyệt đối theo cấu trúc.

Đối với nhóm 11 vùng đô thị multi-county (chiếm 22% tập benchmark), hiệu chỉnh cấp county đạt mức cải thiện tại 9/11 vùng, với mức tăng bổ sung trung bình là $+0.00063$ (Bảng S1 và Hình S1).

![Hình S1](figures/fig_s1_spatial_resolution.png)
**Hình S1. So sánh mức tăng CPC của hiệu chỉnh cấp thành phố và cấp county trên 11 vùng đô thị multi-county. Phân tích mang tính thăm dò; 39 vùng single-county không được hiển thị vì hai cách phân nhóm tương đương về mặt toán học.**

### Bảng S1: Kết quả mô tả theo thành phố cho nhóm phân tích độ phân giải không gian đa county

*Bảng so sánh zero-shot baseline ($M_0$), hiệu chỉnh oracle cấp city ($M1_{\mathrm{city}}$) và hiệu chỉnh oracle có điều kiện theo origin-county ($M1_{\mathrm{county}}$) cho 11 bộ dữ liệu đô thị có các tract được gán vào nhiều hơn một county. Mức tăng do độ phân giải được định nghĩa là $\Delta\mathrm{CPC}_{\mathrm{res},c}=\mathrm{CPC}(M1_{\mathrm{county}})-\mathrm{CPC}(M1_{\mathrm{city}})$. Các giá trị là ước lượng mô tả ở cấp city. Không báo cáo khoảng tin cậy hoặc kiểm định giả thuyết cho subgroup nếu không có artifact bất định riêng đã được xác minh.*

| Thành phố | Số county gốc | $M_0$ CPC | $M1_{\mathrm{city}}$ CPC | $M1_{\mathrm{county}}$ CPC | $\Delta\mathrm{CPC}_{\mathrm{city}}$ | $\Delta\mathrm{CPC}_{\mathrm{county}}$ | $\Delta\mathrm{CPC}_{\mathrm{res}}$ |
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
| **Trung bình đa county** | — | — | — | — | — | — | **+0.000626** |
| **Số thành phố tăng dương** | — | — | — | — | — | — | **9 / 11** |

### S7.3. Giới hạn diễn giải

Kết quả phân tích cấp county cần được diễn giải với các giới hạn nghiêm ngặt sau:

1. **Quy mô mẫu nhỏ và bằng chứng mô tả**: Phân tích chỉ dựa trên 11 vùng đô thị multi-county. Do không có ước lượng bất định phân tầng riêng cho tập con này, kết quả 9/11 vùng cải thiện chỉ mang tính chất mô tả thực nghiệm, không đủ cơ sở để khẳng định tính quy luật thống kê tổng quát.
2. **Ranh giới hành chính so với ranh giới chức năng**: County (đơn vị hành chính cấp hạt) là ranh giới quản lý hành chính lịch sử, không được thiết kế dựa trên lưu vực đi lại, hành lang giao thông hay cấu trúc phân vùng chức năng đô thị. Vì vậy, việc phân nhóm theo county không nhất thiết phản ánh đúng tính không đồng nhất của hành vi di chuyển.
3. **Phạm vi không gian không đầy đủ**: Các nhóm county chỉ bao gồm các tract nằm trong ranh giới vùng đô thị do phòng thí nghiệm cung cấp, không đại diện cho toàn bộ luồng di chuyển trên toàn diện tích địa giới của các county đó.
4. **Không chứng minh quan hệ nhân quả hay bảo đảm thực tế**: Việc gán tâm tract bằng phương pháp hình học và sử dụng phân phối oracle không phản ánh các sai số ghép nối thực tế. Thí nghiệm không chứng minh rằng tăng độ phân giải không gian nói chung sẽ luôn cải thiện việc tái tạo ma trận OD trong các ứng dụng thực tế.
