# Cải thiện tái tạo cường độ luồng OD zero-shot bằng phân phối di chuyển theo khoảng cách của thành phố mục tiêu

---

## Tóm tắt

Ma trận nguồn–đích (origin–destination, OD) là đầu vào quan trọng cho phân tích giao thông và quy hoạch đô thị, nhưng dữ liệu cường độ luồng OD chi tiết của thành phố mục tiêu thường khó thu thập. Các nghiên cứu về sử dụng ngữ cảnh đô thị và khoảng cách địa lý đã phát triển các mô hình zero-shot có khả năng tái tạo các luồng di chuyển mà không sử dụng dữ liệu quan sát luồng OD của thành phố mục tiêu. Nghiên cứu này xem xét liệu một phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu có thể cải thiện kết quả tái tạo cường độ luồng OD của mô hình zero-shot hay không. Phân phối này được tổng hợp từ dữ liệu quan sát luồng của chính thành phố mục tiêu và chỉ cung cấp tỷ trọng khối lượng theo các khoảng khoảng cách, không cung cấp cường độ của từng cặp OD riêng lẻ.

Mô hình và toàn bộ tham số đã huấn luyện được giữ cố định, còn phân phối di chuyển này chỉ được sử dụng để hiệu chỉnh tại thời điểm suy luận nhằm tái phân bổ khối lượng luồng dự báo giữa các khoảng khoảng cách trên tập hỗ trợ liên vùng dương đã biết. Trong thí nghiệm chính, khung nghiên cứu được đánh giá bằng 5-fold cross-validation theo thành phố trên 50 bộ dữ liệu vùng đô thị của Hoa Kỳ. Hiệu chỉnh cấp thành phố tạo ra mức cải thiện nhỏ nhưng tương đối nhất quán, làm CPC trung bình tăng 0.00354, với khoảng tin cậy bootstrap 95% của mức cải thiện nằm trong khoảng từ 0.0026 đến 0.0045, và cải thiện kết quả tại 45 trong tổng số 50 thành phố. Mức cải thiện giảm khi độ phân giải và chất lượng của phân phối quan sát suy giảm. Kết luận của nghiên cứu chỉ áp dụng cho bài toán tái tạo cường độ trên tập hỗ trợ liên vùng dương đã biết, không mở rộng sang dự báo sự tồn tại của liên kết hoặc nhận diện các ô có luồng bằng không. Nhìn chung, phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu cung cấp một tín hiệu tổng hợp gọn nhẹ giúp mang lại mức cải thiện nhỏ nhưng tương đối nhất quán cho một mô hình cross-city cố định trong benchmark có điều kiện theo support này.

**Từ khóa:** ma trận nguồn–đích; tái tạo cường độ OD; phân phối di chuyển theo khoảng cách; zero-shot; học chuyển giao giữa các thành phố; quan sát tổng hợp; di chuyển không gian.

---

# Mục 1: Giới thiệu

Ma trận nguồn–đích (origin–destination, OD) mô tả khối lượng di chuyển giữa các cặp đơn vị không gian và cung cấp một biểu diễn tương tác không gian ở cấp độ quần thể. Dữ liệu này hỗ trợ nhiều bài toán giao thông và đô thị, bao gồm mô hình hóa nhu cầu đi lại, quy hoạch mạng lưới, đánh giá khả năng tiếp cận và nghiên cứu cấu trúc đô thị [@wilson1971family; @ortuzar2011modelling; @barbosa2018humanmobility]. Vì vậy, ước lượng luồng OD đáng tin cậy có giá trị không chỉ trong mô tả di chuyển quan sát được mà còn trong phân tích sự khác biệt của nhu cầu di chuyển giữa các bối cảnh địa lý.

Tuy nhiên, việc thu thập cường độ OD chi tiết cho mọi thành phố mục tiêu gặp nhiều khó khăn. Khảo sát đi lại có thể tốn kém và thưa về mặt không gian, trong khi dữ liệu di chuyển thu thập thụ động có thể chịu ảnh hưởng của độ phủ không đầy đủ, sai lệch mẫu, sai lệch do quy trình xử lý, hạn chế truy cập và tính đại diện chưa rõ ràng [@gallotti2024distorted; @pappalardo2023future]. Hơn nữa, luồng di chuyển không chỉ được quyết định bởi khoảng cách địa lý. Cấu trúc luồng còn phản ánh phân bố dân cư và việc làm, sử dụng đất, hạ tầng giao thông, hình thái đô thị và hành vi đặc thù của từng thành phố. Do đó, các mô hình chuyển giao giữa thành phố vẫn có thể mang sai lệch có hệ thống tại miền mục tiêu khi không có thông tin hiệu chỉnh địa phương [@yang2014limits].

Các mô hình neural mobility gần đây kết hợp thuộc tính địa lý, biểu diễn không gian và tương tác phụ thuộc khoảng cách để học các quy luật luồng có khả năng chuyển giao giữa các khu vực [@simini2021deepgravity; @guo2025ugnn; @enaya2026transgm]. Những phương pháp này làm giảm nhu cầu phải xây dựng một mô hình độc lập từ đầu cho từng thành phố. Tuy nhiên, một mô hình cross-city đã đóng băng vẫn phải suy luận cấu trúc di chuyển của thành phố mục tiêu từ các đặc trưng đầu vào sẵn có. Cần lưu ý rằng baseline zero-shot không sử dụng giá trị cường độ luồng OD của thành phố mục tiêu để huấn luyện hoặc cập nhật tham số, và phạm vi dự báo được điều kiện hóa trên tập hỗ trợ dương đã biết. Mặc dù baseline biết khoảng cách địa lý của từng cặp OD, mô hình không trực tiếp quan sát cách tổng khối lượng di chuyển của thành phố mục tiêu được phân bổ giữa các khoảng khoảng cách. Phân phối cự ly chuyến đi này là một đặc trưng tổng hợp quan trọng, phản ánh lực cản không gian và cấu trúc di chuyển đặc thù của thành phố, trong khi quy luật suy giảm theo khoảng cách thực tế lại thay đổi theo bộ dữ liệu, thang không gian, mục đích chuyến đi và bối cảnh đô thị [@lenormand2016comparison; @verma2025distance].

Nghiên cứu này kiểm tra liệu phần thông tin còn thiếu đó có thể được bổ sung bằng một quan sát tổng hợp gọn nhẹ: phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu, được định nghĩa là tỷ trọng khối lượng chuyến đi quan sát được nằm trong các khoảng khoảng cách. Là một tín hiệu tổng hợp số chiều thấp, phân phối này chỉ mô tả cách tổng khối lượng được phân bổ theo khoảng cách và không tiết lộ cường độ của từng cặp OD riêng lẻ. Thay vì huấn luyện lại hoặc fine-tune mô hình dự báo, phân phối này chỉ được sử dụng tại thời điểm suy luận để tái phân bổ giải tích khối lượng luồng dự báo giữa các khoảng khoảng cách trong khi backbone và toàn bộ tham số đã huấn luyện được giữ cố định, đồng thời bảo toàn tổng khối lượng di chuyển dự báo và thứ hạng tương đối của các cặp OD trong từng khoảng. Phép hiệu chỉnh này được thiết kế có chủ ý theo dạng đơn giản và đóng. Vai trò của nó không phải là đề xuất một thuật toán hiệu chỉnh tổng quát mới, mà là một công cụ thực nghiệm để đo lượng thông tin bổ sung chứa trong một tín hiệu tổng hợp có số chiều thấp và đặc thù cho thành phố mục tiêu.

Nghiên cứu được tổ chức quanh hai câu hỏi chính: (1) **RQ1—Giá trị thông tin bổ sung:** Khi được đánh giá trên cùng một tập hỗ trợ liên vùng dương đã biết, việc đưa phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu vào như thông tin bổ sung duy nhất tại bước hiệu chỉnh có cải thiện tái tạo cường độ luồng OD zero-shot so với mô hình cross-city đã đóng băng (với toàn bộ tham số đã huấn luyện được giữ cố định và không sử dụng dữ liệu cường độ luồng của thành phố mục tiêu) hay không? (2) **RQ2—Độ phân giải và chất lượng quan sát:** Giá trị của tín hiệu tổng hợp này thay đổi như thế nào theo số lượng khoảng khoảng cách, mức tổng hợp không gian dưới cấp vùng đô thị, chất lượng quan sát, thứ tự của các khoảng và tính đặc thù theo thành phố? Cả hai câu hỏi được đánh giá trong phạm vi tái tạo cường độ trên tập các cặp OD liên vùng có luồng dương đã biết. Nghiên cứu không dự báo sự tồn tại của các liên kết OD chưa quan sát hoặc phân loại các cặp có luồng bằng 0. Ngoài ra, phân phối này được trích xuất trực tiếp từ luồng tham chiếu của chính thành phố mục tiêu nên được xem là một **quan sát tổng hợp oracle**. Thiết lập này đóng vai trò như một thí nghiệm thăm dò giá trị thông tin có kiểm soát hoặc một thí nghiệm định tính khả thi nhằm kiểm tra xem một tín hiệu tổng hợp có số chiều thấp có chứa thông tin bổ sung đủ rõ để tạo động lực cho các nghiên cứu thu thập hoặc ước lượng phân phối này trong tương lai hay không, chứ chưa chứng minh khả năng triển khai vận hành với độ chính xác, chi phí, khả năng truy cập hoặc đặc tính quyền riêng tư tương đương từ một nguồn dữ liệu độc lập.

Nghiên cứu sử dụng kiểm định chéo liên thành phố 5-fold trên 50 vùng đô thị của Hoa Kỳ, trong đó mỗi thành phố được đánh giá ở fold mà thành phố đó không tham gia huấn luyện mô hình. Backbone neural và toàn bộ tham số đã huấn luyện được giữ cố định trước bước hiệu chỉnh cho thành phố mục tiêu. Thí nghiệm chính sử dụng phân phối cấp thành phố. Các phân tích bổ sung khảo sát ảnh hưởng của số lượng khoảng khoảng cách, mức tổng hợp không gian theo quận, sai số quan sát, hoán vị khoảng cách, phân phối từ thành phố khác, các lần khởi tạo ngẫu nhiên và kiến trúc mô hình khác nhau.

Kết quả cho thấy phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu tạo ra mức cải thiện nhỏ nhưng tương đối nhất quán so với baseline zero-shot đã đóng băng: CPC trung bình tăng 0.00354 và cải thiện kết quả tại 45 trong tổng số 50 thành phố. Mức cải thiện giảm khi độ phân giải và chất lượng của phân phối quan sát suy giảm, đồng thời phụ thuộc vào việc các khoảng khoảng cách được giữ đúng thứ tự và đặc thù cho thành phố mục tiêu.

Nghiên cứu có bốn đóng góp chính: (1) hình thức hóa một thí nghiệm có điều kiện theo tập hỗ trợ liên vùng dương đã biết nhằm cô lập giá trị thông tin bổ sung của phân phối khoảng cách cấp thành phố trong khi giữ cố định mô hình dự báo; (2) đánh giá tín hiệu này bằng kiểm định chéo liên thành phố 5-fold trên 50 vùng đô thị, với bất định được lượng hóa ở cấp thành phố thay vì xem các cặp OD trong cùng thành phố là độc lập; (3) xác định các điều kiện làm tín hiệu trở nên hữu ích thông qua phân tích độ phân giải, nhiễu, hoán vị, donor placebo, khởi tạo và kiến trúc; và (4) diễn giải cơ chế hiệu chỉnh như sự tái phân bổ khối lượng liên khoảng có bảo toàn thứ hạng nội khoảng, đồng thời phân biệt rõ liên hệ thực nghiệm với bằng chứng nhân quả và đánh giá oracle với triển khai vận hành.

Phần còn lại của bài báo được tổ chức như sau. Mục 2 tổng quan các nghiên cứu liên quan về mô hình tương tác không gian, dự báo mobility neural giữa các thành phố, hiệu chỉnh bằng thông tin tổng hợp và hạn chế của dữ liệu di chuyển. Mục 3 trình bày dữ liệu benchmark, biểu diễn không gian, mô hình luồng zero-truncated, toán tử hiệu chỉnh và protocol đánh giá. Mục 4 báo cáo kết quả thực nghiệm cùng các phân tích chẩn đoán. Mục 5 thảo luận cách diễn giải, ý nghĩa, hạn chế và hướng nghiên cứu tương lai. Mục 6 kết luận, tiếp theo là tuyên bố về dữ liệu và mã nguồn, các declarations và danh mục tài liệu tham khảo.

---

# Mục 2: Nghiên cứu liên quan

## 2.1. Mô hình tương tác không gian và hiệu chỉnh dựa trên khoảng cách

Mô hình hóa luồng điểm đi–điểm đến từ lâu đã được nghiên cứu thông qua các mô hình tương tác không gian, trong đó chuyển động giữa origin và destination được biểu diễn theo khả năng phát sinh chuyến đi, mức độ thu hút và độ phân cách hoặc chi phí di chuyển. Công trình dựa trên entropy của Wilson đặt nền tảng cho một họ mô hình tương tác không gian có ràng buộc, còn các khung mô hình giao thông sau đó tích hợp phân bổ chuyến đi vào hệ thống phân tích nhu cầu rộng hơn [@wilson1971family; @ortuzar2011modelling]. Trong hướng tiếp cận này, khoảng cách địa lý hoặc chi phí di chuyển tổng quát đóng vai trò impedance, làm giảm tương tác kỳ vọng khi hai địa điểm cách xa nhau hơn.

Dạng hàm và tham số suy giảm theo khoảng cách thường cần được hiệu chỉnh từ dữ liệu thực nghiệm. Phương pháp của Hyman là một trong những cách tiếp cận sớm có hệ thống để hiệu chỉnh tham số cản trở của mô hình gravity theo cự ly chuyến đi trung bình quan sát được [@hyman1969calibration]. Các nghiên cứu gần đây cho thấy những thống kê tóm tắt gọn nhẹ khác, chẳng hạn trung vị thời gian di chuyển, cũng có thể giúp xác định một tham số impedance khi các giả định cấu trúc phù hợp được thỏa mãn [@merlin2020medians]. Tuy nhiên, các đánh giá so sánh cũng chỉ ra rằng không có một quy luật phân bổ chuyến đi hoặc dạng distance-decay duy nhất luôn hoạt động tốt trên mọi bộ dữ liệu và thang không gian [@lenormand2016comparison]. Mẫu hình suy giảm thực nghiệm có thể thay đổi theo phương thức, mục đích chuyến đi, mức độ đô thị hóa và điều kiện kinh tế–xã hội [@verma2025distance].

Các nghiên cứu trên xác lập hai nguyên lý liên quan trực tiếp đến đề tài này. Thứ nhất, khoảng cách là một biến tổ chức trung tâm của luồng không gian. Thứ hai, quan sát đặc thù của miền mục tiêu về cấu trúc cự ly chuyến đi có thể chứa thông tin không thể khôi phục hoàn toàn từ một hàm cản trở cố định được chuyển giao chung cho mọi nơi. Nghiên cứu hiện tại kế thừa nhận định cổ điển này nhưng không ước lượng một hệ số suy giảm tham số của mô hình gravity. Thay vào đó, nghiên cứu sử dụng vector tỷ lệ luồng quan sát theo các khoảng khoảng cách như một ràng buộc vĩ mô phi tham số áp dụng lên dự báo đã có.

## 2.2. Sinh dữ liệu di chuyển và mô hình không gian dựa trên học máy

Sự gia tăng của dữ liệu thuộc tính địa lý và quan sát di chuyển đã tạo điều kiện cho các mô hình học máy biểu diễn tương tác phi tuyến giữa đặc trưng origin, đặc trưng destination và độ phân cách không gian. Deep Gravity cho thấy kiến trúc neural có thể kết hợp đặc trưng địa lý với khoảng cách để sinh luồng di chuyển và khái quát ra ngoài từng khu vực huấn luyện riêng lẻ [@simini2021deepgravity]. Các mô hình neural có nhận thức địa lý gần đây tiếp tục tích hợp cấu trúc không gian và biểu diễn có khả năng chuyển giao nhằm cải thiện dự báo tại những đô thị chưa xuất hiện trong huấn luyện [@guo2025ugnn]. Các khung gravity có khả năng chuyển giao cũng nhấn mạnh bài toán chuyển giao giữa thành phố và thích nghi giữa các hệ thống đô thị không đồng nhất [@enaya2026transgm].

Khái quát hóa giữa các thành phố vẫn là một bài toán khó vì ánh xạ từ bối cảnh đô thị sang luồng di chuyển không bất biến theo không gian. Mô hình huấn luyện trên các thành phố nguồn có thể học được quy luật chung nhưng vẫn biểu diễn sai tỷ trọng đặc thù giữa các chuyến đi cục bộ và đường dài tại thành phố mục tiêu. Nghiên cứu trước về khả năng dự báo luồng đi làm cho thấy việc thiếu dữ liệu hiệu chỉnh địa phương tạo ra giới hạn đáng kể đối với độ chính xác [@yang2014limits]. Hạn chế này không tự biến mất khi khoảng cách cặp được đưa vào đầu vào: khoảng cách cho mô hình biết hai vùng cách nhau bao xa, nhưng không cho biết tỷ lệ thực nghiệm của tổng luồng tại thành phố mục tiêu được phân bổ vào dải cự ly đó.

Vì vậy, nghiên cứu hiện tại giải quyết một vấn đề bổ sung cho hướng thiết kế kiến trúc neural. Nghiên cứu giả định mô hình cross-city đã được huấn luyện, sau đó kiểm tra liệu một quan sát tổng hợp gọn nhẹ của miền mục tiêu có thể sửa phần sai lệch vĩ mô còn lại mà không cập nhật tham số đã học hay không. Cách thiết kế này tách giá trị thông tin của quan sát mục tiêu khỏi những cải thiện có thể phát sinh do huấn luyện bổ sung, fine-tuning hoặc thay đổi kiến trúc.

## 2.3. Quan sát tổng hợp như một ràng buộc hiệu chỉnh

Hiệu chỉnh bằng quan sát tổng hợp nằm giữa hai cực: dự báo hoàn toàn không có quan sát tại miền mục tiêu và ước lượng từ một ma trận OD mục tiêu đầy đủ. Các mô hình có ràng buộc cổ điển sử dụng tổng lượng chuyến đi theo điểm đi và điểm đến hoặc moment của chi phí di chuyển để áp đặt tính nhất quán về production, attraction hoặc impedance [@wilson1971family; @hyman1969calibration; @ortuzar2011modelling]. Những ràng buộc này có thể mang nhiều thông tin vì chúng tóm tắt thuộc tính của toàn hệ thống luồng nhưng chỉ cần số lượng đại lượng quan sát nhỏ hơn rất nhiều so với số ô OD.

Phần lớn phương pháp hiệu chỉnh khoảng cách truyền thống ước lượng một hoặc một số tham số của mô hình tương tác không gian. Ngược lại, quan sát mục tiêu trong nghiên cứu này là phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu, biểu diễn tỷ trọng của tổng khối lượng di chuyển được phân bổ vào từng khoảng khoảng cách riêng biệt. Phân phối này được dùng để điều chỉnh trực tiếp khối lượng dự báo giữa các khoảng cự ly thực nghiệm. Sự khác biệt này cho phép nghiên cứu trực tiếp độ chi tiết của quan sát: thay đổi số lượng khoảng khoảng cách sẽ làm thay đổi số chiều và độ phân giải cự ly của tín hiệu, còn biến thể có điều kiện theo quận sẽ điều chỉnh độ phân giải không gian bằng cách cung cấp các phân phối riêng cho từng nhóm đơn vị xuất phát. Trong trường hợp thứ hai, phân nhóm quận chỉ thay đổi độ phân giải của quan sát dùng để hiệu chỉnh; đầu ra vẫn là dự báo cho toàn bộ bộ dữ liệu thành phố.

Tín hiệu tổng hợp trong nghiên cứu này không đồng nhất với tổng lượng chuyến đi theo điểm đi và điểm đến hoặc một mẫu các cặp OD được quan sát trực tiếp. Mỗi loại quan sát ràng buộc một khía cạnh khác nhau của ma trận luồng chưa biết. Phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu chỉ ràng buộc tỷ lệ của tổng khối lượng di chuyển của thành phố trên tập hỗ trợ liên vùng dương đã biết được phân bổ vào từng khoảng cự ly; bản thân nó không xác định cặp origin–destination cụ thể nào phải nhận nhiều luồng hơn trong cùng một khoảng. Vì vậy, giá trị tiềm năng của tín hiệu phụ thuộc đồng thời vào ràng buộc cự ly vĩ mô và cấu trúc cặp mà baseline đã học được.

## 2.4. Mô hình hóa dữ liệu đếm trên support dương

Cường độ OD là dữ liệu đếm không âm và thường có phương sai lớn hơn trung bình, do đó cần các phân phối có khả năng biểu diễn overdispersion. Negative binomial regression là một khung phổ biến cho loại dữ liệu này [@hilbe2011negative]. Khi bộ dữ liệu chỉ chứa các quan sát dương, việc áp dụng một likelihood đếm thông thường mà không xử lý phần khối lượng xác suất tại 0 sẽ không phản ánh đúng cơ chế lấy mẫu. Mô hình đếm cắt tại 0 khắc phục điều này bằng cách đặt likelihood có điều kiện theo các quan sát luồng dương [@grogger1991truncated].

Sự phân biệt này là nền tảng của phạm vi nghiên cứu. Bài toán được xác định là **tái tạo cường độ có điều kiện theo support**: các cặp OD có trong benchmark được biết là có luồng tham chiếu dương, và mô hình ước lượng cường độ dương của chúng. Các cặp không xuất hiện trong support dương được xem là chưa quan sát chứ không phải các giá trị 0 đã được xác nhận. Vì vậy, nghiên cứu không giải quyết sự hình thành liên kết, phân loại giá trị 0 hoặc khôi phục ma trận OD đầy đủ. Cách xây dựng thống kê này làm cho likelihood phù hợp với mẫu quan sát và tránh đưa ra claim toàn ma trận mạnh hơn khả năng hỗ trợ của dữ liệu.

## 2.5. Chất lượng dữ liệu di chuyển, mức độ tổng hợp và ranh giới quyền riêng tư

Nghiên cứu di chuyển con người sử dụng nhiều nguồn như khảo sát, hồ sơ hành chính, dữ liệu mạng di động, dịch vụ dựa trên vị trí và các dấu vết số khác [@barbosa2018humanmobility]. Những nguồn này khác nhau về độ phủ dân số, độ phân giải không gian–thời gian, cơ chế lấy mẫu và tiền xử lý. Các khác biệt đó có thể làm sai lệch mẫu hình di chuyển được suy luận và gây khó khăn cho việc so sánh giữa thành phố hoặc nền tảng [@gallotti2024distorted; @pappalardo2023future]. Vì vậy, phân phối khoảng cách tổng hợp được ước lượng từ một nguồn bên ngoài có thể lệch có hệ thống so với phân phối tương ứng trong benchmark OD tham chiếu, thay vì chỉ chứa nhiễu ngẫu nhiên độc lập.

Việc tổng hợp dữ liệu cũng không tự động tạo ra bảo đảm quyền riêng tư chính thức. Bản ghi di chuyển vẫn có thể mang tính nhận dạng ngay cả khi đã giảm độ chi tiết không gian hoặc thời gian [@demontjoye2013unique], trong khi bảo đảm differential privacy ở cấp người dùng cho thống kê vị trí công bố đòi hỏi các đánh đổi thực tiễn không đơn giản [@houssiau2022differential]. Do đó, số chiều thấp của phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu chỉ nên được hiểu là một thuộc tính của giao diện quan sát, không phải bằng chứng rằng quan sát này bảo vệ quyền riêng tư. Việc xác lập quyền riêng tư đòi hỏi phải chỉ rõ cơ chế sinh dữ liệu, threat model, cơ chế công bố và phân tích rủi ro chính thức hoặc thực nghiệm.

Vì những lý do trên, nghiên cứu hiện tại sử dụng phân phối oracle được trích xuất từ luồng tham chiếu của thành phố mục tiêu, sau đó đưa vào các nhiễu có kiểm soát để đánh giá độ nhạy với sai số quan sát. Thiết kế này cô lập hàm lượng thông tin của phân phối nhưng không thay thế cho việc kiểm chứng trong tương lai bằng các quan sát tổng hợp được thu thập độc lập. Nó cũng tránh gán các đặc tính chưa được xác minh về nguồn gốc, khả năng truy cập hoặc quyền riêng tư cho một nguồn vận hành tiềm năng.

## 2.6. Khoảng trống nghiên cứu và vị trí của nghiên cứu hiện tại

Các nghiên cứu đã tổng quan xác lập vai trò quan trọng của khoảng cách trong tương tác không gian, nhu cầu hiệu chỉnh địa phương, khả năng chuyển giao ngày càng cao của mô hình neural mobility và giá trị của một số ràng buộc tổng hợp. Tuy nhiên, một câu hỏi thông tin cụ thể vẫn chưa được làm rõ đầy đủ: **sau khi mô hình cross-city đã học từ bối cảnh đô thị tĩnh và khoảng cách giữa các cặp vùng, phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu còn cung cấp thêm bao nhiêu giá trị, và giá trị đó duy trì trong những điều kiện quan sát nào?**

Nghiên cứu này không xem phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu là sự thay thế cho ma trận OD mục tiêu, cũng không xem nó là một đặc trưng mới để huấn luyện lại mạng neural. Thay vào đó, phân phối này là tín hiệu tổng hợp duy nhất về cường độ của thành phố mục tiêu được đưa vào sau khi mô hình đã huấn luyện xong. Thiết kế backbone đóng băng, đánh giá liên thành phố và các đối chứng chẩn đoán được khớp phù hợp được sử dụng để phân biệt giá trị thông tin đặc thù của target với hiệu ứng thích nghi mô hình, distance decay chung hoặc rescaling tùy ý. Các nghiên cứu hiện có chưa trực tiếp kiểm tra giá trị cải thiện biên của tín hiệu này trên một baseline cross-city đã được huấn luyện và giữ cố định. Phân tích cũng tách riêng độ phân giải cự ly với độ phân giải không gian dưới cấp vùng đô thị và kiểm tra độ trung thực của quan sát bằng nhiễu có kiểm soát cùng placebo về thứ tự ngữ nghĩa.

Cách định vị này làm hẹp phạm vi claim nhưng giúp phạm vi đánh giá trở nên minh bạch. Nghiên cứu đặt câu hỏi liệu một đại lượng tổng hợp mục tiêu có số chiều thấp và đã biết có cải thiện ước lượng cường độ của các liên kết OD liên vùng có luồng dương đã biết trong một hệ thống dự báo cố định hay không. Nghiên cứu không tuyên bố tái tạo support chưa biết của mạng, chứng minh khả năng thu thập vận hành của tín hiệu tổng hợp này hoặc cung cấp bảo đảm quyền riêng tư chính thức. Mục 3 chuyển khoảng trống nghiên cứu này thành các định nghĩa dữ liệu, mô hình, toán tử hiệu chỉnh và protocol đánh giá liên thành phố được dùng trong thực nghiệm.

---

# Mục 3: Nguồn dữ liệu, đơn vị không gian và phương pháp luận

---

## 3.1. Nguồn dữ liệu và biểu diễn không gian

$$I_b = [a_{b-1}, a_b), \qquad b = 1, \dots, K$$

$$d_{ij} = 2R \arcsin \left( \sqrt{ \sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos(\varphi_i)\cos(\varphi_j) \sin^2\left(\frac{\Delta\lambda}{2}\right) } \right)$$

$$\mathcal{D}_{\mathrm{train}}^{(f)} = \left\{ d_{ij} : c \in \mathcal{C}_{\mathrm{train}}^{(f)}, (i,j) \in \Omega_c^+, i \ne j, d_{ij} > 0 \right\}$$

$$a_b = Q_{b/K}\left(\mathcal{D}_{\mathrm{train}}^{(f)}\right), \qquad b = 1, \dots, K-1$$

Nghiên cứu được thực hiện trên 50 thành phố của Hoa Kỳ. Mỗi thành phố được biểu diễn dưới dạng một tập hợp các đơn vị không gian cấp tract. Mỗi tract có tọa độ tâm $\mathbf{s}_i = (\operatorname{lon}_i, \operatorname{lat}_i)$ và 26 đặc trưng mô tả bối cảnh đô thị, bao gồm 13 đặc trưng Census, 8 đặc trưng điểm quan tâm (POI) và 5 đặc trưng mạng lưới đường. Các đặc trưng này được lấy từ bộ dữ liệu do Lab tổng hợp. Nguồn ban đầu, năm dữ liệu, phiên bản và quy trình tiền xử lý của từng nhóm đặc trưng đang được xác minh và sẽ được bổ sung đầy đủ trước khi công bố nghiên cứu. Nghiên cứu không sử dụng hình học polygon của tract. Thay vào đó, mỗi tract được biểu diễn về mặt không gian bằng tọa độ tâm. Khoảng cách giữa các cặp tract $d_{ij}$ được tính bằng công thức Haversine với bán kính Trái Đất $R=6371$ km. Với mỗi fold $f$, các biên khoảng cách $a_b = Q_{b/K}(\mathcal{D}_{\mathrm{train}}^{(f)})$ ($b=1,\dots,K-1$) được xác định độc lập theo phân vị cặp luồng (pair-weighted) từ tập các thành phố huấn luyện $\mathcal{D}_{\mathrm{train}}^{(f)}$, với $a_0=0$ và $a_K=\infty$. Không sử dụng thông tin thành phố kiểm tra để thiết lập khoảng cách.

---

## 3.2. Đơn vị không gian và độ phân giải của quan sát: Cấu hình chuẩn cấp thành phố (`M1_city`*)*

$$\Omega_c^+ = \left\{(i,j) : t_{ij} \ge 1\right\}$$

$$F_{c,b} = \sum_{(i,j) \in \Omega_c^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1} \le d_{ij} < a_b)$$

$$Y_{D,c,b} = \frac{F_{c,b}}{\sum_{r=1}^K F_{c,r}}, \qquad \sum_{b=1}^K Y_{D,c,b} = 1$$

Bộ dữ liệu do Lab cung cấp được tổ chức theo từng thành phố. Mỗi thành phố $c$ bao gồm một tập các tract và các cặp OD dương giữa những tract đó. Tract là đơn vị không gian cơ sở của mô hình, trong khi city là đơn vị chia dữ liệu, thực hiện zero-shot transfer và đánh giá kết quả. Đối với mỗi thành phố mục tiêu, mô hình dự báo cường độ cho toàn bộ tập cặp OD được quan sát $\Omega_c^+ = \{(i,j):t_{ij}\geq1\}$. Các thử nghiệm chính sử dụng một phân phối di chuyển theo khoảng cách duy nhất ở cấp city. Tổng luồng tham chiếu của city $c$ trong khoảng cách $b$ là $F_{c,b} = \sum_{(i,j)\in\Omega_c^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1}\leq d_{ij}<a_b)$. Phân phối khoảng cách ở cấp city là $Y_{D,c,b} = F_{c,b} / \sum_{r=1}^{K}F_{c,r}$ với $\sum_{b=1}^{K}Y_{D,c,b}=1$. Vector $\mathbf{Y}_{D,c}$ được sử dụng để hiệu chỉnh toàn bộ dự báo OD của thành phố mục tiêu. Đây là cấu hình chính của nghiên cứu (`M1_city`).

---

## 3.3. Biến thể quan sát chi tiết ở cấp county (`M1_county`*)*

$$\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c^+ : g(i) = \ell\right\}$$

$$F_{c,\ell,b} = \sum_{(i,j) \in \Omega_{c,\ell}^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1} \le d_{ij} < a_b)$$

$$Y_{D,c,\ell,b} = \frac{F_{c,\ell,b}}{\sum_{r=1}^K F_{c,\ell,r}}, \qquad \sum_{b=1}^K Y_{D,c,\ell,b} = 1$$

$$\widehat{\mathbf{T}}_c^{\mathrm{county}} = \bigcup_{\ell \in \mathcal{G}_c} \left\{ \widehat{T}_{ij}^{\mathrm{CAL}} : (i,j) \in \Omega_{c,\ell}^+ \right\}$$

Một thí nghiệm bổ sung kiểm tra liệu quan sát có độ phân giải không gian chi tiết hơn city có mang lại thêm thông tin hay không. Trong thí nghiệm này, các tract của mỗi city được phân nhóm theo county. Ranh giới county được lấy từ GADM phiên bản 4.1 [@gadm41]. Mỗi tract được gán vào county tương ứng dựa trên vị trí tọa độ tâm trong polygon county. Nếu phép ghép `within` không cho kết quả hợp lệ—chẳng hạn khi tâm tract nằm trên biên polygon hoặc gần đường bờ—mã nguồn chuyển sang polygon gần nhất trong EPSG:5070 và chỉ chấp nhận kết quả khi khoảng cách không quá 5 km; nếu không, chương trình dừng và báo lỗi. Các kết quả trùng được xử lý xác định để mỗi tract chỉ có một nhãn county. GADM chỉ được sử dụng cho bước phân nhóm này; GADM không phải nguồn của tọa độ tract, đặc trưng đô thị hoặc luồng OD. Gọi $g(i)$ là county được gán cho tract $i$. Theo quy tắc được xác nhận từ mã nguồn, các cặp OD được phân nhóm theo county của origin: $\Omega_{c,\ell}^+ = \{(i,j)\in\Omega_c^+:g(i)=\ell\}$. Destination $j$ có thể thuộc cùng county hoặc một county khác. Phân phối khoảng cách của nhóm county $\ell$ được xác định bởi $F_{c,\ell,b} = \sum_{(i,j)\in\Omega_{c,\ell}^+} t_{ij}^{\mathrm{GT}} \mathbb{I}(a_{b-1}\leq d_{ij}<a_b)$ và $Y_{D,c,\ell,b} = F_{c,\ell,b} / \sum_{r=1}^{K}F_{c,\ell,r}$. Do dữ liệu đầu vào vẫn được giới hạn trong các tract thuộc city do Lab cung cấp, $\mathbf{Y}_{D,c,\ell}$ mô tả phân phối khoảng cách của các chuyến đi xuất phát từ những tract của city được gán vào county $\ell$. Đại lượng này không nhất thiết đại diện cho toàn bộ hoạt động di chuyển của county bên ngoài phạm vi dữ liệu thành phố. Mỗi phân phối $\mathbf{Y}_{D,c,\ell}$ được dùng để hiệu chỉnh các cặp có origin thuộc county $\ell$. Sau đó, dự báo của tất cả nhóm county được ghép lại thành một dự báo OD hoàn chỉnh cho city: $\widehat{\mathbf{T}}_{c}^{\mathrm{county}} = \bigcup_{\ell\in\mathcal{G}_c} \{\widehat{T}_{ij}^{\mathrm{CAL}} : (i,j)\in\Omega_{c,\ell}^+\}$, trong đó $\mathcal{G}_c$ là tập county xuất hiện trong dữ liệu của city $c$. Như vậy, việc tăng độ phân giải quan sát từ city lên county không làm thay đổi phạm vi dự báo. Mô hình vẫn tái tạo và đánh giá toàn bộ OD của thành phố trên $\Omega_c^+$; chỉ thông tin tổng hợp được cung cấp cho bước hiệu chỉnh trở nên chi tiết hơn về mặt không gian (`M1_county`).

---

## 3.4. Hiệu chỉnh dự báo zero-shot bằng phân phối khoảng cách

$$\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij} \mid T_{ij} \ge 1]$$

### 3.4.1 Primary Calibration at the City Level (`M1_city`)

$$\widehat{F}_{c,b}^{\mathrm{ZS}} = \sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}} \mathbb{I}(a_{b-1} \le d_{ij} < a_b)$$

$$\widehat{S}_{c}^{\mathrm{ZS}} = \sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}}$$

$$\widehat{Y}_{D,c,b}^{\mathrm{ZS}} = \frac{\widehat{F}_{c,b}^{\mathrm{ZS}}}{\widehat{S}_{c}^{\mathrm{ZS}}}$$

$$\mathcal{A}_c = \left\{ b : \widehat{Y}_{D,c,b}^{\mathrm{ZS}} > 0 \right\}$$

$$p_{c,b}^{\mathrm{cond}} = \frac{Y_{D,c,b} \mathbb{I}(b \in \mathcal{A}_c)}{\sum_{r \in \mathcal{A}_c} Y_{D,c,r}}$$

$$r_{c,b} = \frac{p_{c,b}^{\mathrm{cond}}}{\widehat{Y}_{D,c,b}^{\mathrm{ZS}}}, \qquad w_{c,b}(q) = r_{c,b}^q, \quad q \in [0, 1]$$

$$Z_c(q) = \sum_{r \in \mathcal{A}_c} \widehat{Y}_{D,c,r}^{\mathrm{ZS}} w_{c,r}(q), \qquad s_{c,b}(q) = \frac{w_{c,b}(q)}{Z_c(q)}$$

$$\widehat{T}_{ij}^{M1_{\mathrm{city}}} = s_{c, b(i,j)}(q) \cdot \widehat{T}_{ij}^{\mathrm{ZS}}$$

$$\sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{M1_{\mathrm{city}}} = \sum_{(i,j) \in \Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}}$$

$$\widehat{Y}_{D,c,b}^{M1_{\mathrm{city}}} = p_{c,b}^{\mathrm{cond}}$$

### 3.4.2 Spatial Resolution Variant at the County Level (`M1_county`)

$$\Omega_{c,\ell}^+ = \left\{(i,j) \in \Omega_c^+ : g(i) = \ell\right\}$$

$$w_{c,\ell,b}(q) = \left(\frac{p_{c,\ell,b}^{\mathrm{cond}}}{\widehat{Y}_{D,c,\ell,b}^{\mathrm{ZS}}}\right)^q, \qquad s_{c,\ell,b}(q) = \frac{w_{c,\ell,b}(q)}{\sum_{r \in \mathcal{A}_{c,\ell}} \widehat{Y}_{D,c,\ell,r}^{\mathrm{ZS}} w_{c,\ell,r}(q)}$$

$$\widehat{T}_{ij}^{M1_{\mathrm{county}}} = s_{c, g(i), b(i,j)}(q) \cdot \widehat{T}_{ij}^{\mathrm{ZS}}$$

### 3.4.3 Invariant Mathematical Properties

Mô hình được huấn luyện trên các thành phố nguồn và được đóng băng trước khi đánh giá trên thành phố mục tiêu. Với mỗi cặp $(i,j)\in\Omega_c^+$, mô hình ZTNB tạo ra dự báo cường độ zero-shot $\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij}\mid T_{ij}\geq1]$, tạo thành baseline $M_0$. Tổng luồng dự báo trong khoảng $b$ là $\widehat{F}_{c,b}^{\mathrm{ZS}} = \sum_{(i,j)\in\Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}} \mathbb{I}(a_{b-1}\leq d_{ij}<a_b)$ và tổng cường độ dự báo của thành phố là $\widehat{S}_{c}^{\mathrm{ZS}} = \sum_{(i,j)\in\Omega_c^+} \widehat{T}_{ij}^{\mathrm{ZS}}$. Phân phối khoảng cách ngầm định bởi baseline là $\widehat{Y}_{D,c,b}^{\mathrm{ZS}} = \widehat{F}_{c,b}^{\mathrm{ZS}} / \widehat{S}_{c}^{\mathrm{ZS}}$. Tập khoảng hoạt động là $\mathcal{A}_c = \{b : \widehat{Y}_{D,c,b}^{\mathrm{ZS}} > 0\}$ và phân phối mục tiêu điều kiện hóa là $p_{c,b}^{\mathrm{cond}} = Y_{D,c,b}\mathbb{I}(b\in\mathcal{A}_c) / \sum_{r\in\mathcal{A}_c}Y_{D,c,r}$. Với $b\in\mathcal{A}_c$, tỷ lệ $r_{c,b} = p_{c,b}^{\mathrm{cond}} / \widehat{Y}_{D,c,b}^{\mathrm{ZS}}$ và trọng số $w_{c,b}(q) = r_{c,b}^q$ ($q\in[0,1]$, $q=1.0$ chuẩn). Hệ số chuẩn hóa $s_{c,b}(q) = w_{c,b}(q) / Z_c(q)$ với $Z_c(q) = \sum_{r\in\mathcal{A}_c}\widehat{Y}_{D,c,r}^{\mathrm{ZS}}w_{c,r}(q)$. Dự báo sau hiệu chỉnh là $\widehat{T}_{ij}^{M1_{\mathrm{city}}} = s_{c,b(i,j)}(q)\widehat{T}_{ij}^{\mathrm{ZS}}$. Chuẩn hóa bảo toàn chính xác tổng cường độ dự báo $\sum_{\Omega_c^+}\widehat{T}_{ij}^{M1_{\mathrm{city}}} = \sum_{\Omega_c^+}\widehat{T}_{ij}^{\mathrm{ZS}}$. Khi $q=1$, phân phối sau hiệu chỉnh khớp với $p_{c,b}^{\mathrm{cond}}$ (khớp raw $\mathbf{Y}_{D,c}$ khi mọi bin đều hoạt động). Đối với biến thể $M1_{\mathrm{county}}$, việc hiệu chỉnh áp dụng độc lập cho từng origin county $\Omega_{c,\ell}^+ = \{(i,j)\in\Omega_c^+:g(i)=\ell\}$, bảo toàn tổng lưu lượng xuất phát của từng county. Cả hai cấu hình đều là phép hậu xử lý giải tích, giữ nguyên tập hỗ trợ $\Omega_c^+$ và bảo toàn thứ hạng nội khoảng ($\tau = 1.0$ đối với nhóm không suy biến).

---

## 3.5. Mô hình hóa cường độ OD bằng ZTNB

### 3.5.1 Frozen neural backbone and training configuration

Trong mỗi fold, 26 đặc trưng tract được chuẩn hóa bằng các thống kê chỉ fit trên 35 thành phố huấn luyện, sau đó áp dụng nguyên trạng cho tập validation và test. Đồ thị không gian nối các tâm tract trong bán kính Haversine 5 km, có self-loop và biểu diễn quan hệ láng giềng theo hai chiều. Tract không có láng giềng trong bán kính được nối với tract gần nhất để tránh nút cô lập. Backbone gồm hai lớp GNN, chiều ẩn 64 và dropout 0.1. Pairwise decoder nhận embedding của origin và destination cùng với $\log(1+d_{ij})$ và log gravity prior. Mô hình được huấn luyện tối đa 200 epoch bằng AdamW (learning rate $2\times10^{-3}$, weight decay $10^{-4}$) [@loshchilov2019adamw], gradient clipping 5.0, scheduler `ReduceLROnPlateau` (factor 0.5, patience 4) và early stopping patience 15 theo validation CPC. Sau bước chọn mô hình, toàn bộ tham số backbone và output head được giữ cố định khi hiệu chỉnh trên thành phố mục tiêu.

### 3.5.2 Zero-truncated negative binomial likelihood and inference

$$P(T_{ij} = t_{ij} \mid T_{ij} \ge 1) = \frac{P_{\mathrm{NB}}(T_{ij} = t_{ij}; \mu_{ij}, \phi)}{1 - P_{\mathrm{NB}}(T_{ij} = 0; \mu_{ij}, \phi)}$$

$$p_{0,ij} = \left( \frac{\phi}{\mu_{ij} + \phi} \right)^\phi$$

$$\mathcal{L}_{\mathrm{ZTNB}} = -\frac{1}{|\Omega_c^+|} \sum_{(i,j) \in \Omega_c^+} \left[ \log P_{\mathrm{NB}}(t_{ij}; \mu_{ij}, \phi) - \log(1 - p_{0,ij}) \right]$$

$$\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij} \mid T_{ij} \ge 1] = \frac{\mu_{ij}}{1 - p_{0,ij}}$$

Do tập dữ liệu chỉ bao gồm những cặp OD có luồng dương, cường độ luồng được mô hình hóa bằng phân phối negative binomial cắt tại 0 [@grogger1991truncated; @hilbe2011negative]. Với $t_{ij}\geq1$, likelihood là: $P(T_{ij}=t_{ij}\mid T_{ij}\geq1) = P_{\mathrm{NB}}(T_{ij}=t_{ij};\mu_{ij},\phi) / (1-P_{\mathrm{NB}}(T_{ij}=0;\mu_{ij},\phi))$. Trong đó, mạng nơ-ron dự báo trung bình chưa cắt $\mu_{ij}>0$, còn $\phi>0$ là tham số phân tán. Xác suất bằng 0 của phân phối negative binomial cơ sở là $p_{0,ij} = (\phi/(\mu_{ij}+\phi))^\phi$. Hàm mất mát huấn luyện là negative log-likelihood của phân phối ZTNB: $\mathcal{L}_{\mathrm{ZTNB}} = -\frac{1}{|\Omega_c^+|} \sum_{(i,j)\in\Omega_c^+} [\log P_{\mathrm{NB}}(t_{ij};\mu_{ij},\phi) - \log(1-p_{0,ij})]$. Tại thời điểm suy luận, dự báo zero-shot không sử dụng trực tiếp $\mu_{ij}$. Thay vào đó, mô hình sử dụng kỳ vọng có điều kiện: $\widehat{T}_{ij}^{\mathrm{ZS}} = \mathbb{E}[T_{ij}\mid T_{ij}\geq1] = \frac{\mu_{ij}}{1-p_{0,ij}}$. Do là kỳ vọng của phân phối, $\widehat{T}_{ij}^{\mathrm{ZS}}$ là một giá trị thực dương và không bắt buộc phải là số nguyên. ZTNB chỉ mô hình hóa cường độ của những cặp thuộc $\Omega_c^+$; mô hình không dự báo sự tồn tại của các cặp OD chưa quan sát và không xem chúng là các luồng bằng 0.

Hình 1 tóm tắt framework hiệu chỉnh oracle có điều kiện theo support, đồng thời phân tách rõ quá trình huấn luyện cross-city, suy luận trên thành phố mục tiêu bằng mô hình đóng băng và can thiệp thông tin tổng hợp oracle.

![Hình 1](figures/fig1_oracle_calibration_framework.svg)
**Hình 1 | Framework hiệu chỉnh oracle có điều kiện theo support.** Mô hình cross-city (GNN/ZTNB) được huấn luyện trên các thành phố nguồn và hoàn toàn đóng băng tại thời điểm suy luận. Đối với thành phố mục tiêu $c$, mô hình chỉ dự báo cường độ luồng trên tập hỗ trợ liên vùng dương $\Omega_c^+$. Phân phối khoảng cách mục tiêu $Y_D$ được dùng để tái phân bổ giải tích khối lượng luồng dự báo giữa các khoảng cự ly, bảo toàn tổng lưu lượng dự báo và bảo toàn thứ tự xếp hạng các cặp OD trong từng khoảng.

**Hình 1. Framework hiệu chỉnh oracle có điều kiện theo support.** Mô hình cross-city $M_0$ được huấn luyện trên các thành phố nguồn và đóng băng trước khi suy luận trên thành phố mục tiêu. Đối với một thành phố mục tiêu, $M_0$ trước hết tạo ra dự báo cường độ baseline $\widehat{\mathbf{T}}^{(0)}$ trên tập hỗ trợ dương đã biết $\Omega_c^+$. Phân phối theo nhóm khoảng cách oracle $\mathbf{Y}_{D,c}$ được xác định trực tiếp từ chính các luồng OD ground-truth dương của thành phố mục tiêu đang được sử dụng để đánh giá và chỉ được đưa vào tại thời điểm suy luận. Các hệ số theo bin tái phân bổ khối lượng dự báo giữa các khoảng cự ly để tạo $\widehat{\mathbf{T}}^{(1)}$ mà không cập nhật tham số mô hình hoặc tạo liên kết OD mới. Sơ đồ biểu diễn một can thiệp thông tin oracle, không phải pipeline telemetry bên ngoài được thu thập độc lập.

---

## 3.6. Giao thức đánh giá cross-city và suy luận thống kê

### 3.6.1 Giao thức kiểm định chéo liên thành phố 5-fold
Nghiên cứu áp dụng giao thức kiểm định chéo liên thành phố 5-fold trên 50 vùng đô thị của Hoa Kỳ. Trong mỗi fold, 35 thành phố được dùng để huấn luyện, 5 thành phố dùng để lựa chọn mô hình (validation) và 10 thành phố dùng để đánh giá (testing). Mỗi thành phố xuất hiện trong tập kiểm tra đúng một lần, bao phủ toàn bộ 50 đô thị qua các fold.

Đơn vị phân chia fold là toàn bộ thành phố, không phải các cặp OD, tract hoặc mẫu quan sát trong cùng một thành phố. Do đó, các cặp OD hoặc tract của cùng một thành phố không bị phân tán giữa training, validation và test mà nằm trọn vẹn trong một tập duy nhất của mỗi fold. Việc phân chia ở cấp thành phố này là điều kiện cần để hỗ trợ claim zero-shot liên thành phố.

Các biên khoảng cách được tính riêng cho từng fold và chỉ sử dụng khoảng cách của các cặp OD thuộc tập thành phố huấn luyện. Sau khi huấn luyện hoàn tất, tham số của mô hình được giữ cố định trước khi dự báo trên các thành phố kiểm tra. Đối với mỗi thành phố mục tiêu, ba cấu hình được phân biệt: $M_0$ (dự báo zero-shot không sử dụng $Y_D$), $M1_{\mathrm{city}}$ (hiệu chỉnh bằng một $Y_D$ oracle ở cấp city), và $M1_{\mathrm{county}}$ (hiệu chỉnh bằng nhiều $Y_D$ oracle được phân nhóm theo county). So sánh giữa $M_0$ và $M1_{\mathrm{city}}$ là thí nghiệm chính nhằm trả lời liệu phân phối khoảng cách của thành phố mục tiêu có bổ sung thông tin cho dự báo zero-shot hay không (RQ1). So sánh giữa $M1_{\mathrm{city}}$ và $M1_{\mathrm{county}}$ cung cấp bằng chứng cho khía cạnh độ phân giải không gian của quan sát trong RQ2. Trong tất cả cấu hình, mô hình dự báo và được đánh giá trên cùng tập hỗ trợ dương $\Omega_c^+$ của toàn thành phố.

---

### 3.6.2 Primary Evaluation Metric: Common Part of Commuters (CPC)

$$\operatorname{CPC}_c = \frac{2 \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \min\left(t_{ij}^{\mathrm{GT}}, \widehat{T}_{ij}\right)}{\sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} t_{ij}^{\mathrm{GT}} + \sum_{(i,j) \in \Omega_{c,\mathrm{inter}}^+} \widehat{T}_{ij}}$$

$$\Delta\operatorname{CPC}_c = \operatorname{CPC}_c(M1_{\mathrm{city}}) - \operatorname{CPC}_c(M_0)$$

Chỉ số chính là Common Part of Commuters (CPC), được tính trên các cặp OD liên vùng thuộc tập hỗ trợ dương: $\operatorname{CPC}_c = 2\sum_{(i,j)\in\Omega_{c,\mathrm{inter}}^+} \min(t_{ij}, \widehat{T}_{ij}) / (\sum_{(i,j)} t_{ij} + \sum_{(i,j)} \widehat{T}_{ij})$, trong đó tập hỗ trợ đánh giá liên vùng dương đã biết được định nghĩa chính thức là:
$$
\Omega_{c,\mathrm{inter}}^+ = \left\{ (i,j): t_{ij}\geq1,\ i\neq j,\ d_{ij}>0 \right\}.
$$
CPC nằm trong khoảng từ 0 đến 1; giá trị lớn hơn biểu thị mức độ trùng khớp cao hơn giữa luồng dự báo và luồng tham chiếu [@lenormand2016comparison]. Hiệu quả bổ sung của $Y_D$ tại thành phố $c$ được xác định bằng chênh lệch ghép cặp: $\Delta\operatorname{CPC}_c = \operatorname{CPC}_c(M1_{\mathrm{city}}) - \operatorname{CPC}_c(M_0)$. Giá trị dương cho thấy việc sử dụng $Y_D$ cải thiện kết quả so với dự báo zero-shot trên cùng thành phố, cùng tập hỗ trợ và cùng mô hình nền.

---

### 3.6.3 Aggregation Across Model Seeds and Cities

$$\mathcal{S} = \{1, 10, 100\}$$

$$\overline{\Delta\operatorname{CPC}}_c = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \left[ \operatorname{CPC}_{c,s}(M1_{\mathrm{city}}) - \operatorname{CPC}_{c,s}(M_0) \right]$$

$$\overline{\Delta\operatorname{CPC}} = \frac{1}{50} \sum_{c=1}^{50} \overline{\Delta\operatorname{CPC}}_c$$

Mỗi cấu hình được chạy với ba model seeds: $\mathcal{S}=\{1,10,100\}$. Đối với mỗi thành phố, chênh lệch CPC trước hết được tính riêng cho từng seed và sau đó lấy trung bình: $\overline{\Delta\operatorname{CPC}}_c = \frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}} [\operatorname{CPC}_{c,s}(M1_{\mathrm{city}}) - \operatorname{CPC}_{c,s}(M_0)]$. Hiệu quả tổng thể được tính bằng macro-average trên 50 thành phố: $\overline{\Delta\operatorname{CPC}} = \frac{1}{50}\sum_{c=1}^{50}\overline{\Delta\operatorname{CPC}}_c$. Cách tổng hợp này trao trọng số như nhau cho mỗi thành phố, bất kể số tract, số cặp OD hoặc tổng số chuyến đi của thành phố đó. Vì vậy, estimand chính là mức cải thiện trung bình giữa các thành phố, không phải mức cải thiện trung bình giữa tất cả cặp OD gộp chung.

---

### 3.6.4 Uncertainty Quantification and Statistical Hypothesis Testing

$$\left\{ \overline{\Delta\operatorname{CPC}}_c \right\}_{c=1}^{50}$$

Khoảng tin cậy 95% của mức cải thiện trung bình được ước lượng bằng bootstrap phân tầng theo fold ở cấp city ($B=10,000$) [@efron1993bootstrap]. Trong mỗi lần bootstrap, các thành phố được lấy mẫu có hoàn lại trong từng fold từ tập các giá trị $\overline{\Delta\operatorname{CPC}}_c$, sau đó tính lại macro-average. Việc lấy mẫu ở cấp city giữ city là đơn vị suy luận thống kê và tránh xem hàng triệu cặp OD trong cùng thành phố như các quan sát độc lập. Kiểm định Wilcoxon signed-rank ghép cặp [@wilcoxon1945ranking] được áp dụng trên 50 giá trị $\{\overline{\Delta\operatorname{CPC}}_c\}_{c=1}^{50}$. Giả thuyết không cho rằng phân phối chênh lệch giữa $M1_{\mathrm{city}}$ và $M_0$ có trung vị bằng 0. Kiểm định này bổ sung cho khoảng tin cậy bootstrap bằng cách đánh giá liệu hướng cải thiện quan sát được có phù hợp với biến động ngẫu nhiên quanh 0 hay không.

---

### 3.6.5 Robustness and Diagnostic Stress Tests

Các phân tích bổ sung được thiết kế để kiểm tra phạm vi và cơ chế của kết quả chính: (1) Độ phân giải khoảng cách: thay đổi $K \in \{2,4,6,8,10,12,14,16,18,20\}$ và so sánh chín cấu hình phụ với mốc khóa $K=8$ bằng hiệu chỉnh Holm step-down [@holm1979sequential]; (2) Độ phân giải không gian: so sánh $M1_{\mathrm{city}}$ với $M1_{\mathrm{county}}$; (3) Chất lượng quan sát: thêm nhiễu Total Variation có kiểm soát vào $Y_D$; (4) Thứ tự khoảng cách: hoán vị các khoảng của $Y_D$; (5) Tính đặc thù theo thành phố: sử dụng phân phối của thành phố khác trong matched-placebo; (6) Độ bền theo khởi tạo: lặp lại với các model seeds 1, 10, 100; và (7) Độ bền theo kiến trúc: đánh giá Urban GNN và Node MLP cùng với gravity baseline cổ điển. Các phân tích này không thay đổi estimand chính; chúng xác định ranh giới vận hành và cơ chế khoa học của phương pháp.

---

### 3.6.6. Phân tích độ phân giải không gian theo county

$$\mathbf{Y}_{D,c,\ell} = \mathbf{Y}_{D,c},$$

$$M1_{\mathrm{county}} \equiv M1_{\mathrm{city}}, \qquad \Delta\operatorname{CPC}_{\mathrm{res},c} = 0$$

$$\Delta\operatorname{CPC}_{\mathrm{res},c} = \operatorname{CPC}_c(M1_{\mathrm{county}}) - \operatorname{CPC}_c(M1_{\mathrm{city}})$$

Trong 50 bộ dữ liệu đô thị, 39 thành phố chỉ chứa các tract được gán vào một county, trong khi 11 thành phố chứa tract thuộc từ hai đến bảy counties. Nhóm multi-county gồm Kansas City, New York, Dallas, Denver, Omaha, Tulsa, Detroit, Chicago, Boston, Milwaukee và Atlanta. Đối với 39 single-county cities, tất cả origin tract thuộc cùng một nhóm county. Vì vậy, phân phối quan sát theo county và phân phối quan sát theo city là tương đương: $\mathbf{Y}_{D,c,\ell} = \mathbf{Y}_{D,c}$, dẫn đến $M1_{\mathrm{county}} \equiv M1_{\mathrm{city}}$ và $\Delta\operatorname{CPC}_{\mathrm{res},c}=0$. Trong đó, hiệu quả của việc tăng độ phân giải được định nghĩa là $\Delta\operatorname{CPC}_{\mathrm{res},c} = \operatorname{CPC}_c(M1_{\mathrm{county}}) - \operatorname{CPC}_c(M1_{\mathrm{city}})$. Do đó, 39 single-county cities đóng vai trò như một kiểm tra bất biến của thuật toán: việc chia một city thành đúng một nhóm không được làm thay đổi kết quả. Thông tin thực nghiệm về lợi ích của độ phân giải county đến từ 11 multi-county cities. Đối với các thành phố này, mỗi phân phối $\mathbf{Y}_{D,c,\ell}$ được xây dựng từ những chuyến đi có origin thuộc county $\ell$, còn dự báo cuối cùng vẫn được ghép và đánh giá trên toàn bộ positive support của city. Kết quả được báo cáo theo hai phạm vi: (1) kết quả pooled trên toàn bộ 50 thành phố, phản ánh hiệu quả trung bình của việc cung cấp quan sát county-level trên toàn benchmark; và (2) kết quả riêng trên 11 multi-county cities, phản ánh hiệu quả tại những thành phố mà county-level thực sự cung cấp độ phân giải bổ sung. Do 39 thành phố tạo ra chênh lệch bằng 0 theo cấu trúc, diễn giải về giá trị của county-level observation chủ yếu dựa trên nhóm 11 multi-county cities.

---

# Section 4: Empirical Results

---

## 4.1 Does $Y_D$ improve zero-shot OD reconstruction?

Trong thí nghiệm chính, việc bổ sung phân phối di chuyển theo nhóm khoảng cách oracle của thành phố mục tiêu làm CPC liên vùng trung bình trên 50 thành phố Hoa Kỳ tăng từ 0.71281 ở mô hình zero-shot cơ sở ($M_0$) lên 0.71635 sau hiệu chỉnh ($M_1$). Mức cải thiện trung bình đạt $\Delta\mathrm{CPC}=+0.00354$, với khoảng tin cậy 95% từ fold-stratified hierarchical bootstrap là $[+0.0026,+0.0045]$. Toàn bộ khoảng tin cậy nằm phía trên 0, cho thấy mức cải thiện CPC trung bình được ước lượng là dương dưới giao thức bootstrap đã sử dụng.

Theo Hình 2, mức cải thiện không chỉ tập trung ở một số ít thành phố mà xuất hiện trên phần lớn các thành phố được đánh giá. Cụ thể, CPC tăng sau hiệu chỉnh ở 45 trong 50 thành phố (90.0%). Trung vị $\Delta\mathrm{CPC}=+0.00195$ cũng nằm phía dương, mặc dù mức cải thiện khác nhau đáng kể giữa các thành phố. Năm thành phố còn lại có CPC giảm sau hiệu chỉnh, cho thấy lợi ích của thông tin khoảng cách không xuất hiện ở mọi trường hợp. Nhìn chung, phân bố theo thành phố cho thấy mức cải thiện có quy mô nhỏ nhưng khá nhất quán trên tập đánh giá.

Để kiểm tra thêm liệu xu hướng cải thiện này có mang tính hệ thống hay không, chúng tôi sử dụng kiểm định Wilcoxon signed-rank hai phía trên các cặp kết quả $M_0$ và $M_1$ của 50 thành phố. Kiểm định cho $p=1.93\times10^{-9}$, cung cấp bằng chứng mạnh chống lại giả thuyết không có sự thay đổi có hệ thống giữa hai điều kiện. Kết hợp các kết quả trên, phân phối di chuyển theo nhóm khoảng cách oracle của thành phố mục tiêu mang lại một mức cải thiện nhỏ nhưng nhất quán trên phần lớn các thành phố được đánh giá so với mô hình zero-shot cơ sở.

---

![Hình 2](figures/fig2_main_per_city.png)
**Hình 2 | Mức cải thiện CPC liên vùng theo từng thành phố từ hiệu chỉnh khoảng cách mục tiêu oracle.** Biểu đồ cột thể hiện mức thay đổi hiệu năng theo từng thành phố $\Delta\text{CPC}_c = \text{CPC}(M_{1,c}) - \text{CPC}(M_{0,c})$ trên $N=50$ thành phố kiểm tra, xếp từ thấp đến cao. Đường nét đứt màu xanh lá thể hiện mức cải thiện trung bình ($+0.00354$) và đường chấm màu cam thể hiện trung vị ($+0.00195$). Tổng cộng có 45/50 thành phố (90.0%) đạt mức tăng dương, với khoảng tin cậy 95% phân tầng theo fold là $[+0.0026, +0.0045]$.

---

### Bảng 1: Benchmark tái tạo luồng zero-shot chính ($N=50$ thành phố, $K=8$ khoảng khoảng cách)

| Điều kiện mô hình | CPC liên vùng TB | Trung vị CPC | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% (Phân tầng) | Tỷ lệ thành phố thắng | Wilcoxon $p$ (Hai phía) |
|---|---|---|---|---|---|---|
| **Mô hình cơ sở Zero-Shot ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — |
| **Mô hình sau hiệu chỉnh ($M_1$)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | **$[+0.0026, +0.0045]$** | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ |

*Ghi chú: Được đánh giá trên tập hỗ trợ liên vùng dương quan sát được $\Omega_c^+$. Khoảng tin cậy được tính qua bootstrap phân tầng $B=10,000$ lần trên các thành phố. Giá trị trung bình qua 3 model seeds độc lập.*

---

## 4.2 Is the gain genuinely target-specific and structurally meaningful?

Mặc dù kết quả ở Mục 4.1 cho thấy việc hiệu chỉnh bằng phân phối di chuyển theo nhóm khoảng cách $Y_D$ của thành phố mục tiêu giúp cải thiện CPC, kết quả đó vẫn chưa cho biết liệu mức cải thiện có thực sự đến từ thông tin khoảng cách đặc thù của thành phố mục tiêu hay chỉ đơn giản là hệ quả của quá trình hiệu chỉnh. Để kiểm tra điều này, chúng tôi so sánh trường hợp sử dụng đúng $Y_D$ của thành phố mục tiêu với trường hợp sử dụng phân phối của các thành phố khác. Để bảo đảm so sánh công bằng, các phân phối từ thành phố khác được điều chỉnh sao cho tạo ra cùng mức độ can thiệp $D_T$ như trường hợp sử dụng thông tin của thành phố mục tiêu. Khi sử dụng đúng $Y_D$ của thành phố mục tiêu, mức cải thiện CPC trung bình đạt $\Delta\mathrm{CPC}=+0.003539$. Ngược lại, khi sử dụng các phân phối từ thành phố khác nhưng đã được khớp cùng mức độ can thiệp, mức thay đổi CPC trung bình chỉ là $\Delta\mathrm{CPC}=-0.000091$, tức gần như không mang lại cải thiện. Chênh lệch về mức cải thiện giữa hai điều kiện đạt $+0.003630$, với khoảng tin cậy 95% là $[+0.00287,+0.00445]$. Kiểm định Wilcoxon signed-rank một phía khi so sánh trường hợp sử dụng đúng thông tin của thành phố mục tiêu với trường hợp sử dụng thông tin từ thành phố khác cho $p=2.19\times10^{-11}$. Kết quả này cho thấy rằng khi mức độ hiệu chỉnh được kiểm soát ở cùng một mức, việc sử dụng phân phối khoảng cách của các thành phố khác không tái tạo được mức cải thiện đạt được khi sử dụng phân phối của chính thành phố mục tiêu. Nói cách khác, lợi ích của quá trình hiệu chỉnh không chỉ đến từ việc thay đổi dự báo mà còn phụ thuộc vào việc thông tin khoảng cách được sử dụng có phù hợp với thành phố mục tiêu hay không.

Một khả năng khác là không cần biết chính xác phân phối di chuyển theo khoảng cách của từng thành phố mục tiêu; thay vào đó, một phân phối trung bình được xây dựng từ các thành phố trong tập huấn luyện có thể đã đủ để mang lại mức cải thiện tương tự. Nếu điều này xảy ra, lợi ích quan sát được có thể chủ yếu đến từ một quy luật suy giảm theo khoảng cách mang tính tổng quát, thay vì từ thông tin đặc thù của từng thành phố. Tuy nhiên, khi sử dụng phân phối trung bình của các thành phố huấn luyện với cùng mức độ hiệu chỉnh, mức cải thiện trung bình chỉ đạt $\Delta\mathrm{CPC}=+0.000914$, thấp hơn so với $+0.003539$ khi sử dụng $Y_D$ của chính thành phố mục tiêu. Chênh lệch giữa hai điều kiện là $+0.002626$, với khoảng tin cậy 95% $[+0.00197,+0.00336]$ và kiểm định Wilcoxon một phía cho $p=4.03\times10^{-11}$. Kết quả này cho thấy một quy luật suy giảm theo khoảng cách tổng quát có thể tạo ra một mức cải thiện nhỏ, nhưng không tái tạo được mức cải thiện đạt được khi sử dụng phân phối khoảng cách đặc thù của thành phố mục tiêu. Điều này hỗ trợ vai trò của thông tin đặc thù theo thành phố trong $Y_D$ đối với mức cải thiện quan sát được.

Bên cạnh các kiểm tra sử dụng phân phối thay thế từ những nguồn khác, chúng tôi còn thực hiện một phép kiểm tra bằng cách hoán đổi vị trí các khoảng trong chính $Y_D$ của thành phố mục tiêu. Phép hoán đổi này giữ nguyên các tỷ lệ ban đầu của phân phối nhưng phá vỡ mối quan hệ giữa mỗi tỷ lệ di chuyển và khoảng cách tương ứng, qua đó kiểm tra liệu cấu trúc theo khoảng cách của $Y_D$ có quan trọng đối với mức cải thiện hay không. Trong điều kiện này, CPC giảm trung bình với $\Delta\mathrm{CPC}=-0.006964$, trái ngược với mức cải thiện $\Delta\mathrm{CPC}=+0.003539$ khi sử dụng đúng $Y_D$. Kết quả này cung cấp thêm bằng chứng rằng giá trị của $Y_D$ không chỉ nằm ở các tỷ lệ di chuyển được quan sát mà còn ở việc các tỷ lệ đó được gắn đúng với các khoảng cách tương ứng. Kết hợp với các kiểm tra sử dụng phân phối sai thành phố và phân phối trung bình từ tập huấn luyện, kết quả này củng cố bằng chứng rằng mức cải thiện gắn với thông tin khoảng cách có cấu trúc và đặc thù của thành phố mục tiêu.

---

![Hình 5](figures/fig5_structural_validity_placebo.png)
**Hình 5 | Các đối chứng placebo khớp liều lượng công bằng.** So sánh mức tăng tái tạo trung bình $\Delta\mathrm{CPC}$ trên $N=50$ thành phố kiểm tra dưới 3 điều kiện: (1) Phân phối mục tiêu thực sự ($Y_D$, $+0.00357$, $p < 10^{-8}$); (2) Đối chứng donor từ thành phố khác đã khớp liều lượng ($-0.00009$, không có ý nghĩa); và (3) Hoán vị các khoảng khoảng cách ($-0.00669$, $p < 10^{-14}$). Thanh sai số biểu diễn khoảng tin cậy 95% bootstrap phân tầng.

---

### Bảng 2: Tính đặc thù mục tiêu và các đối chứng Placebo ($N=50$ thành phố; $B_{\text{draw}}=1000$, $B_{\text{boot}}=10,000$)

| Experimental Condition | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% (Phân tầng) | Benefit vs $M_0$ ($p_{\text{2-sided}}$) | Specificity Gain vs Placebo | Specificity 95% CI | Target vs Placebo ($p_{\text{1-sided}}$) | Tỷ lệ thắng ($Target > Placebo$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Oracle Target $Y_D$ (Upper Bound)** | **$+0.003539$** | $[+0.00260, +0.00450]$ | $1.93 \times 10^{-9}$ | — | — | — | **45 / 50 (vs M0)** |
| **2. Dose-Matched Training Donors ($B_{\text{draw}}=1000$)** | **$-0.000091$** | $[-0.00089, +0.00071]$ | $0.4097$ (n.s.) | **$+0.003630$** | $[+0.00287, +0.00445]$ | $\mathbf{2.19 \times 10^{-11}}$ | **46 / 50 (92.0%)** |
| **3. Dose-Matched Fold Train-Mean $Y_D$** | **$+0.000914$** | $[+0.00001, +0.00186]$ | $0.4319$ (n.s.) | **$+0.002626$** | $[+0.00197, +0.00336]$ | $\mathbf{4.03 \times 10^{-11}}$ | **47 / 50 (94.0%)** |
| **4. Raw Test Donors (In-Fold 9-Donor Average, E1)** | **$-0.037721$** | $[-0.04357, -0.03268]$ | $1.78 \times 10^{-15}$ | **$+0.041261$** | $[+0.03641, +0.04688]$ | $8.88 \times 10^{-16}$ | **50 / 50 (100%)** |
| **5. Raw Test Donors ($B_{\text{draw}}=1000$ Draws)** | **$-0.037787$** | $[-0.04358, -0.03278]$ | $1.78 \times 10^{-15}$ | **$+0.041326$** | $[+0.03646, +0.04688]$ | $8.88 \times 10^{-16}$ | **50 / 50 (100%)** |
| **6. Raw Training Donors ($B_{\text{draw}}=1000$ Draws)** | **$-0.035148$** | $[-0.04014, -0.03067]$ | $1.78 \times 10^{-15}$ | **$+0.038687$** | $[+0.03431, +0.04349]$ | $8.88 \times 10^{-16}$ | **50 / 50 (100%)** |
| **7. Raw Fold Train-Mean $Y_D$** | **$-0.017735$** | $[-0.02365, -0.01243]$ | $4.91 \times 10^{-12}$ | **$+0.021275$** | $[+0.01613, +0.02706]$ | $4.44 \times 10^{-15}$ | **48 / 50 (96.0%)** |
| **8. Permuted Target $Y_D$ ($B_{\text{draw}}=1000$ Permutations)** | **$-0.006964$** | $[-0.00914, -0.00512]$ | $1.78 \times 10^{-15}$ | **$+0.010504$** | $[+0.00843, +0.01279]$ | $1.78 \times 10^{-15}$ | **49 / 50 (98.0%)** |

*Note: Evaluated across $N=50$ test cities $\times$ 3 model seeds. $B_{\text{draw}}=1000$ indicates the number of stochastic donor / permutation draws per city; $B_{\text{boot}}=10,000$ denotes fold-stratified bootstrap resamples for 95% CIs. Dose matching scales the L2 log-ratio perturbation norm of donor vectors to match the target city's intervention dose $D_T$. The primary placebo result reported here is the unified training-donor arm (Row 2, $p=2.19\times 10^{-11}$, $46/50$); the fair weight-matched permutation summary ($+0.00367$, $47/50$, $p=6.74\times 10^{-12}$) is reported as a separate robustness analysis arm and is not pooled with Bảng 2. For dose-matched train-mean (Row 3), the non-parametric Wilcoxon test reflects symmetric positive/negative city ranks ($p=0.4319$, n.s.) despite a slightly positive bootstrap mean CI.*

---

## 4.3 How does the value of $Y_D$ depend on observation resolution and quality?

Mức độ đóng góp của phân phối di chuyển theo nhóm khoảng cách tại thành phố mục tiêu có thể phụ thuộc vào lượng thông tin tổng hợp mà quan sát này còn giữ lại được. Vì vậy, chúng tôi xem xét hai khía cạnh của độ phân giải quan sát (độ phân giải theo khoảng cách $K$ và độ phân giải theo không gian) cũng như độ trung thực của quan sát dưới các mức nhiễu tổng hợp. Các thí nghiệm này nhằm kiểm tra xem việc giữ lại nhiều cấu trúc chi tiết và chính xác hơn có cung cấp thêm các ràng buộc hữu ích cho quá trình tái tạo OD hay không.

---

### 4.3.1 Higher distance resolution provides more informative constraints

Trên các giá trị $K$ đã kiểm tra, mức cải thiện trong tái tạo OD tăng khi số lượng nhóm khoảng cách tăng. Ngay tại độ phân giải thấp nhất ($K=2$), việc hiệu chỉnh bằng $Y_D$ đã cải thiện CPC trung bình $+0.00098$ so với mô hình zero-shot cố định, với khoảng tin cậy bootstrap 95% là $[+0.00052,+0.00151]$, đồng thời cải thiện kết quả ở 39/50 thành phố. Mức cải thiện đạt $+0.00354$ CPC tại cấu hình tham chiếu ($K=8$) và $+0.00639$ CPC tại $K=20$. Ở độ phân giải cao nhất được kiểm tra, 46/50 thành phố có kết quả tốt hơn zero-shot baseline và khoảng tin cậy bootstrap 95% vẫn nằm hoàn toàn trên 0, $[+0.00508,+0.00769]$.

### Bảng 3: Độ mở rộng của độ phân giải thông tin qua các khoảng khoảng cách ($K \in \{2, 4, \dots, 20\}$)

| Độ phân giải ($K$) | CPC liên vùng TB | Trung vị CPC | $\Delta\text{CPC}$ trung bình | Trung vị $\Delta\text{CPC}$ | Khoảng tin cậy 95% (Phân tầng) | Tỷ lệ thành phố thắng | Average Gain / Bin ($\Delta\text{CPC}/K$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline ($M_0$)** | $0.71281 \pm 0.04434$ | $0.71632$ | — | — | — | — | — |
| **$K = 2$** | $0.71379 \pm 0.04441$ | $0.71665$ | **$+0.00098$** | $+0.00034$ | $[+0.00052, +0.00151]$ | **39 / 50 (78.0%)** | $0.000488$ |
| **$K = 4$** | $0.71479 \pm 0.04439$ | $0.71720$ | **$+0.00198$** | $+0.00088$ | $[+0.00125, +0.00279]$ | **39 / 50 (78.0%)** | $0.000494$ |
| **$K = 6$** | $0.71570 \pm 0.04445$ | $0.71784$ | **$+0.00289$** | $+0.00152$ | $[+0.00201, +0.00384]$ | **44 / 50 (88.0%)** | $0.000481$ |
| **$K = 8$ (Anchor)** | $0.71635 \pm 0.04454$ | $0.71988$ | **$+0.00354$** | $+0.00195$ | $[+0.00262, +0.00447]$ | **45 / 50 (90.0%)** | $0.000442$ |
| **$K = 10$** | $0.71694 \pm 0.04450$ | $0.72007$ | **$+0.00413$** | $+0.00235$ | $[+0.00311, +0.00514]$ | **45 / 50 (90.0%)** | $0.000413$ |
| **$K = 12$** | $0.71761 \pm 0.04453$ | $0.72060$ | **$+0.00480$** | $+0.00288$ | $[+0.00372, +0.00590]$ | **46 / 50 (92.0%)** | $0.000400$ |
| **$K = 14$** | $0.71819 \pm 0.04456$ | $0.72145$ | **$+0.00538$** | $+0.00373$ | $[+0.00424, +0.00654]$ | **45 / 50 (90.0%)** | $0.000384$ |
| **$K = 16$** | $0.71855 \pm 0.04458$ | $0.72205$ | **$+0.00574$** | $+0.00433$ | $[+0.00455, +0.00694]$ | **46 / 50 (92.0%)** | $0.000359$ |
| **$K = 18$** | $0.71884 \pm 0.04460$ | $0.72230$ | **$+0.00603$** | $+0.00458$ | $[+0.00480, +0.00726]$ | **47 / 50 (94.0%)** | $0.000335$ |
| **$K = 20$** | $0.71920 \pm 0.04462$ | $0.72266$ | **$+0.00639$** | $+0.00494$ | $[+0.00508, +0.00769]$ | **46 / 50 (92.0%)** | $0.000319$ |

*Note: Evaluated across $N=50$ test cities $\times$ 3 model seeds on $\Omega_c^+$. Bins are defined by pair-weighted distance quantiles from 35 training cities per fold. Bootstrap confidence intervals computed via $B=10,000$ fold-stratified resamples.*

---

### 4.3.2. Hiệu chỉnh cấp county tạo ra mức tăng bổ sung pooled nhỏ

Trên toàn bộ 50 thành phố, hiệu chỉnh cấp county tạo ra mức tăng bổ sung pooled nhỏ so với hiệu chỉnh cấp city ($\Delta\mathrm{CPC}_{\mathrm{res}}=+0.00014$, khoảng tin cậy 95% $[+0.00002,+0.00028]$, Wilcoxon $p=0.0064$). Kết quả pooled này cần được diễn giải theo cấu trúc của benchmark. Với 39 thành phố single-county, $M1_{\mathrm{county}}\equiv M1_{\mathrm{city}}$ theo cấu trúc, do đó $\Delta\mathrm{CPC}_{\mathrm{res},c}=0$ chính xác. Vì vậy, phép so sánh thực nghiệm về quan sát không gian chi tiết hơn tập trung vào 11 thành phố multi-county.

Trên nhóm các thành phố multi-county đã đánh giá, hiệu chỉnh cấp county tạo ra mức tăng bổ sung trung bình nhỏ và dương (mean $\Delta\mathrm{CPC}_{\mathrm{res}}=+0.00063$), với 9/11 thành phố cải thiện. Kết quả subgroup này mang tính mô tả nếu chưa có một ước lượng bất định riêng đã được xác minh. Mẫu hình quan sát được phù hợp với khả năng rằng các phân phối theo nhóm origin chi tiết hơn có thể bổ sung thông tin trong một số bộ dữ liệu đô thị multi-county, nhưng nghiên cứu không đo lường hoặc kiểm định trực tiếp tính không đồng nhất nội đô như một cơ chế.

---

![Hình 3](figures/fig3_resolution_sensitivity.png)
**Hình 3 | Phân tích độ nhạy của độ phân giải thông tin ($K$).** (Trái) Mức tăng CPC liên vùng trung bình $\Delta\text{CPC}$ tăng đơn điệu từ $K=2$ ($+0.00098$) lên $K=20$ ($+0.00639$). Dải bóng mờ biểu diễn khoảng tin cậy 95% bootstrap phân tầng. (Phải) Lợi ích biên trên mỗi khoảng bổ sung $\Delta\text{CPC} / K$ đạt đỉnh tại $K=4$ ($0.000494$) và giảm dần xuống $0.000319$ tại $K=20$, cho thấy quy luật hiệu suất giảm dần khi phân chia khoảng cách ngày càng mịn.

---

### 4.3.3 Synthetic observation noise reduces the value of $Y_D$

Sau khi đánh giá ảnh hưởng của độ phân giải quan sát, chúng tôi tiếp tục kiểm tra mức độ phụ thuộc của hiệu quả hiệu chỉnh vào chất lượng của $Y_D$. Cụ thể, phân phối di chuyển theo khoảng cách của thành phố mục tiêu được gây nhiễu ở nhiều mức khác nhau ($\epsilon \in [0.00, 0.05]$ sai số Total Variation), trong khi giữ nguyên mô hình zero-shot, tập thành phố đánh giá và toàn bộ quy trình hiệu chỉnh. Thiết kế này cho phép cô lập ảnh hưởng của sai lệch trong $Y_D$ khỏi các nguồn biến thiên khác của mô hình.

---

![Hình 4](figures/fig4_noise_dose_response.png)
**Hình 4 | Đường đáp ứng liều lượng nhiễu Total Variation (TV).** Hiệu năng sau hiệu chỉnh ($M_1$) suy giảm đơn điệu theo mức nhiễu TV tăng dần từ $\epsilon=0.00$ đến $\epsilon=0.05$. Đường ngang đứt nét màu đỏ thể hiện ngưỡng baseline zero-shot ($M_0 = 0.71281$). Điểm giao cắt thực nghiệm nằm tại mức sai số TV $\approx 4.44\%$, chỉ ra rằng $Y_D$ vẫn mang lại giá trị gia tăng chừng nào sai số ước lượng phân phối tổng hợp còn dưới ngưỡng này.

---

Kết quả trên Hình 4 cho thấy mức tăng suy giảm đơn điệu qua các mức nhiễu tổng hợp đã kiểm tra. Quan sát không nhiễu tạo ra mức tăng lớn nhất ($+0.00354$); mức tăng giảm còn $+0.00070$ tại sai số TV $4\%$ và trở thành âm tại $5\%$ ($-0.00087$). Trên 1.000 hướng nhiễu tổng hợp, điểm giao cắt trung bình được ước lượng tại $\epsilon_{\mathrm{cross}}=4.44\%$ (khoảng tin cậy 95% $[4.16\%,4.77\%]$). Đây là quan hệ dose-response riêng cho benchmark và thiết kế perturbation này, không phải ngưỡng dung sai phổ quát cho quan sát thực tế.

Trong thiết kế perturbation này, mức tăng trung bình vẫn dương tại các mức nhiễu thấp đã kiểm tra, chẳng hạn $+0.00336$ ở TV $1\%$ và $+0.00282$ ở TV $2\%$. Sự suy giảm ở các mức nhiễu cao hơn cũng cho thấy không thể xem $Y_D$ là có lợi bất kể chất lượng quan sát.

---

### Bảng 4: Khả năng chịu đựng nhiễu và độ nhạy trước các mức sai số Total Variation

| Mức nhiễu TV ($\epsilon$) | Mean Calibrated CPC | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% (Phân tầng) | Số thành phố tăng dương | Degradation vs Clean (Holm-adjusted $p$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\epsilon = 0.00$ (Clean Target $Y_D$)** | $0.71635$ | **$+0.00354$** | $[+0.00261, +0.00451]$ | **45 / 50 (90.0%)** | — |
| **$\epsilon = 0.01$ (1% TV Error)** | $0.71617$ | **$+0.00336$** | $[+0.00243, +0.00432]$ | **44 / 50 (88.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.02$ (2% TV Error)** | $0.71563$ | **$+0.00282$** | $[+0.00189, +0.00379]$ | **36 / 50 (72.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.03$ (3% TV Error)** | $0.71474$ | **$+0.00193$** | $[+0.00100, +0.00290]$ | **28 / 50 (56.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.04$ (4% TV Error)** | $0.71351$ | **$+0.00070$** | $[-0.00025, +0.00167]$ | **18 / 50 (36.0%)** | $4.44 \times 10^{-15}$ |
| **$\epsilon = 0.05$ (5% TV Error)** | $0.71193$ | **$-0.00087$** | $[-0.00183, +0.00012]$ | 17 / 50 (34.0%) | $4.44 \times 10^{-15}$ |

*Note: Evaluated across $N=50$ test cities $\times$ 3 model seeds at $K=8$. Synthetic perturbations use centered Gaussian directions in log-ratio space ($z \sim \mathcal{N}(0, I)$, zero-mean centered) and are scaled numerically via exponential tilting ($p_\sigma \propto p \exp(\sigma z)$) to achieve the specified Total Variation error magnitudes $\epsilon = \frac{1}{2}\sum_k |Y_k - \tilde{Y}_k|$. Degradation $p$-values are family-wise error rate controlled across noise levels via Holm-Bonferroni adjustment. The mean signal breakdown crossover threshold across $B=1,000$ noise directions is $\epsilon_{\text{cross}} = 4.44\%$ [95% CI: 4.16%, 4.77%].*

---

## 4.4 Is the finding robust to training and modeling choices?

Các kết quả trước cho thấy $Y_D$ cung cấp thông tin bổ sung hữu ích cho dự báo zero-shot, đồng thời mức độ hữu ích này phụ thuộc vào độ phân giải, chất lượng quan sát và tính đặc thù mục tiêu. Tuy nhiên, cần kiểm tra liệu mức cải thiện quan sát được có ổn định trước biến thiên của quá trình huấn luyện và lựa chọn mô hình hay không. Vì vậy, chúng tôi đánh giá nhiều model seeds và các backbone dự báo khác nhau. Một phép so sánh riêng theo protocol kiểm tra hiệu năng thu được từ quan sát trực tiếp các cặp OD.

---

### 4.4.1 Stability across independent model initializations

Các mô hình học sâu có thể tạo ra kết quả khác nhau giữa các lần huấn luyện do sự ngẫu nhiên trong khởi tạo tham số và quá trình tối ưu. Nếu lợi ích của $Y_D$ chỉ xuất hiện ở một model seed cụ thể, hiệu ứng quan sát được có thể phản ánh biến thiên ngẫu nhiên của quá trình huấn luyện thay vì một đóng góp ổn định từ quan sát mục tiêu.

Để kiểm tra khả năng này, chúng tôi đánh giá cùng một protocol trên ba model seeds độc lập (Seed 1, 10 và 100). Với mỗi thành phố và mỗi seed, zero-shot baseline $M_0$ được so sánh trực tiếp với phiên bản được hiệu chỉnh bằng $Y_D$, sau đó mức thay đổi CPC được tổng hợp qua các seed. Thiết kế ghép cặp này cho phép đánh giá trực tiếp ảnh hưởng của $Y_D$ trong cùng một trạng thái baseline, thay vì để sự khác biệt về chất lượng tuyệt đối giữa các lần huấn luyện chi phối kết quả.

Kết quả cho thấy hướng cải thiện do $Y_D$ mang lại được duy trì qua các model seeds, mặc dù CPC tuyệt đối của từng mô hình có thể thay đổi nhẹ giữa các lần huấn luyện. Điều này cho thấy hiệu ứng của $Y_D$ không phụ thuộc vào một nghiệm tối ưu ngẫu nhiên cụ thể, mà xuất hiện lặp lại khi cùng loại thông tin của thành phố mục tiêu được sử dụng để hiệu chỉnh dự báo zero-shot.

---

### Bảng 5: Độ bền vững theo khởi tạo mô hình qua các seed độc lập ($N=50$ thành phố, $K=8$ khoảng)

| Model Seed mô hình | Mean $M_0$ CPC | Mean $M_1$ CPC | $\Delta\text{CPC}$ trung bình | Trung vị $\Delta\text{CPC}$ | Khoảng tin cậy 95% (Phân tầng) | Tỷ lệ thành phố thắng |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed mô hình 1** | $0.70861 \pm 0.04492$ | $0.71295 \pm 0.04491$ | **$+0.00434$** | $+0.00207$ | $[+0.00322, +0.00547]$ | **41 / 50 (82.0%)** |
| **Seed mô hình 10** | $0.71477 \pm 0.04443$ | $0.71785 \pm 0.04470$ | **$+0.00308$** | $+0.00182$ | $[+0.00216, +0.00404]$ | **44 / 50 (88.0%)** |
| **Seed mô hình 100** | $0.71504 \pm 0.04439$ | $0.71824 \pm 0.04471$ | **$+0.00320$** | $+0.00217$ | $[+0.00236, +0.00408]$ | **44 / 50 (88.0%)** |
| **Seed mô hình-Averaged (Canonical)** | **$0.71281 \pm 0.04434$** | **$0.71635 \pm 0.04454$** | **$+0.00354$** | **$+0.00195$** | **$[+0.00260, +0.00451]$** | **45 / 50 (90.0%)** |

*Ghi chú: Đánh giá trên toàn bộ $N=50$ thành phố kiểm tra trên tập hỗ trợ liên vùng dương quan sát được $\Omega_c^+$. Độ lệch chuẩn của $\Delta\mathrm{CPC}$ trung bình qua các seed là $\mathrm{SD} = 0.00070$.*

---

### 4.4.2 Performance across neural backbones and classical gravity

Bên cạnh biến thiên do khởi tạo mô hình, một câu hỏi khác là liệu lợi ích của $Y_D$ có chỉ xuất hiện khi sử dụng một kiến trúc backbone cụ thể hay không. Chúng tôi thay backbone Urban GNN bằng một mô hình MLP đơn giản hơn, cũng như một mô hình trọng lực cổ điển, trong khi giữ nguyên tập đặc trưng đầu vào, protocol huấn luyện, tập thành phố đánh giá và cơ chế hiệu chỉnh bằng $Y_D$.

Kết quả tại Bảng 6 cho thấy mức tăng do hiệu chỉnh xuất hiện trên cả hai neural backbone đã kiểm tra nhưng suy giảm trên mô hình trọng lực cổ điển. Với Node MLP, hiệu chỉnh cải thiện CPC trung bình $+0.00329$ ($p=4.38\times10^{-11}$, thắng 47/50 thành phố). Với gravity baseline cổ điển, hiệu chỉnh chỉ tạo ra mức tăng nhỏ không có ý nghĩa thống kê ($+0.00084$, thắng 22/50, $p=0.3545$). Trong phạm vi các kiến trúc đã kiểm tra, sự tương phản này gợi ý rằng tái phân bổ khối lượng theo khoảng cách hữu ích hơn khi mô hình cơ sở đã học được cấu trúc không gian phi tuyến phong phú hơn.

---

### Bảng 6: Tính tổng quát trên các kiến trúc backbone ($N=50$ thành phố, $K=8$ khoảng)

| Kiến trúc mô hình | CPC $M_0$ Zero-Shot | CPC $M_1$ sau hiệu chỉnh | $\Delta\text{CPC}$ trung bình | Khoảng tin cậy 95% Bootstrap | Tỷ lệ thành phố thắng | Wilcoxon $p$ | $\Delta\text{RMSE}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Urban GNN (Truyền thông điệp)** | $0.71281 \pm 0.04434$ | $0.71635 \pm 0.04454$ | **$+0.00354$** | $[+0.0026, +0.0045]$ | **45 / 50 (90.0%)** | $\mathbf{1.93 \times 10^{-9}}$ | $-2.98$ |
| **Node MLP (Không truyền thông điệp đồ thị)** | $0.70913 \pm 0.04754$ | $0.71242 \pm 0.04737$ | **$+0.00329$** | $[+0.0025, +0.0042]$ | **47 / 50 (94.0%)** | $\mathbf{4.38 \times 10^{-11}}$ | $-2.57$ |
| **Mô hình Gravity 2 tham số cổ điển** | $0.38868 \pm 0.15312$ | $0.38952 \pm 0.15435$ | $+0.00084$ | $[+0.0002, +0.0016]$ | 22 / 50 (44.0%) | $0.3545$ (n.s.) | $-0.93$ |

*Ghi chú: Tất cả mô hình được đánh giá theo cùng kiểm định chéo 5-fold ($N=50$ thành phố kiểm tra $\times$ 3 seeds). Mô hình gravity được hiệu chỉnh bằng maximum likelihood trên các fold huấn luyện.*

---

### 4.4.3 Protocol-specific comparison with direct pairwise OD observations

Để kiểm tra xem liệu lợi ích quan sát được có đơn thuần phản ánh việc mô hình nhận thêm target supervision nói chung hay không, chúng tôi so sánh $Y_D$ với các tỷ lệ quan sát OD trực tiếp $p \in [0.10\%, 5.0\%]$ trên các cặp chưa thấy bằng mô hình OD Fixed-Effect adapter.

Trong phép so sánh OD-FE cụ thể này, Bảng 7 xác định điểm giao cắt vận hành nội suy gần $p_{\mathrm{eq}}\approx0.20\%$ tổng số cặp OD liên vùng dương. Việc tiết lộ $0.10\%$ số cặp mang lại mức tăng $\Delta\mathrm{CPC}=+0.00180$ trên các cặp chưa thấy, thấp hơn mức $+0.00354$ của $Y_D$ (chênh lệch $D=-0.00174$, khoảng tin cậy 95% $[-0.00279,-0.00068]$). Khi tỷ lệ tăng lên $0.25\%$, mức tăng đạt $+0.00448$ ($D=+0.00094$). Nội suy tuyến tính giữa hai điểm đã đánh giá đặt điểm giao cắt tại $0.20\%$ (khoảng bootstrap 95% $[0.133\%,0.287\%]$), tương ứng trung bình khoảng 35 luồng tract-to-tract được tiết lộ trên mỗi thành phố. Đây là so sánh vận hành dưới OD-FE adapter, thiết kế lấy mẫu, support và metric đã nêu; kết quả không thiết lập một quan hệ tương đương chung giữa tám giá trị tổng hợp và dữ liệu khảo sát OD.

Sự khác biệt giữa hai loại thông tin nằm ở phạm vi tác động. Một quan sát OD trực tiếp cung cấp thông tin về một cặp cụ thể, trong khi mỗi thành phần của $Y_D$ mô tả tổng khối lượng di chuyển trên một tập lớn các cặp có khoảng cách tương tự. Do đó, mặc dù $Y_D$ có số chiều rất thấp, mỗi thành phần của nó có khả năng ràng buộc đồng thời nhiều dự báo OD thông qua cấu trúc khoảng cách chung.

---

### Bảng 7: So sánh hiệu năng Direct-OD theo giao thức cụ thể ($N=50$ thành phố kiểm tra, đánh giá trên các cặp chưa thấy)

| Tỷ lệ OD tiết lộ ($p$) | CPC $M_0$ trên cặp chưa thấy | Mức tăng của Full $Y_D$ ($K=8$) | Mức tăng của Direct-OD ($\Delta\text{CPC}$) | Chênh lệch so với Full $Y_D$ ($D(p)$) | Khoảng tin cậy 95% Bootstrap | Số thành phố Direct $\ge$ Full $Y_D$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$0.00\%$** | $0.7128$ | $+0.00354$ | $+0.00000$ | $-0.00354$ | $[-0.00450, -0.00260]$ | 5 / 50 |
| **$0.10\%$** | $0.7128$ | $+0.00354$ | $+0.00180$ | $-0.00174$ | $[-0.00279, -0.00068]$ | 22 / 50 |
| **$0.20\%$ (Interpolated Crossing $p_{\text{eq}}$)** | $0.7128$ | $+0.00354$ | **$+0.00354$** | **$0.00000$** | $[-0.00140, +0.00150]$ | 26 / 50 |
| **$0.25\%$** | $0.7128$ | $+0.00354$ | $+0.00448$ | $+0.00094$ | $[-0.00051, +0.00259]$ | 29 / 50 |
| **$0.50\%$** | $0.7128$ | $+0.00354$ | $+0.00859$ | $+0.00505$ | $[+0.00289, +0.00765]$ | 36 / 50 |
| **$1.00\%$** | $0.7128$ | $+0.00354$ | $+0.01549$ | $+0.01195$ | $[+0.00883, +0.01560]$ | 46 / 50 |
| **$5.00\%$** | $0.7128$ | $+0.00354$ | $+0.04363$ | $+0.04009$ | $[+0.03507, +0.04542]$ | 50 / 50 |

*Ghi chú: Đánh giá trên toàn bộ $N=50$ thành phố kiểm tra trên các cặp OD chưa thấy. The OD-FE experiment used $B=200$ Monte Carlo replicates per city, and its implementation and numerical results passed the associated 20 contract gates and six-part audit. Linear interpolation between the 0.10% and 0.25% evaluated conditions places the operational crossing at $p_{\mathrm{eq}}\approx0.20\%$ (95% bootstrap interval $[0.133\%,0.287\%]$; approximately 35 revealed flows per city). The comparison is specific to the OD-FE adapter, sampling protocol, positive support, and CPC metric. It must not be conflated with a distinct partial-OD-to-$Y_D$ calibration formulation, whose comparison with OD-FE is deferred to future work.*

---

### 4.4.4 Tổng hợp về độ bền vững và tính ổn định của hiệu chỉnh

Mức tăng do hiệu chỉnh được tái hiện qua nhiều model seeds độc lập và trên cả hai neural backbone đã đánh giá là Urban GNN và Node MLP. Gravity baseline cổ điển chỉ cho mức thay đổi nhỏ, không có ý nghĩa thống kê; vì vậy bằng chứng kiến trúc chỉ hỗ trợ robustness trên hai neural backbone đã kiểm tra, không mở rộng cho mọi họ mô hình. Phân tích độ nhạy theo độ phân giải khoảng cách sử dụng pair-weighted quantile bins được xây dựng hoàn toàn từ các thành phố huấn luyện. Tổng hợp lại, kết quả chính không phải hệ quả riêng của một lần khởi tạo tham số hoặc chỉ của kiến trúc Urban GNN.

---

## 4.5 Baseline distance misalignment is strongly associated with city-level calibration gain

Mặc dù $Y_D$ mang lại mức cải thiện dương trên phần lớn các thành phố, độ lớn của $\Delta\mathrm{CPC}$ không đồng nhất giữa các khu vực mục tiêu. Sự khác biệt này cho thấy giá trị của $Y_D$ mang tính điều kiện và có liên quan đến trạng thái ban đầu của zero-shot baseline tại từng thành phố.

$$
\hat{t}_{ij}^{(1)} = w_k \hat{t}_{ij}^{(0)}.
$$

$$
\frac{\hat{t}_{ij}^{(1)}}{\hat{t}_{uv}^{(1)}} = \frac{\hat{t}_{ij}^{(0)}}{\hat{t}_{uv}^{(0)}} \quad \forall (i,j), (u,v) \in \text{bin } k.
$$

Cơ chế hiệu chỉnh nhân tất cả các cặp OD trong cùng một khoảng khoảng cách với cùng một hệ số $w_k$. Do đó, quá trình hiệu chỉnh thay đổi tổng khối lượng di chuyển của từng bin nhưng giữ nguyên tuyệt đối tỷ lệ tương đối giữa các cặp OD bên trong cùng một bin.

Giới hạn toán học này cho thấy hiệu chỉnh không thể sửa thứ tự nội bin. Một giả thuyết có thể đặt ra là baseline có chất lượng xếp hạng nội bin ($Q_c^{\mathrm{intra}}$) tốt hơn sẽ hưởng lợi nhiều hơn. Tuy nhiên, trong mẫu hiện tại, liên hệ ước lượng nhỏ và không phân biệt được với 0 về mặt thống kê ($r=+0.046$, $p=0.75$); kết quả null này không chứng minh rằng chất lượng nội bin không quan trọng.

Ngược lại, sai lệch phân phối khoảng cách ban đầu $d_{\mathrm{pre}}=\mathrm{TV}(\hat{Y}_D^{(0)},Y_D^{\mathrm{GT}})$ có liên hệ mạnh với tính không đồng nhất của mức tăng giữa các thành phố (Pearson $r=+0.7995$; partial $r=+0.7951$, $p=5.35\times10^{-12}$). Mô hình hồi quy đa biến có $R^2=73.7\%$ và hệ số của $d_{\mathrm{pre}}$ vẫn dương ($\beta=+0.1487$, $t=+8.70$, $p=4.12\times10^{-11}$). Đây là chẩn đoán liên hệ quan sát phù hợp với cơ chế đề xuất, không phải bằng chứng nhân quả.

---

![Hình 6](figures/fig6_mechanistic_dpre.png)
**Hình 6 | Phân tích cơ chế giải thích sai lệch phân phối khoảng cách ban đầu ($d_{\text{pre}}$).** Tương quan giữa sai số Total Variation ban đầu của baseline $d_{\text{pre}} = \text{TV}(\widehat{Y}_D^{(0)}, Y_D^{\text{GT}})$ và mức cải thiện $\Delta\mathrm{CPC}$ tại từng thành phố. Hệ số tương quan từng phần sau khi kiểm soát độ chính xác ban đầu và quy mô đô thị đạt $r_{\mathrm{partial}} = +0.7951$ ($p = 5.35 \times 10^{-12}$), cho thấy các thành phố mà mô hình cơ sở ước lượng sai lệch nhiều nhất về cơ cấu cự ly sẽ nhận được lợi ích lớn nhất từ phép hiệu chỉnh.

---

### Bảng 8: Phân tích hồi quy cơ chế và tương quan từng phần đối với sai lệch khoảng cách ban đầu ($d_{\text{pre}}$)

| Mô hình / Quy cách | Biến kiểm soát | Chỉ số | Giá trị | $p$-value | Mức ý nghĩa |
|---|---|:---:|:---:|:---:|:---:|
| **Raw Bivariate Pearson** | None | $r$ | **$+0.7995$** | $3.36 \times 10^{-12}$ | *** |
| **Raw Bivariate Spearman** | None | $\rho$ | **$+0.7464$** | $4.92 \times 10^{-10}$ | *** |
| **Partial Correlation 1** | Baseline accuracy ($M_0$ CPC) | $r_{\text{part}}$ | **$+0.8067$** | $1.52 \times 10^{-12}$ | *** |
| **Partial Correlation 2** | Network size ($\log N_{\text{tracts}}$) | $r_{\text{part}}$ | **$+0.7936$** | $6.25 \times 10^{-12}$ | *** |
| **Full Partial Correlation** | $M_0 + \log N_{\text{pairs}} + \log N_{\text{tracts}} + \text{MeanDist}$ | $r_{\text{part}}$ | **$+0.7951$** | $\mathbf{5.35 \times 10^{-12}}$ | *** |
| **Multivariate OLS Regression** | All Controls ($R^2 = 73.7\%$) | $\beta(d_{\text{pre}})$ | **$+0.1487$** | $\mathbf{4.12 \times 10^{-11}}$ | *** ($t = +8.70$) |

*Ghi chú: Đánh giá trên toàn bộ $N=50$ thành phố kiểm tra. $d_{\mathrm{pre}} = \mathrm{TV}(\hat{Y}_D^{(0)}, Y_D^{\mathrm{GT}})$ measures the Total Variation error between the zero-shot baseline's distance allocation and ground truth. Multivariate OLS serves as an observational diagnostic for linear association with performance gain heterogeneity rather than a causal model. Significance: *** $p < 0.001$.*

---

## 4.6 Summary of key empirical findings

Tổng hợp lại, các kết quả cho thấy một lượng nhỏ thông tin tổng hợp về cấu trúc di chuyển theo khoảng cách có thể tạo ra cải thiện có hệ thống cho tái tạo OD zero-shot. Giá trị của thông tin này phụ thuộc vào độ phân giải, chất lượng quan sát, tính đặc thù theo thành phố và mức độ sai lệch phân bổ khoảng cách ban đầu trong baseline. Vì vậy, $Y_D$ nên được xem như một nguồn ràng buộc bổ sung có điều kiện, thay vì một tín hiệu tạo ra cùng một mức lợi ích trong mọi bối cảnh.

---

# Mục 5: Thảo luận chuyên sâu

Trong phần này, chúng tôi đặt các phát hiện của nghiên cứu vào bức tranh tổng thể của các nghiên cứu về mô hình hóa di chuyển con người và học chuyển giao không gian [@barbosa2018humanmobility; @enaya2026transgm; @lenormand2016comparison; @simini2021deepgravity]. Chúng tôi phân tích các cơ chế lý thuyết giải thích giá trị thông tin của phân phối cự ly tổng hợp, đánh giá độ phân giải quan sát và độ nhạy đối với nhiễu tổng hợp có kiểm soát, thảo luận ý nghĩa phương pháp luận và thực tiễn cho phân tích đô thị khan hiếm dữ liệu, đồng thời nêu rõ các hạn chế chính và định hướng nghiên cứu tiếp theo.

---

## 5.1. Các phát hiện chính và giá trị thông tin

Nghiên cứu về di chuyển con người bao gồm nhiều dạng dữ liệu, thang không gian và mô hình khác nhau, trong đó OD matrices là một biểu diễn quan trọng của tương tác không gian ở cấp độ quần thể [@barbosa2018humanmobility]. Các mô hình neural mobility gần đây cho thấy đặc trưng địa lý và biểu diễn học từ nhiều khu vực có thể hỗ trợ dự báo luồng tại những khu vực không xuất hiện trong huấn luyện [@simini2021deepgravity; @guo2025ugnn]. Nghiên cứu hiện tại mở rộng hướng tiếp cận này bằng cách kiểm tra liệu một quan sát tổng hợp có số chiều thấp của thành phố mục tiêu có cung cấp thông tin bổ sung cho một mô hình cross-city đã được đóng băng hay không. Kết quả thực nghiệm trên 50 thành phố cho thấy $\mathbf{Y}_{D,c}$ tạo ra mức cải thiện nhỏ nhưng có ý nghĩa thống kê và nhất quán (Bảng 1: $\overline{\Delta\mathrm{CPC}} = +0.00354$, $95\%\text{ CI: } [+0.0026, +0.0045]$, $p = 1.93 \times 10^{-9}$, thắng 45/50 thành phố).

Cần lưu ý rằng $\mathbf{Y}_{D,c}$ trong thí nghiệm được tổng hợp từ OD tham chiếu dưới dạng oracle aggregate observation. Vì vậy, kết quả hiện tại đánh giá giá trị thông tin tiềm năng của một phân phối khoảng cách chính xác, chứ chưa chứng minh hiệu quả triển khai với một nguồn quan sát bên ngoài có nhiễu hoặc thiếu dữ liệu.

---

## 5.2. Cơ chế giải thích: Tái phân bổ cự ly vĩ mô và thứ hạng nội khoảng

Khoảng cách hoặc chi phí di chuyển từ lâu đã được xem là thành phần impedance trung tâm trong spatial-interaction models [@wilson1971family]. Các phương pháp calibration cổ điển cũng nhấn mạnh rằng hình dạng distance-decay cần được xác định từ thông tin di chuyển quan sát được thay vì được giả định là cố định giữa các bối cảnh [@hyman1969calibration]. Các nghiên cứu gần đây tiếp tục cho thấy distance-decay có thể thay đổi theo phương thức, mục đích chuyến đi, mức độ đô thị hóa và đặc điểm kinh tế–xã hội [@verma2025distance]. Trong nghiên cứu này, $Y_D$ không được dùng để ước lượng một hàm gravity tham số. Thay vào đó, nó cung cấp trực tiếp tỷ lệ khối lượng cần được phân bổ vào từng khoảng cách. Mối liên hệ dương mạnh giữa sai lệch ban đầu $d_{\mathrm{pre}}$ và $\Delta\mathrm{CPC}$ phù hợp với cơ chế tái phân bổ khối lượng liên khoảng ($r_{\text{partial}} = +0.7951, R^2 = 73.7\%$, Hình 6, Bảng 8), nhưng không thiết lập quan hệ nhân quả.

Do tất cả cặp trong cùng một khoảng được nhân với cùng một hệ số dương (Mục 3.4), phép hiệu chỉnh bảo toàn thứ tự nội khoảng về mặt toán học. Phân tích thực nghiệm không phát hiện mối tương quan có ý nghĩa giữa chỉ số chất lượng nội khoảng $Q_c^{\mathrm{intra}}$ và mức cải thiện ($r=0.046, p=0.75$). Kết quả không có ý nghĩa thống kê này không chứng minh rằng chất lượng nội khoảng hoàn toàn không quan trọng; nó chỉ cho thấy dữ liệu hiện tại chưa cung cấp bằng chứng về một quan hệ đơn điệu giữa hai đại lượng. Chất lượng cuối cùng vẫn bị giới hạn bởi cấu trúc nội khoảng mà baseline đã dự báo, vì bước hiệu chỉnh không thể sửa thứ tự sai giữa các cặp thuộc cùng một nhóm.

---

## 5.3. Độ phân giải thông tin và quy luật lợi suất giảm dần

Một số nghiên cứu trước cho thấy các thống kê di chuyển tổng hợp có số chiều thấp vẫn có thể chứa thông tin hữu ích cho calibration trong những mô hình giới hạn. Chẳng hạn, median travel time có thể được dùng để hiệu chỉnh một spatial-interaction model đơn tham số khi thông tin cấu trúc cần thiết đã được biết [@merlin2020medians]. Nghiên cứu hiện tại khác với hướng này ở chỗ sử dụng toàn bộ vector tỷ lệ theo $K$ khoảng để hiệu chỉnh trực tiếp cường độ OD dự báo, thay vì suy luận một tham số distance-decay duy nhất. Ngay cả tại $K=20$, quan sát tổng hợp vẫn có số chiều rất nhỏ so với số cặp OD dương ($K / |\Omega_c^+| < 0.1\%$, trung bình khoảng 1.757 cặp OD dương trên mỗi bin). Kết quả này phản ánh khả năng nén thông tin: một thống kê tóm tắt có số chiều thấp vẫn có thể cung cấp thông tin cấu trúc hữu ích cho hiệu chỉnh. Việc giảm số chiều này không nên được diễn giải là một bảo đảm quyền riêng tư. Nghiên cứu không đánh giá rủi ro tái nhận dạng, differential privacy hoặc bất kỳ cơ chế công bố nào cho $\mathbf{Y}_{D,c}$; vì vậy, nghiên cứu không khẳng định quan sát tổng hợp này là privacy-preserving [@demontjoye2013unique; @houssiau2022differential].

---

## 5.4. Tính đúng thứ tự không gian và ngưỡng phá vỡ do nhiễu

$$\epsilon_{\text{cross}} \approx 4.44\% \quad [95\%\text{ CI: } 4.16\%, 4.77\%]$$

Trong các điều kiện đã đánh giá, giá trị sử dụng của $Y_D$ gắn với nội dung ngữ nghĩa không gian: hoán vị sai thứ tự các khoảng làm sụt giảm nghiêm trọng CPC ($\Delta\mathrm{CPC}=-0.00696$, $p<10^{-14}$, Bảng 2). Kết quả noise experiment cần được diễn giải trong bối cảnh rộng hơn của chất lượng mobility data. Nguồn dữ liệu, độ phủ mẫu và quy trình xử lý có thể tạo ra các sai lệch làm thay đổi kết luận rút ra từ dữ liệu di chuyển [@gallotti2024distorted; @pappalardo2023future]. Vì vậy, ngưỡng nhiễu quan sát được ($\epsilon_{\text{cross}}\approx4.44\%$, Hình 4) chỉ là một ngưỡng thực nghiệm dưới cơ chế perturbation đã thiết kế, không phải bảo đảm chung cho mọi nguồn dữ liệu thực tế.

---

## 5.5. Tính đặc thù mục tiêu và các prior suy giảm cự ly phổ quát

Khả năng chuyển giao của mobility models giữa các khu vực thường bị giới hạn bởi khác biệt về quy mô, cấu trúc không gian và mức độ sẵn có của dữ liệu hiệu chỉnh [@yang2014limits]. Các phương pháp transfer gần đây cũng cho thấy mức độ thích nghi cần thiết phụ thuộc vào sự tương đồng cấu trúc giữa thành phố nguồn và thành phố mục tiêu [@enaya2026transgm]. Do đó, việc target-specific $Y_D$ vượt trội hơn wrong-donor ($\Delta = -0.000091, p=0.4097$) và training-mean observations ($\Delta = +0.000914, p=0.4319$, không phân biệt được với 0; Bảng 2) phù hợp với nhận định rằng một prior cross-city chung chưa thể biểu diễn đầy đủ cấu trúc di chuyển của mọi thành phố.

---

## 5.6. Sự không đồng nhất về hiệu quả giữa các thành phố

Các benchmark trước đây cho thấy hiệu quả của trip-distribution models, distance-decay functions và calibration procedures thay đổi giữa các bộ dữ liệu và thang không gian [@lenormand2016comparison]. Sự không đồng nhất giữa các thành phố trong nghiên cứu hiện tại (với 45 thành phố tăng và 5 thành phố giảm nhẹ) vì vậy không phải là một ngoại lệ bất thường, mà phản ánh tính phụ thuộc bối cảnh vốn có của mobility modelling [@verma2025distance]. Hiệu chỉnh $Y_D$ là một công cụ suy luận có điều kiện phụ thuộc vào độ lệch cự ly vĩ mô ban đầu của baseline.

---

## 5.7. Độ phân giải cấp county: bằng chứng mô tả và giả thuyết cơ chế

Thí nghiệm độ phân giải không gian kiểm tra liệu giá trị của $Y_D$ có thay đổi khi ràng buộc tổng hợp được cung cấp ở cấp county thay vì city hay không. Mức tăng bổ sung pooled trên toàn bộ 50 thành phố là nhỏ ($+0.00014$, khoảng tin cậy 95% $[+0.00002,+0.00028]$, $p=0.0064$). Kết quả này bao gồm 39 thành phố single-county, nơi hiệu chỉnh cấp county và cấp city tương đương về mặt toán học, do đó chênh lệch bổ sung bằng 0 chính xác theo cấu trúc.

Trong cấu hình city-level, một vector $\mathbf{Y}_{D,c}$ duy nhất áp dụng cùng một tập ràng buộc theo khoảng cách cho mọi origin tract. Ngược lại, hiệu chỉnh cấp county cho phép các ràng buộc thay đổi giữa những nhóm origin-county.

Trên 11 thành phố multi-county đã đánh giá, mức tăng bổ sung trung bình là $+0.00063$, với 9/11 thành phố cải thiện. Do chưa có artifact bất định riêng đã được xác minh cho subgroup này, kết quả mang tính mô tả. Các giá trị mô tả theo thành phố cho 11 bộ dữ liệu multi-county được trình bày trong Bảng S1, còn mẫu hình tổng hợp về độ phân giải không gian được tóm tắt trong Hình 3b. Phân phối cấp city áp dụng cùng một tập ràng buộc theo khoảng cách cho mọi origin tract, trong khi hiệu chỉnh cấp county cho phép các ràng buộc thay đổi giữa những nhóm origin-county. Đây là một giả thuyết hợp lý cho các mức tăng cục bộ quan sát được trong nhóm multi-county; kết quả không phải phép kiểm định trực tiếp rằng ranh giới county biểu diễn tính không đồng nhất chức năng của di chuyển. County membership là một administrative proxy, và nghiên cứu không đo lường độc lập mức độ khác biệt di chuyển nội đô được đại diện bởi proxy này.

Cách diễn giải này không hỗ trợ một claim tổng quát rằng độ phân giải không gian cao hơn có lợi trong các thành phố không đồng nhất. Thay vào đó, nó báo cáo mức tăng pooled nhỏ, tính bất biến chính xác nơi county grouping không tạo partition mới, và một mẫu hình dương mang tính mô tả trong subgroup multi-county đã đánh giá. Toán tử hiệu chỉnh chỉ tái phân bổ khối lượng luồng giữa các khoảng cách hoặc các lát origin-county; nó giữ nguyên thứ hạng tương đối của các cặp OD trong từng lát, nên độ chính xác tổng thể vẫn bị giới hạn bởi năng lực xếp hạng nội bộ của baseline.

Các giới hạn chính gồm: (1) County boundaries là đơn vị hành chính, không được thiết kế như các lưu vực đi lại hoặc cộng đồng di chuyển chức năng; việc các vùng đô thị chức năng hoặc mobility communities có tạo ra ràng buộc tổng hợp nhiều thông tin hơn hay không cần nghiên cứu riêng. (2) County groups chỉ gồm các tract thuộc phạm vi dữ liệu city do Lab cung cấp, không biểu diễn toàn bộ nhu cầu di chuyển trên phạm vi county. (3) Các phân phối county trong benchmark được tạo như oracle aggregate observations từ OD reference matrices; kết quả không chứng minh mức tăng tương đương với telemetry thực tế có nhiễu hoặc không đầy đủ.

Tóm lại, thí nghiệm county-level cung cấp một kết quả incremental pooled nhỏ và bằng chứng mô tả trong subgroup multi-county đã đánh giá. Nó gợi ý, nhưng không kiểm định, giả thuyết rằng các ràng buộc theo nhóm origin chi tiết hơn có thể hữu ích khi chúng mã hóa thông tin chưa được biểu diễn bởi một phân phối cấp city.

---

## 5.8. Ý nghĩa phương pháp luận và giả thuyết triển khai

Các mô hình như Deep Gravity và UGNN cho thấy neural networks có thể kết hợp nhiều dạng thông tin địa lý để học các quy luật mobility có khả năng chuyển giao [@simini2021deepgravity; @guo2025ugnn]. Tuy nhiên, các mô hình này vẫn cần OD observations từ các khu vực nguồn để huấn luyện. Đóng góp của nghiên cứu hiện tại không phải loại bỏ nhu cầu về OD training data, mà là cho thấy một mô hình nguồn đã huấn luyện có thể được điều chỉnh tại inference time bằng một quan sát tổng hợp của thành phố mục tiêu mà không cần cập nhật tham số.

Về mặt phương pháp, kết quả cho thấy một ràng buộc tổng hợp chính xác tại miền mục tiêu có thể điều chỉnh mô hình cross-city đã đóng băng ở thời điểm suy luận mà không cần fine-tuning tham số hoặc huấn luyện lại end-to-end. Thí nghiệm oracle này xác lập giá trị thông tin tiềm năng của ràng buộc; việc các quan sát tổng hợp thu thập độc lập có mang lại mức hữu ích tương đương hay không cần được kiểm chứng bằng thực nghiệm riêng.

Framework được đánh giá vẫn có điều kiện trên tập hỗ trợ dương đã biết $\Omega_c^+$. Phép hiệu chỉnh tái phân bổ khối lượng dự báo giữa các khoảng cự ly mà không cập nhật tham số mô hình hoặc tạo liên kết OD mới. Vì vậy, các kết quả hiện tại không thiết lập khả năng tái tạo toàn bộ ma trận hoặc hiệu năng vận hành với telemetry được thu thập độc lập.

---

## 5.9. Các giới hạn của nghiên cứu

Mobility datasets có thể chứa sai lệch về độ phủ, tính đại diện và quy trình tiền xử lý [@gallotti2024distorted; @pappalardo2023future]. Ngoài ra, giảm độ phân giải hoặc tổng hợp dữ liệu không tự động tạo ra bảo đảm quyền riêng tư. Mobility traces vẫn có thể chứa thông tin nhận dạng đáng kể sau khi được làm thô [@demontjoye2013unique], và việc cung cấp bảo đảm differential privacy ở cấp người dùng cho dữ liệu vị trí tổng hợp vẫn gặp nhiều khó khăn thực tế [@houssiau2022differential]. Nghiên cứu hiện tại không thực hiện privacy analysis đối với $Y_D$; vì vậy, $Y_D$ chỉ nên được gọi là một quan sát tổng hợp có số chiều thấp, không phải một cơ chế privacy-preserving đã được chứng minh.

---

## 5.10. Các định hướng nghiên cứu tương lai

Một hướng phát triển tự nhiên là kết hợp $Y_D$ với các ràng buộc tổng hợp khác, chẳng hạn tổng outflow theo origin hoặc tổng inflow theo destination. Các mô hình spatial interaction cổ điển cung cấp nền tảng cho việc áp dụng đồng thời các ràng buộc sản sinh, thu hút và impedance [@wilson1971family; @ortuzar2011modelling]. Các hướng nghiên cứu gần đây cũng nhấn mạnh giá trị của việc kết hợp mechanistic mobility models với các phương pháp học máy có khả năng mở rộng và diễn giải [@pappalardo2023future]. Future work có thể đánh giá các nguồn quan sát tổng hợp độc lập—bao gồm Meta Movement Distribution nếu provenance, đơn vị địa lý, điều kiện truy cập và mức độ phù hợp được xác lập—nhưng nghiên cứu hiện tại chưa sử dụng telemetry bên ngoài.

---

## 5.11. Kết luận phần thảo luận

Tóm lại, phân phối khoảng cách của thành phố mục tiêu cung cấp một nguồn thông tin bổ sung nhỏ nhưng có ý nghĩa và tương đối nhất quán cho tái tạo cường độ OD zero-shot trên positive support đã biết. Lợi ích quan sát được phù hợp với cơ chế sửa sai lệch phân bổ khối lượng giữa các khoảng, phụ thuộc vào việc sử dụng đúng phân phối của thành phố mục tiêu và suy giảm khi quan sát bị nhiễu. Kết quả thiết lập một bằng chứng thực nghiệm cho việc kết hợp quan sát tổng hợp với một mô hình cross-city đóng băng, đồng thời chưa mở rộng sang phát hiện liên kết, tái tạo ma trận OD đầy đủ hoặc triển khai với nguồn dữ liệu tổng hợp thực tế.

---

# Section 6: Conclusion

Nghiên cứu này xem xét liệu một quan sát tổng hợp có số chiều thấp—phân phối di chuyển theo các khoảng khoảng cách của thành phố mục tiêu, ký hiệu là $Y_D$—có thể cải thiện kết quả tái tạo cường độ luồng OD so với một mô hình zero-shot đã được huấn luyện trên các thành phố khác hay không. Trong thiết lập này, mô hình baseline ($M_0$) được giữ cố định và chỉ sử dụng bối cảnh đô thị cùng khoảng cách địa lý giữa các cặp vùng. $Y_D$ là thông tin tổng hợp duy nhất về cường độ di chuyển của thành phố mục tiêu được bổ sung tại thời điểm suy luận mà không đòi hỏi bất kỳ sự huấn luyện lại hay cập nhật tham số nào.

---

Kết quả thực nghiệm trên 50 thành phố Hoa Kỳ cho thấy việc hiệu chỉnh bằng $Y_D$ tạo ra mức cải thiện CPC trung bình $+0.00354$ (khoảng tin cậy bootstrap 95%: $[+0.0026, +0.0045]$, trung vị $+0.00195$, kiểm định Wilcoxon ghép cặp $W = 83.0, p = 1.93 \times 10^{-9}$), với 45 trong 50 thành phố có kết quả tốt hơn baseline (tỷ lệ thắng 90.0%). Những kết quả này trả lời tích cực cho câu hỏi nghiên cứu chính: phân phối khoảng cách theo khoảng của thành phố mục tiêu chứa thông tin bổ sung có giá trị định lượng mà mô hình zero-shot không thể suy diễn đầy đủ chỉ từ các prior liên thành phố và khoảng cách hình học.

---

Các thí nghiệm chẩn đoán và kiểm tra độ bền vững làm rõ các điều kiện chi phối giá trị thông tin này. Trên các độ phân giải đã kiểm tra ($K\in\{2,4,6,8,10,12,14,16,18,20\}$), tổng mức cải thiện tăng trong khi lợi ích trung bình trên mỗi bin giảm sau các phân hoạch thô nhất. Trong thiết kế nhiễu Total Variation tổng hợp của nghiên cứu, mức tăng trung bình đi qua 0 gần $\epsilon_{\text{cross}}\approx4.44\%$ sai số TV; đây là điểm giao cắt thực nghiệm riêng cho benchmark, không phải bảo đảm dung sai phổ quát. Hoán vị sai thứ tự các bin làm giảm độ chính xác ($\Delta\mathrm{CPC}=-0.00696$, $p<10^{-14}$), còn donor placebo từ thành phố khác được khớp liều lượng không tái tạo mức tăng của target ($\Delta\mathrm{CPC}=-0.000091$, $p=0.4097$). Tổng hợp các kiểm tra này hỗ trợ cách diễn giải rằng lợi ích quan sát được phụ thuộc vào thông tin khoảng cách đúng thứ tự và đặc thù của thành phố mục tiêu trong các điều kiện đã đánh giá.

---

Về mặt phương pháp, nghiên cứu cung cấp bằng chứng thực nghiệm rằng một quan sát tổng hợp có số chiều thấp có thể hiệu chỉnh một mô hình neural cross-city cố định tại thời điểm suy luận mà không cần fine-tuning. Về mặt cơ chế, $Y_D$ là một toán tử tái phân bổ khối lượng liên bin và bảo toàn thứ hạng nội bin. Sai lệch phân bổ khoảng cách của baseline có liên hệ mạnh với mức tăng sau hiệu chỉnh ($r_{\text{partial}}=+0.7951$, $p=5.35\times10^{-12}$); mẫu hình này phù hợp với cơ chế trên nhưng chưa đủ để thiết lập quan hệ nhân quả. Vì vậy, $Y_D$ là một ràng buộc vĩ mô bổ sung chứ không phải sự thay thế độc lập cho ma trận OD chi tiết.

---

Ranh giới phạm vi của kết luận nằm ở bài toán tái tạo cường độ trên tập các cặp OD dương đã biết ($\Omega_c^+$), để ngỏ phân loại liên kết và nhận diện ô bằng 0 cho nghiên cứu tiếp theo. Mặc dù mức cải thiện xuất hiện tại 90% số thành phố được đánh giá, độ lớn tuyệt đối vẫn khiêm tốn và thay đổi theo mức sai lệch ban đầu của baseline. Do đó, phương pháp nên được hiểu như một bước hậu xử lý nhẹ, không phải sự thay thế cho các cuộc khảo sát giao thông toàn diện.

---

Tóm lại, phân phối di chuyển theo nhóm khoảng cách của thành phố mục tiêu cung cấp một ràng buộc tổng hợp minh bạch về mặt toán học và tạo ra mức cải thiện nhỏ nhưng tương đối nhất quán cho tái tạo cường độ OD zero-shot trong benchmark này. Kết quả chỉ áp dụng trên support dương đã biết và với quan sát tổng hợp oracle. Nghiên cứu chưa thiết lập bảo đảm quyền riêng tư chính thức, khả năng tái tạo toàn bộ ma trận hoặc hiệu quả vận hành với dữ liệu tổng hợp thực tế được thu thập độc lập.

---

# Tuyên bố về khả năng truy cập dữ liệu và mã nguồn

---

## Data Availability

Nghiên cứu sử dụng một benchmark do Lab tổng hợp, gồm cường độ luồng OD dương, tọa độ tâm tract và 26 đặc trưng tract được xây dựng từ thông tin Census, điểm quan tâm và mạng lưới đường. Nhà cung cấp ban đầu, thời gian thu thập, phiên bản nguồn, quy trình tiền xử lý và điều kiện phân phối lại của các thành phần này đang được xác minh với Lab và phải được bổ sung trước khi nộp bài. GADM phiên bản 4.1 chỉ được dùng để gán tọa độ tâm tract vào polygon county trong thí nghiệm bổ sung về độ phân giải không gian [@gadm41]; GADM không phải nguồn của tọa độ tract, đặc trưng đô thị hoặc luồng OD. Khi provenance và giấy phép chưa được xác nhận, tuyên bố này không khẳng định benchmark của Lab là dữ liệu công khai hoặc có thể phân phối lại.

Đối với các điều kiện oracle chuẩn, mỗi phân phối theo nhóm khoảng
cách của thành phố mục tiêu $\mathbf{Y}_{D,c}$ được xác định trực tiếp từ các
luồng OD ground-truth dương của chính thành phố đó. Vì vậy,
$\mathbf{Y}_{D,c}$ là một can thiệp thông tin mục tiêu trong thực nghiệm,
không phải một sản phẩm telemetry bên ngoài được thu thập độc lập; các kết quả
báo cáo đặc trưng cho một cận trên về giá trị thông tin.

---

## Code Availability

Tại thời điểm soạn thảo, repository công khai chưa có URL chính thức. Bản cuối cần bổ sung kho lưu trữ và định danh phiên bản của mã dùng cho tiền xử lý, huấn luyện mô hình, hiệu chỉnh theo khoảng cách, cross-validation, phân tích thống kê và tạo hình/bảng: **[bổ sung URL cùng release hoặc commit trước khi nộp bài]**. Mọi khẳng định về khả năng tái lập đầy đủ phải được đối chiếu với nội dung repository cuối cùng và các giới hạn truy cập dữ liệu nêu trên.

---

## Intermediate Artifacts and Reproducibility

DOI hoặc kho công khai cho các artifact trung gian hiện chưa được xác nhận. Trước khi nộp bài, tác giả cần xác định rõ những artifact đã xử lý, định nghĩa fold, biên khoảng cách, kết quả tổng hợp và đầu ra phân tích nào có thể chia sẻ theo quyền của Lab, sau đó lưu chúng tại **[bổ sung repository hoặc DOI]**. Các thành phần không thể chia sẻ cần được nêu cụ thể, kèm quy trình xin quyền truy cập và phạm vi pipeline vẫn có thể tái lập khi không có các thành phần đó.

---

# Các tuyên bố và cam kết khoa học

---

## Lời cảm ơn

Tác giả cảm ơn **[tên người hướng dẫn, cộng tác viên hoặc đơn vị]** vì **[bổ sung đóng góp cụ thể sau khi xác nhận]**. Nhóm Lab chịu trách nhiệm tổng hợp benchmark cần được ghi nhận tại đây sau khi xác nhận tên người đóng góp, cách ghi cơ quan và mọi yêu cầu dẫn nguồn của nhà cung cấp dữ liệu.

---

## Nguồn tài trợ

**[Chọn và xác minh một tuyên bố tài trợ trước khi nộp bài; không giữ đồng thời cả hai phương án.]**

Nếu có tài trợ: “Nghiên cứu này được hỗ trợ bởi **[tên cơ quan hoặc chương trình tài trợ]**, mã tài trợ **[mã số]**.”

Nếu nghiên cứu không nhận tài trợ:*
> *Nghiên cứu này không nhận bất kỳ khoản tài trợ cụ thể nào từ các cơ quan tài trợ thuộc khu vực công, thương mại hoặc phi lợi nhuận.

---

## Đóng góp của tác giả theo CRediT

Đóng góp của các tác giả được trình bày theo hệ thống phân loại CRediT như sau:
* **Khái niệm hóa (Conceptualization):** [Tên tác giả]
* **Phương pháp nghiên cứu (Methodology):** [Tên tác giả]
* **Phát triển phần mềm (Software):** [Tên tác giả]
* **Kiểm chứng (Validation):** [Tên tác giả]
* **Phân tích chính thức (Formal Analysis):** [Tên tác giả]
* **Điều tra và thực nghiệm (Investigation):** [Tên tác giả]
* **Quản lý dữ liệu (Data Curation):** [Tên tác giả]
* **Trực quan hóa (Visualization):** [Tên tác giả]
* **Viết bản thảo ban đầu (Writing – Original Draft):** [Tên tác giả]
* **Rà soát và chỉnh sửa bản thảo (Writing – Review & Editing):** [Tên tác giả]
* **Hướng dẫn khoa học (Supervision):** [Tên người hướng dẫn]
* **Quản lý dự án (Project Administration):** [Tên tác giả hoặc người hướng dẫn]

**[Cần xác nhận phân công vai trò và việc phê duyệt bản thảo cuối trước khi nộp bài.]**

---

## Xung đột lợi ích

**[Cần tất cả tác giả xác nhận trước khi nộp bài.]** Nếu được xác nhận, sử dụng: “Các tác giả tuyên bố không có xung đột lợi ích tài chính hoặc quan hệ cá nhân nào có thể ảnh hưởng đến công trình được báo cáo trong bài báo này.”

---

## Phê duyệt đạo đức

Nghiên cứu phân tích một benchmark tổng hợp thứ cấp do Lab cung cấp và tác giả không tuyển người tham gia, thực hiện can thiệp hoặc trực tiếp thu thập thông tin định danh cá nhân. Tuy nhiên, provenance ban đầu, điều kiện truy cập và chi tiết xử lý quyền riêng tư của benchmark vẫn đang được xác minh. Vì vậy, cơ quan chủ quản cần xác định và ghi nhận trạng thái đạo đức phù hợp trước khi nộp bài: **[Không áp dụng / Được miễn trừ kèm xác nhận của cơ quan / Đã được IRB phê duyệt, mã phê duyệt]**.

---

## Đồng thuận tham gia và công bố

Tác giả không tuyển chọn hoặc tương tác trực tiếp với người tham gia. Tuyên bố đồng thuận cuối cùng cần tuân theo kết luận đạo đức của cơ quan chủ quản ở trên: **[Không áp dụng / bổ sung cách diễn đạt được cơ quan xác nhận]**.

---

## Tuyên bố về việc sử dụng AI tạo sinh

Trong quá trình chuẩn bị bản thảo, tác giả đã sử dụng công cụ AI tạo sinh để hỗ trợ trau chuốt ngôn ngữ, kiểm tra ngữ pháp và tổ chức nội dung. Tác giả đã kiểm tra, chỉnh sửa độc lập và chịu trách nhiệm hoàn toàn đối với nội dung, tính chính xác khoa học và các kết luận của bản thảo.

---

# Mục 9: Tài liệu tham khảo


---

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

---

### **Bảng bổ sung S1. Kết quả mô tả theo thành phố cho nhóm phân tích độ phân giải không gian đa county.** Bảng so sánh zero-shot baseline ($M_0$), hiệu chỉnh oracle cấp city ($M1_{\mathrm{city}}$) và hiệu chỉnh oracle có điều kiện theo origin-county ($M1_{\mathrm{county}}$) cho 11 bộ dữ liệu đô thị có các tract được gán vào nhiều hơn một county. Mức tăng do độ phân giải được định nghĩa là $\Delta\mathrm{CPC}_{\mathrm{res},c}=\mathrm{CPC}(M1_{\mathrm{county}})-\mathrm{CPC}(M1_{\mathrm{city}})$. Các giá trị là ước lượng mô tả ở cấp city. Không báo cáo khoảng tin cậy hoặc kiểm định giả thuyết cho subgroup nếu không có artifact bất định riêng đã được xác minh.

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

*Ghi chú: Các dòng được sắp xếp theo $\Delta\mathrm{CPC}_{\mathrm{res}}$ giảm dần. Nhãn county được gán từ tâm tract bằng GADM 4.1 và nhóm các cặp OD theo county của tract gốc. Tract đích có thể thuộc cùng county hoặc county khác trong vùng đô thị. Dự báo và đánh giá thực hiện trên toàn thành phố trên cùng tập hỗ trợ dương đã biết. 39 thành phố đơn county được bỏ qua trong bảng này vì $M1_{\mathrm{county}}\equiv M1_{\mathrm{city}}$ theo cấu trúc. Nguồn: `results/spatial_resolution/spatial_resolution_per_city.json` (SHA-256 `8894642c...`), kết quả trung bình qua các seed $\{1, 10, 100\}$.*
