# Báo cáo Kỹ thuật Nội bộ: Đánh giá Khả năng Tái lập Mô hình UGNN (Guo et al., 2025) trên Benchmark Hiện tại

**Mục đích**: Tài liệu lưu trữ nội bộ phục vụ phản biện (rebuttal) các câu hỏi của reviewer về lý do không đưa nguyên bản UGNN vào làm baseline bổ sung trong bài báo.  
**Tham chiếu**: Guo, J., Bai, S., Li, X., Xian, K., Liu, E., Ding, W., & Ma, X. (2025). *A universal geography neural network for mobility flow prediction in planning scenarios*. Computer-Aided Civil and Infrastructure Engineering, 40, 5769–5789.

---

### 1. Bản chất Kiến trúc của UGNN
* **Cấu trúc đa nhánh (Multi-branch Architecture)**:
  * Nhánh biểu diễn địa lý/không gian: Sử dụng mạng tích chập kết hợp kết nối đầy đủ (CNN / FCNN) trích xuất đặc trưng bối cảnh từ cửa sổ lân cận dạng raster $3 \times 3$ bao quanh mỗi ô/vùng phân tích.
  * Nhánh đặc trưng cạnh (Edge/Pairwise features): Mã hóa khoảng cách địa lý, thuộc tính kết nối và ma trận trở lực giữa cặp nguồn–đích.
  * Nhánh đặc trưng đô thị toàn cục và thời gian (City-level / Temporal context): Đưa vào các chỉ số vĩ mô của thành phố và đặc tính chu kỳ thời gian.
* **Cơ chế dự báo**: Mặc dù về mặt khái niệm UGNN định nghĩa bài toán trên toàn bộ ma trận ứng viên nguồn–đích (full candidate OD matrix), trong thực nghiệm mô hình thực hiện dự báo theo từng cặp (pairwise prediction) kết hợp kỹ thuật lấy mẫu (sampling) đối với nhóm cặp có lưu lượng rất thấp (0–1 chuyến đi) để cân bằng dữ liệu trong quá trình tối ưu hóa.

---

### 2. Các Rào cản Kỹ thuật và Phương pháp luận đối với Benchmark Hiện tại

#### 2.1. Khác biệt về Định nghĩa Nhiệm vụ và Không gian Hỗ trợ ($\Omega_c$ vs. Sampling/Full Matrix)
* **UGNN gốc**: Được thiết kế cho kịch bản quy hoạch toàn diện, dự báo cường độ luồng trên không gian ứng viên đầy đủ (bao gồm cả phân loại/xử lý nhóm cặp có 0 hoặc 1 chuyến đi qua cơ chế sampling có chủ đích).
* **Bài báo hiện tại**: Cố định nghiêm ngặt phạm vi đánh giá trên tập hỗ trợ dương liên vùng đã biết $\Omega_c = \{(i,j) \in \mathcal{P}_c : t_{ij} \ge 1\}$. Mô hình không giải quyết bài toán phát hiện liên kết (link discovery) hay phân loại luồng bằng 0. Hàm mục tiêu sử dụng phân phối Zero-Truncated Negative Binomial (ZTNB) điều kiện hóa trên $t \ge 1$, khác hoàn toàn với giao thức huấn luyện và phân phối tổn thất của UGNN.

#### 2.2. Không tương thích về Hệ thống Đặc trưng Đầu vào
* **UGNN gốc**: Yêu cầu cấu trúc đầu vào không gian dạng lưới raster lân cận $3 \times 3$ cùng các lớp đặc trưng quy hoạch đô thị đa tầng (phân loại chi tiết công trình, hạ tầng giao thông phân cấp, đặc trưng bối cảnh toàn thành phố).
* **Dữ liệu 50 vùng đô thị Hoa Kỳ**: Mỗi Census Tract được biểu diễn bởi 26 đặc trưng đô thị thuộc các nhóm Census, POI và Road (chờ xác nhận provenance đầy đủ theo tài liệu dữ liệu gốc), không có sẵn cấu trúc lưới raster $3 \times 3$ của các đặc trưng lân cận chuẩn hóa cho cả 50 đô thị.

#### 2.3. Trạng thái Sẵn có của Mã nguồn (Code/Implementation Availability)
* Tính đến thời điểm hiện tại, trang xuất bản chính thức của bài báo (Guo et al., 2025) trên *Computer-Aided Civil and Infrastructure Engineering* chưa cung cấp một mã nguồn (repository) hoàn chỉnh kèm trọng số tiền huấn luyện có thể dùng để tái lập trực tiếp (*turnkey reproduction*) trên các tập dữ liệu ngoại lai.
* Bất kỳ nỗ lực tự dựng lại (re-implementation) nào đều sẽ dẫn đến một biến thể phái sinh ("Adapted UGNN"), đòi hỏi phải đưa vào nhiều giả định tùy ý về cấu trúc raster hóa, tỷ lệ lấy mẫu lớp 0–1, siêu tham số và bộ trích xuất đặc trưng, làm mất tính công bằng và độ tin cậy khi so sánh baseline.

---

### 3. Chiến lược Xử lý trong Bản thảo và Phản biện
* **Trong Bản thảo (`paper/full_paper_vi.md`, `paper/full_paper_en.md`)**:
  * Giữ nguyên trích dẫn UGNN [@guo2025ugnn] trong phần Giới thiệu và Nghiên cứu liên quan.
  * Giữ mô tả ngắn gọn rằng UGNN và các mô hình học sâu liên thành phố giải quyết bài toán chuyển giao bằng cách học quy luật từ các thành phố nguồn, trong khi bài báo này tập trung vào một câu hỏi trực giao: giá trị bổ sung của phân phối khoảng cách tổng hợp cấp thành phố mục tiêu khi giữ nguyên tham số mô hình zero-shot.
  * Tuyệt đối không đưa phần phân tích kỹ thuật chi tiết về UGNN ở trên vào bản thảo để tránh làm loãng trọng tâm bài báo.
* **Khi Trả lời Reviewer (Rebuttal)**:
  * Sử dụng các luận điểm tại Mục 1 và Mục 2 để giải thích tính không khả thi và rủi ro phương pháp luận nếu cố gắng tạo ra một bản "Adapted UGNN" không chính thức.
  * Nhấn mạnh rằng nghiên cứu đã thiết lập đối chứng nội tại nghiêm ngặt giữa GNN (với message passing) và MLP (cùng biểu diễn đầu vào, decoder, ZTNB loss, chỉ bỏ message passing) để kiểm tra tính ổn định của toán tử hiệu chỉnh trên các mức năng lực mô hình khác nhau.

---

### 4. Định vị Khoa học và Quy tắc Thuật ngữ Nghiêm ngặt (Naming & Positioning Protocol)

* **Cách phát biểu được phép (Approved Wording)**:
  > *"Thiết kế thuộc họ mô hình GNN dự báo tương tác không gian, nhưng được hiện thực hóa độc lập cho benchmark và likelihood của nghiên cứu này."*

* **Các cụm từ TUYỆT ĐỐI KHÔNG SỬ DỤNG (Prohibited Phrasings)**:
  * ❌ `"adapted from UGNN"`
  * ❌ `"our implementation of UGNN"`
  * ❌ `"UGNN-based model"`
  * ❌ `"reproduced UGNN"`
  * ❌ `"modified UGNN"`

* **Kết luận & Định danh**:
  * **CÓ**: Đây là một baseline GNN được thiết kế / thích nghi độc lập cho bài toán OD trên tập hỗ trợ dương $\Omega_c$ và hàm mất mát ZTNB của nghiên cứu này.
  * **KHÔNG**: Đây **không** phải là adapted UGNN.
  * **Tên gọi trong toàn bộ bài báo**: Giữ nguyên tên chuẩn mực là **`GNN`**.
  * **Định vị khoa học**: Một baseline GNN hiện thực hóa độc lập, dùng làm môi trường thực nghiệm để kiểm tra giá trị gia tăng của phép hiệu chỉnh phân phối khoảng cách, không phải là một đóng góp kiến trúc mạng mới.
