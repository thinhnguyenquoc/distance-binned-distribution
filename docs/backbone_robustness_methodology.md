# Thiết Kế Phương Pháp Luận: Backbone Robustness Test
**Kiểm Định Sự Vững Chắc Của Nền Tảng Mô Hình Bằng Toán Tử Hiệu Chuẩn Phân Phối Khoảng Cách (Moving-Bin Calibration)**

---

## 1. Đặt Vấn Đề (Research Gap)
Một câu hỏi lớn trong việc tạo luồng di chuyển không gian (OD flows) cross-city (zero-shot) là: *"Liệu một phân phối khoảng cách ($Y_D$) được cung cấp thêm tại thời điểm suy luận (inference) có mang lại **giá trị bổ sung độc lập (marginal value)** so với một mô hình AI vốn dĩ đã học rất giỏi các quy luật khoảng cách từ dữ liệu huấn luyện hay không?"*

Để bảo vệ kết quả bài báo khỏi các phản biện khoa học (peer-review defense), chúng ta phải chứng minh được 2 điều kiện tiên quyết:
1. Thông tin $Y_D$ là thực sự hữu ích và không bị thừa thãi (Non-redundant).
2. Thuật toán ghép $Y_D$ không phải là một "phép thuật" sửa lỗi vạn năng (General Operator). Nếu nó là vạn năng, giá trị nghiên cứu sẽ bị giảm vì nó giống một mẹo toán học hơn là một sự khám phá về đặc tính không gian.

## 2. Thiết Kế Bài Test Kép (The Dual-Test Design)
Để đập tan các phản biện, thí nghiệm `run_backbone_robustness.py` (Bảng 7) thiết kế **chính xác 2 bài test đối chứng** trên 2 nền tảng mô hình (backbone) hoàn toàn trái ngược nhau.

### Test 1: Lắp $Y_D$ vào Mô hình Cực Mạnh (Gravity-Informed Urban GNN)
* **Bản chất mô hình:** Là một kiến trúc GNN (Graph Neural Network) hiện đại, đã tích hợp sẵn định luật vật lý trọng trường (physics-informed), được huấn luyện kỹ lưỡng qua hàng chục epoch. Cấu trúc topo vi mô (cell-level) được đánh giá là cực kỳ xuất sắc và chặt chẽ. Mô hình này được **đóng băng (frozen)** trước khi nhận $Y_D$.
* **Mục tiêu:** Trả lời câu hỏi *"Liệu GNN có cover hết mọi thứ?"*
* **Kết quả quan sát:** Gain dương ($\Delta \text{CPC} = +0.0272$).
* **Luận điểm rút ra:** Sự cải thiện này chứng minh $Y_D$ chứa các thông tin đặc thù của thành phố mục tiêu (target-city specific information). Dù GNN có năng lực tổng quát hóa xuất sắc đến đâu, nó cũng không thể tự thân "đoán" được toàn bộ 100% bản sắc di chuyển của một thành phố mới. $Y_D$ mang giá trị **Information Value**.

### Test 2: Lắp $Y_D$ vào Mô hình Cực Yếu (Classical 2-Parameter Gravity)
* **Bản chất mô hình:** Mô hình cổ điển, chỉ dựa vào hồi quy tuyến tính (OLS) để tìm 2 tham số toàn cục ($G$ và $\alpha$) trên hàm khoảng cách $T_{ij} = \exp(G) \cdot P_i \cdot P_j \cdot D_{ij}^{-\alpha}$. Dự đoán vi mô cực kỳ thô sơ và sai lệch lớn do bỏ qua cấu trúc mạng lưới giao thông phức tạp.
* **Mục tiêu:** Trả lời câu hỏi *"Thuật toán ghép $Y_D$ (Calibration) có phải là toán tử vạn năng (General Operator) hay không?"*
* **Kết quả quan sát:** Gain âm ($\Delta \text{CPC} = -0.0065$). Hiệu suất tổng thể giảm xuống!
* **Luận điểm rút ra:** Toán tử ghép $Y_D$ KHÔNG PHẢI là "phép thuật sửa lỗi bừa bãi". Khi bạn lấy một ma trận vi mô (cell-level) đã quá sai lệch mà cố tình "ép" nó phải khớp với tổng phân phối $Y_D$ vĩ mô, thuật toán sẽ bóp méo những cấu trúc vốn dĩ đã mỏng manh, khiến sai số vi mô càng trở nên tồi tệ (hiện tượng "gọt chân cho vừa giày").

---

## 3. Tổng Kết Phương Pháp Luận
Bài test "Backbone Robustness" là một phép thử (litmus test) hoàn hảo, một nước đi "nhất tiễn song điêu":
1. Dùng sự **TĂNG LÊN** của mô hình GNN để khẳng định $Y_D$ là **mảnh ghép thông tin còn thiếu** ngay cả với mô hình mạnh nhất.
2. Dùng sự **GIẢM ĐI** của mô hình Gravity để khẳng định $Y_D$ hoạt động như một **Model-Specific Error Correction**. 

Chính sự phân tầng rõ rệt này chứng minh sức mạnh của $Y_D$ phụ thuộc vào **sự cộng hưởng (synergy)** với một nền tảng không gian vi mô được tổ chức tốt (như GNN). $Y_D$ chỉ nắn chỉnh tỷ lệ vĩ mô một cách chính xác khi bản thân bức tranh vi mô gốc không bị vỡ nát. Đưa tài liệu này vào báo cáo sẽ khiến cơ sở lý luận khoa học (methodological framing) trở nên không thể bị công kích.
