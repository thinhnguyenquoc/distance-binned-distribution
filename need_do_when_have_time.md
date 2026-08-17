Có, **giả định missing pair = 0 có một số lợi ích thực nghiệm**, nhưng đổi lại bạn phải chấp nhận một giả định mạnh.

Lợi ích lớn nhất là bạn có thể khôi phục bài toán thành full binary support:

[
z_{ij}=
\begin{cases}
1 & \text{pair xuất hiện, }T_{ij}>0\
0 & \text{pair không xuất hiện}
\end{cases}
]

Khi đó BCE trở lại có ý nghĩa:

[
e_{ij}\rightarrow \pi_{ij}=P(T_{ij}>0)
]

và magnitude head học:

[
e_{ij}\rightarrow \mu_{ij}.
]

Final flow:

[
\hat T_{ij}=\pi_{ij}\mu_{ij}.
]

Cái hay là model lúc này học được **hai thứ**:

1. pair nào có khả năng tồn tại flow;
2. nếu có thì flow lớn bao nhiêu.

Nhờ vậy bạn có thể predict trên full (N^2), thay vì chỉ trên support (\Omega_c).

Ví dụ:

[
A\rightarrow B=100
]

[
A\rightarrow C=20
]

[
A\rightarrow D=\text{missing}.
]

Nếu giả định missing = 0:

[
z_{AB}=1,\quad z_{AC}=1,\quad z_{AD}=0.
]

Model có thể học:

[
\pi_{AB}=0.95,\quad
\pi_{AC}=0.8,\quad
\pi_{AD}=0.05.
]

Đây là thông tin mà single-head ZTNB không học được.

Nhưng rủi ro là nếu thực tế (A\rightarrow D) bị thiếu vì filtering chứ không phải true zero, bạn đang tạo một **false negative**:

[
\text{unknown}\rightarrow0.
]

Khi đó BCE sẽ học sai support.

Vì vậy tôi sẽ không dùng assumption này làm primary design ngay. Tôi sẽ dùng nó như một **sensitivity experiment** rất hữu ích:

[
\boxed{
\text{Primary: missing = unknown}
}
]

so với

[
\boxed{
\text{Sensitivity: missing = zero}
}
]

Sau đó so sánh:

[
\Delta R_{\text{unknown}}
]

và

[
\Delta R_{\text{zero-assumption}}.
]

Nếu kết luận về (Y_D) vẫn giống nhau ở cả hai:

[
\Delta R>0,
]

thì paper mạnh hơn đáng kể, vì result **robust to support assumption**.

Đây mới là điểm “hay” nhất của giả định missing = 0: không nhất thiết vì nó đúng, mà vì nó tạo ra một **stress test rất tốt** cho conclusion chính của nghiên cứu.


