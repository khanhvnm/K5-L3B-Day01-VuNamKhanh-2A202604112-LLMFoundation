# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604112
- **Họ và tên:** Vũ Nam Khánh
- **SEED:** 7590

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| `day01_lab.ipynb` — notebook đã chạy, còn output | Có |
| `lab1_report.json` — báo cáo cuối | Có |
| `hand_calc.jpg` — ảnh tính tay có các bước | Có |

Notebook báo cáo **15/15 kiểm tra đạt**.

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.884 | 0.919 | 0.089 | 0 |
| BoW + MLP | 0.953 | 0.974 | 0.452 | 0 |
| MiniLM + kNN | 0.898 | 0.905 | 122.673 | 0 |
| MiniLM + LogReg | 0.944 | 0.927 | 36.962 | 0 |
| MiniLM + MLP head | 0.935 | 0.950 | 31.678 | 0 |

Trong các mô hình trên, BoW + MLP có F1 test cao nhất. MiniLM + kNN có độ trễ p95 cao nhất. Các mô hình chạy cục bộ trong bảng không phát sinh chi phí API theo kết quả báo cáo.

## 3. Dự đoán và thực tế

- **Learning rate:** Dự đoán `lr=50` làm loss không ổn định hoặc tăng. Trong phần trả lời ở notebook, kết quả thực tế lại giảm. Như vậy dự đoán này chưa đúng.
- **Bỏ ReLU:** Dự đoán F1 sẽ giảm đáng kể và gần với Logistic Regression. Nhưng khi chạy, F1 vẫn là `0.953`, không thay đổi so với mô hình có ReLU.
- **Embedding:** Dự đoán Encoder + LogReg sẽ làm tốt hơn BoW ở các câu khó. Kết quả thử 6 câu cho thấy Encoder + LogReg đúng 6/6, còn BoW đúng 5/6. Dự đoán này phù hợp với kết quả.

## 4. Phân tích 50 case khó

(Chưa hoàn thiện)

## 5. Thử thách cá nhân

(Chưa hiểu và hoàn thiện)

## 6. Phương án cho sản phẩm thật

Chưa có nhận định rõ

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Machine Learning | Logistic Regression dùng để làm gì trong bài toán phân loại spam? | AI giải thích Logistic Regression dùng xác suất để phân loại SMS thành spam hoặc ham. | Đối chiếu với phần Logistic Regression trong notebook. |
| Machine Learning | Sigmoid function là gì và tại sao Logistic Regression lại cần sigmoid? | AI giải thích sigmoid biến giá trị đầu ra thành khoảng từ 0 đến 1, có thể hiểu như xác suất. | Tự kiểm tra lại công thức `σ(z)=1/(1+e^-z)`. |
| Machine Learning | Accuracy, Precision, Recall và F1 khác nhau như thế nào? | AI giải thích ý nghĩa cơ bản của 4 metric và trường hợp nào mỗi metric quan trọng. | Đối chiếu công thức với hàm `metrics()` trong notebook. |
| Machine Learning | Nếu mô hình dự đoán một tin spam thành ham thì đó là loại lỗi gì? | AI giải thích đây là False Negative (FN). | Kiểm tra lại bằng confusion matrix và công thức Recall. |
| Machine Learning | Tại sao trong bài toán spam không nên chỉ nhìn vào Accuracy? | AI giải thích vì dữ liệu có nhiều tin ham hơn spam nên Accuracy cao chưa chắc mô hình phát hiện spam tốt. | Đối chiếu với baseline luôn đoán ham và F1 của các mô hình. |
| Gradient Descent | `gradient` trong quá trình train Logistic Regression có ý nghĩa gì? | AI giải thích gradient cho biết hướng cần thay đổi trọng số để giảm loss. | Kiểm tra lại công thức cập nhật `w = w - lr * gradient`. |
| Learning Rate | Learning rate quá nhỏ hoặc quá lớn có thể ảnh hưởng thế nào đến quá trình học? | AI giải thích learning rate nhỏ làm học chậm, lớn có thể làm cập nhật quá mạnh hoặc không ổn định. | Đối chiếu với thí nghiệm learning rate trong notebook. |
| Deep Learning | ReLU là gì và tại sao thường được dùng trong hidden layer? | AI giải thích ReLU giữ giá trị dương và đưa giá trị âm về 0, giúp neural network học quan hệ phi tuyến. | Đối chiếu kiến trúc MLP trong notebook. |
| Deep Learning | `loss.backward()` và `optimizer.step()` làm nhiệm vụ gì? | AI giải thích `backward()` tính gradient, còn `step()` cập nhật trọng số. | Kiểm tra trực tiếp trong training loop của notebook. |
| Deep Learning | Vì sao cần `optimizer.zero_grad()` trước mỗi batch? | AI giải thích PyTorch mặc định cộng dồn gradient nên cần xóa gradient cũ trước khi tính gradient mới. | Kiểm tra lại training loop trong notebook. |
| Overfitting | Làm sao nhận biết model bắt đầu overfit từ train loss và validation loss? | AI giải thích khi train loss tiếp tục giảm nhưng validation loss bắt đầu tăng thì có dấu hiệu overfitting. | Đối chiếu với biểu đồ loss và chọn checkpoint tốt nhất. |
| Embedding | Embedding là gì nếu giải thích cho người mới học AI? | AI giải thích embedding biến text thành vector số để máy tính có thể biểu diễn và so sánh ý nghĩa của text. | Đối chiếu với vector 384 chiều từ MiniLM. |
| Embedding | Cosine similarity dùng để làm gì trong bài toán embedding? | AI giải thích nó đo mức độ giống nhau về hướng giữa hai vector. | Tự tính lại `cosine([1,2,0],[2,1,2])`. |
| Embedding | Tại sao embedding có thể hiểu câu “get a phone for free” giống với một câu spam khác dù không có cùng từ? | AI giải thích embedding biểu diễn ý nghĩa/ngữ cảnh của câu chứ không chỉ dựa trên việc hai câu có cùng từ hay không. | Kiểm tra bằng thí nghiệm 3.B trong notebook. |
| kNN | kNN trong bài toán embedding hoạt động như thế nào? | AI giải thích tìm k vector gần nhất rồi dựa vào nhãn của các hàng xóm để dự đoán. | Đối chiếu với hàm `knn_predict()`. |
| LLM | Zero-shot và few-shot khác nhau như thế nào? | AI giải thích zero-shot không đưa ví dụ mẫu, còn few-shot cung cấp một số ví dụ trước khi yêu cầu model dự đoán. | Đối chiếu với `build_classification_prompt()`. |
| LLM | Tại sao prompt yêu cầu model chỉ trả về `spam` hoặc `ham`? | AI giải thích để kết quả dễ parse tự động thành nhãn 1 hoặc 0. | Kiểm tra với hàm `parse_label()`. |
| LLM | Tại sao gọi LLM cho từng SMS lại có latency cao hơn Logistic Regression? | AI giải thích vì mỗi SMS phải gửi request tới API và chờ model xử lý, trong khi Logistic Regression chạy trực tiếp trên máy. | Đối chiếu benchmark latency trong notebook. |
| Chi phí | Làm thế nào để tính chi phí API LLM dựa trên số input/output token? | AI giải thích lấy số token × đơn giá tương ứng cho input và output rồi cộng lại. | Tự tính lại `llm_cost_per_day = $10.50`. |
| So sánh | Tại sao F1 có thể phù hợp hơn Accuracy trong bài toán spam? | AI giải thích F1 kết hợp Precision và Recall, nên phản ánh tốt hơn khả năng phát hiện spam khi dữ liệu không cân bằng. | Đối chiếu với tỷ lệ spam/ham trong dataset. |
| Tổng kết | Nếu là một người mới học AI, tôi nên hiểu khác nhau cơ bản giữa ML, Deep Learning, Embedding và LLM như thế nào? | AI giải thích 4 cách tiếp cận theo mức độ đơn giản: đặc trưng thủ công → neural network → biểu diễn ngữ nghĩa → mô hình ngôn ngữ lớn. | Đối chiếu với 4 phần chính của Day 1. |

Qua bài này, tôi hiểu rằng các mô hình có cách xử lý văn bản khác nhau nên kết quả cũng khác nhau. F1 không phải yếu tố duy nhất cần quan tâm, vì tốc độ xử lý cũng quan trọng khi đưa vào sử dụng. Tôi cũng hiểu rằng kết quả tốt trên tập test thông thường chưa chắc có nghĩa là mô hình xử lý tốt mọi tin nhắn ngoài thực tế.
