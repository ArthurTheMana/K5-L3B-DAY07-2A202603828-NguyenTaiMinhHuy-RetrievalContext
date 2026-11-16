# Báo cáo Day 7

**MSSV:** 2A202603828  
**Họ và tên:** Nguyễn Tài Minh Huy  
**Bài thực hành:** Day 7, Retrieval & Context Engineering

## 1. Dữ liệu và Benchmark

### 1.1. Nguồn dữ liệu

Trong bài thực hành này, mình sử dụng hai tài liệu PDF đã được xử lý từ Day 6:

1. **AI & ML Fundamentals (UOW):** File `ai_ml_fundamentals_uow.pdf`, gồm 56 trang PDF. Nội dung giới thiệu những khái niệm cơ bản về Artificial Intelligence và Machine Learning.

2. **Introduction to Machine Learning (Olivier Colliot, 2023):** File `ml_introduction_colliot_2023.pdf`, gồm 26 trang PDF. Nội dung giới thiệu các phương pháp Machine Learning và một số ứng dụng trong y học.

Nguồn tham khảo của tài liệu Colliot: https://hal.science/hal-03957125v3

Hai tài liệu được chuyển thành **181 đoạn văn bản (chunks)** thông qua quy trình xử lý dữ liệu từ Day 6.

Mình sử dụng các chunks này để xây dựng hệ thống truy xuất thông tin (Retrieval) và cung cấp bằng chứng cho AI Agent khi trả lời câu hỏi.

Dữ liệu đã xử lý được lưu tại:

`data/converted/chunks.json`

Về quyền sử dụng, mình sử dụng các tài liệu này cho mục đích học tập. Quyền phân phối lại các file PDF gốc cần được kiểm tra riêng trước khi đăng lên repository công khai.

### 1.2. Embedding Model và cách lưu trữ dữ liệu

Đối với Dense Retrieval, mình sử dụng mô hình `sentence-transformers/all-MiniLM-L6-v2`.

Mô hình này chuyển mỗi đoạn văn bản thành một vector gồm 384 chiều, giúp hệ thống tìm kiếm dựa trên ý nghĩa của câu hỏi thay vì chỉ so sánh từ khóa.

Mình chọn MiniLM vì mô hình tương đối nhỏ, có thể chạy trên máy cá nhân và không cần sử dụng API trả phí để tạo embeddings.

Thông tin cấu hình:

| Thành phần | Cấu hình |
|---|---|
| Dữ liệu | 181 chunks từ Day 6 |
| Sparse Retrieval | TF-IDF |
| Dense Retrieval | MiniLM |
| Embedding Model | `all-MiniLM-L6-v2` |
| Embedding Dimensions | 384 |
| Hybrid Retrieval | Reciprocal Rank Fusion (RRF) |
| RRF k | 60 |
| Top-k | 3 |
| Context Budget đã chọn | 1.200 ký tự |
| Agent Model | `nvidia/nemotron-3.5-lightning:free` |
| API Provider | OpenRouter |

#### Vector Store sử dụng PostgreSQL và pgvector

Trong Day 6, mình đã sử dụng PostgreSQL kết hợp với pgvector để lưu embeddings. Database có 371 records, gồm Strategy A với 190 chunks và Strategy B với 181 chunks.

Khi thực hiện Day 7, mình kiểm tra và xác nhận 181 Chunk IDs của Strategy B khớp hoàn toàn với bộ dữ liệu đang sử dụng.

Tuy nhiên, chỉ có 61/181 chunks khớp chính xác về nội dung văn bản. Phần còn lại có những khác biệt về mã hóa ký tự khi xử lý PDF.

Để tránh ảnh hưởng đến dữ liệu gốc của Day 6, mình tạo bảng `day7_minilm_chunks` trong cùng PostgreSQL và lưu lại 181 embeddings bằng mô hình `all-MiniLM-L6-v2`, mỗi vector có 384 chiều.

Sau khi lưu, mình kiểm tra và xác nhận cả 181 records đều khớp với văn bản hiện tại thông qua SHA-256.

Mình tiếp tục cập nhật Dense Retrieval, Hybrid Retrieval và AI Agent để đọc embeddings từ PostgreSQL thay vì tạo lại toàn bộ vectors mỗi lần chạy.

Các phép tính cosine similarity và RRF vẫn được thực hiện bằng NumPy và Python. PostgreSQL đóng vai trò lưu trữ embeddings lâu dài (Persistent Vector Store).

Khi kiểm tra lại trên Dev, Dense Retrieval và Hybrid Retrieval giữ nguyên các chỉ số Hit@1, Hit@3 và Context Evidence Rate so với trước khi kết nối PostgreSQL.


### 1.3. Xây dựng Benchmark

Bộ benchmark gồm tổng cộng **24 câu hỏi**, trong đó có 20 câu có thể trả lời bằng tài liệu và 4 câu cố tình thiếu bằng chứng.

Mình chia các câu hỏi thành hai tập Dev và Test:

| Loại câu hỏi | Dev | Test | Tổng |
|---|---:|---:|---:|
| Answerable | 15 | 5 | 20 |
| Unanswerable | 2 | 2 | 4 |
| **Tổng** | **17** | **7** | **24** |

Đối với câu hỏi Answerable, mình kiểm tra đoạn tài liệu nào chứa thông tin cần thiết để trả lời và ghi nhận ID của chunk làm bằng chứng (Evidence).

Đối với các câu hỏi Unanswerable, mình kiểm tra cả hai tài liệu để xác định rằng thông tin được hỏi không xuất hiện trong nguồn dữ liệu.

Các câu Unanswerable được sử dụng để đánh giá khả năng từ chối trả lời (Abstention) của AI Agent.

Mình không đưa nhãn đáp án hoặc danh sách evidence chuẩn vào prompt của Agent khi chạy Test.

### 1.4. Các chỉ số đánh giá

Mình sử dụng những chỉ số sau để đánh giá Retrieval:

- **Hit@1:** Kiểm tra kết quả đứng đầu có chứa bằng chứng cần thiết hay không.
- **Evidence Hit@3:** Kiểm tra trong ba kết quả đầu tiên có đủ các bằng chứng cần thiết hay không.
- **Context Evidence Rate:** Kiểm tra context cuối cùng được đưa cho Language Model có chứa đầy đủ bằng chứng hay không.
- **Latency:** Thời gian hệ thống cần để thực hiện một truy vấn.

Đối với AI Agent, mình đánh giá thêm Grounded Answer, Citation Validity, Tool Use, Correct Abstention và False Abstention.

## 2. Retrieval và thử nghiệm

### 2.1. So sánh các phương pháp Retrieval

Mình thực hiện ba phương pháp Retrieval chính:

1. TF-IDF.
2. Dense Retrieval sử dụng MiniLM.
3. Hybrid Retrieval kết hợp TF-IDF và MiniLM bằng RRF.

Ngoài ra, mình thử thay đổi Context Budget để kiểm tra ảnh hưởng của cách chọn context.

Kết quả trên tập Dev:

| Cấu hình | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median Latency |
|---|---:|---:|---:|---:|
| TF-IDF Baseline | 10/15 | 11/15 | 10/15 | 0,47 ms |
| Dense MiniLM | 9/15 | 12/15 | 9/15 | 15,46 ms |
| Hybrid RRF, 800 ký tự | 10/15 | 12/15 | 11/15 | 18,14 ms |
| Hybrid RRF, 1.200 ký tự | 10/15 | 12/15 | 12/15 | 17,04 ms |

Các số liệu thời gian được lấy từ kết quả thực nghiệm đã tổng hợp trước đó và cần được đối chiếu với log gốc nếu sử dụng cho bản báo cáo chính thức.

### 2.2. Phương pháp 1: TF-IDF

TF-IDF là phương pháp tìm kiếm dựa trên mức độ quan trọng của các từ trong văn bản.

Mình sử dụng TF-IDF làm Baseline vì phương pháp này khá đơn giản, dễ triển khai và có tốc độ tìm kiếm nhanh.

Kết quả cho thấy TF-IDF đạt:

- Hit@1: 10/15.
- Evidence Hit@3: 11/15.
- Context Evidence Rate: 10/15.

Ưu điểm lớn nhất của TF-IDF là tốc độ. Tuy nhiên, nếu câu hỏi và tài liệu sử dụng các từ khác nhau để diễn đạt cùng một ý, TF-IDF có thể không tìm được đoạn văn phù hợp.

### 2.3. Phương pháp 2: Dense Retrieval với MiniLM

Dense Retrieval sử dụng Embedding Model để chuyển câu hỏi và các chunks thành vector.

Sau đó, hệ thống so sánh mức độ tương đồng giữa các vector để tìm những đoạn văn có ý nghĩa gần với câu hỏi.

Mình sử dụng MiniLM vì mô hình này có thể chạy cục bộ, tương đối nhẹ và phù hợp với quy mô dữ liệu của bài Lab.

Kết quả:

- Hit@1: 9/15.
- Evidence Hit@3: 12/15.
- Context Evidence Rate: 9/15.

Dense Retrieval tìm được đủ bằng chứng trong Top 3 nhiều hơn TF-IDF, nhưng Context Evidence Rate lại thấp hơn.

Điều này cho thấy việc tìm được chunk trong danh sách kết quả chưa đảm bảo rằng phần bằng chứng đó sẽ xuất hiện đầy đủ trong context cuối cùng.

### 2.4. Phương pháp 3: Hybrid Retrieval

Hybrid Retrieval kết hợp kết quả của TF-IDF và Dense Retrieval.

Mình sử dụng Reciprocal Rank Fusion (RRF) để kết hợp thứ hạng của các kết quả.

Phương pháp này giúp tận dụng cả hai cách tìm kiếm:

- TF-IDF có lợi thế khi câu hỏi chứa từ khóa rõ ràng.
- Dense Retrieval có lợi thế khi câu hỏi và tài liệu diễn đạt cùng một ý bằng những từ khác nhau.

Với Context Budget 800 ký tự, Hybrid Retrieval đạt:

- Hit@1: 10/15.
- Evidence Hit@3: 12/15.
- Context Evidence Rate: 11/15.

Kết quả này tốt hơn Dense Retrieval về Context Evidence Rate và tốt hơn TF-IDF về Evidence Hit@3.

### 2.5. Cải thiện 1: Tăng Context Budget

Ở thử nghiệm đầu tiên, mình tăng Context Budget từ 800 lên 1.200 ký tự.

Giả thuyết của mình là context dài hơn sẽ giúp hệ thống giữ lại đầy đủ bằng chứng, đặc biệt trong những trường hợp đoạn văn bị cắt do giới hạn số ký tự.

Kết quả:

| Chỉ số | 800 ký tự | 1.200 ký tự |
|---|---:|---:|
| Hit@1 | 10/15 | 10/15 |
| Evidence Hit@3 | 12/15 | 12/15 |
| Context Evidence Rate | 11/15 | 12/15 |

Context Evidence Rate tăng từ 11/15 lên 12/15.

Qua thử nghiệm này, mình nhận thấy giới hạn context có thể ảnh hưởng trực tiếp đến chất lượng bằng chứng mà AI Agent nhận được.

Tuy nhiên, context dài hơn cũng làm tăng lượng văn bản cần gửi cho Language Model.

### 2.6. Cải thiện 2: Context Selection và Neighboring Chunk Retrieval

Trong quá trình chạy AI Agent trên Dev, mình gặp một số câu hỏi mà hệ thống không trả lời được dù bằng chứng tồn tại trong tài liệu.

Mình kiểm tra các trace và nhận thấy có hai vấn đề.

Thứ nhất, kết quả tìm kiếm mới có thể không được ưu tiên khi xây dựng context. Thứ hai, một số đoạn văn bị chia thành nhiều chunks. Nếu Agent chỉ lấy một chunk thì có thể thiếu phần quan trọng của câu trả lời.

Để giải quyết, mình điều chỉnh cách chọn context:

1. Ưu tiên kết quả từ lần tìm kiếm mới nhất.
2. Khi phù hợp, lấy thêm chunk liền kề trong cùng tài liệu và cùng trang.

Mình gọi cách thứ hai là Neighboring Chunk Retrieval.

Thay đổi này được kiểm tra thông qua các câu hỏi Agent Dev, đặc biệt là Q06 và Q14.

Đây là cải thiện về cách Agent xây dựng context. Mình chưa có một benchmark Retrieval độc lập để chứng minh nó làm tăng Hit@1 hoặc Evidence Hit@3.

### 2.7. Cấu hình cuối cùng

Sau khi so sánh các kết quả trên Dev, mình lựa chọn:

| Thành phần | Cấu hình cuối |
|---|---|
| Retrieval | Hybrid TF-IDF + MiniLM |
| Ranking Fusion | RRF, k = 60 |
| Top-k | 3 |
| Context Budget | 1.200 ký tự |
| Context Selection | Ưu tiên kết quả mới |
| Context Expansion | Neighboring Chunk Retrieval |
| Maximum Search Calls | 4 |

Mình chọn cấu hình này vì nó có Context Evidence Rate tốt nhất trong các cấu hình Retrieval đã đo trên Dev, đồng thời giúp cải thiện khả năng trả lời của Agent trong những trường hợp thiếu context.

Sau khi chốt cấu hình, mình sử dụng cùng cấu hình đó để chạy Agent trên Test.

### 2.8. Retrieval Benchmark trên Test

Sau khi lựa chọn cấu hình trên Dev, mình tiến hành đánh giá TF-IDF Baseline và Hybrid RRF trên 5 câu Answerable Test.

Các câu hỏi được sử dụng gồm Q09, Q10, Q18, Q19 và Q20.

Trước khi đánh giá, mình kiểm tra và chuẩn hóa các đoạn Gold Evidence để chúng khớp với văn bản đã trích xuất từ PDF. Việc chuẩn hóa chỉ nhằm xử lý lỗi ký tự, xuống dòng và ranh giới chunks. Riêng Q19 có bằng chứng nằm trong hai chunks liên tiếp.

Mình giữ nguyên thuật toán và cấu hình Hybrid đã lựa chọn trên Dev, không điều chỉnh theo kết quả Test.

| Chỉ số | TF-IDF Baseline | Hybrid RRF |
|---|---:|---:|
| Hit@1 | 1/5 (20%) | 2/5 (40%) |
| Evidence Hit@3 | 3/5 (60%) | 3/5 (60%) |
| Context Evidence Rate | 2/5 (40%) | 3/5 (60%) |
| Median Latency | 0,455 ms | 11,04 ms |
| Average Context | 528,6 ký tự | 990,2 ký tự |
| Service Cost | 0 USD | 0 USD |

**Nhận xét**

Hybrid Retrieval đạt Hit@1 là 40%, cao hơn TF-IDF ở mức 20%.

Evidence Hit@3 của hai phương pháp đều đạt 60%. Tuy nhiên, Hybrid có Context Evidence Rate cao hơn, đạt 60% so với 40% của TF-IDF.

Điều này cho thấy Hybrid không tìm được nhiều bộ bằng chứng đầy đủ hơn trong Top 3, nhưng giữ lại bằng chứng trong context tốt hơn.

Đổi lại, thời gian truy vấn trung vị của Hybrid là 11,04 ms, cao hơn TF-IDF với 0,455 ms.

Mình nhận thấy Hybrid có lợi thế về chất lượng context, nhưng phải đánh đổi bằng thời gian xử lý. Do tập Test chỉ gồm 5 câu hỏi, kết quả này chưa đủ để khẳng định Hybrid luôn tốt hơn trên mọi dữ liệu.

**Kết luận**

Mình tiếp tục sử dụng Hybrid RRF với Top-k = 3, RRF k = 60 và Context Budget = 1.200 ký tự làm cấu hình cuối cùng.

Cấu hình này đã được lựa chọn dựa trên Dev trước khi chạy Test.

Kết quả chi tiết được lưu tại:

- `results/baseline_test/summary.json`
- `results/baseline_test/traces.json`
- `results/hybrid_test_1200/summary.json`
- `results/hybrid_test_1200/traces.json`


## 3. AI Agent và đánh giá

### 3.1. Cách AI Agent hoạt động

Mình xây dựng AI Agent sử dụng OpenRouter để kết nối với mô hình `nvidia/nemotron-3.5-lightning:free`.

Agent không trả lời trực tiếp bằng kiến thức có sẵn của mô hình mà phải thực hiện tìm kiếm trong tài liệu.

Quy trình hoạt động:

1. Agent nhận câu hỏi.
2. Language Model tạo truy vấn tìm kiếm.
3. Hybrid Retriever tìm các chunks liên quan.
4. Hệ thống chọn context và có thể bổ sung chunk liền kề.
5. Language Model đọc context để tạo câu trả lời.
6. Hệ thống kiểm tra citation ID.
7. Agent trả lời nếu có đủ bằng chứng hoặc từ chối nếu thông tin không đầy đủ.

Mình giới hạn Agent tối đa 4 lượt tìm kiếm cho mỗi câu hỏi.

Kết quả mỗi lần chạy được lưu lại dưới dạng JSON, bao gồm câu hỏi, trạng thái, câu trả lời, citations, kết quả tìm kiếm, context, số lần gọi Language Model, thời gian và chi phí.

### 3.2. Đánh giá AI Agent trên Dev

Tập Dev gồm 17 câu hỏi:

- 15 Answerable.
- 2 Unanswerable.

Kết quả cuối cùng:

| Chỉ số | Kết quả Dev |
|---|---:|
| Total Cases | 17 |
| Tool Use | 17/17 |
| Grounding Score 0 | 1 câu |
| Grounding Score 1 | 1 câu |
| Grounding Score 2 | 13 câu |
| Grounding trung bình | 1,80/2 |
| Tổng Grounding | 27/30 |
| Citation Validity | 18/18 |
| Answer Citation Coverage | 14/14 |
| Correct Abstention | 2/2 |
| False Abstention | 1/15 |
| Provider Errors | 0 |

Mình sử dụng các nhãn đánh giá đã chuẩn bị để tính metrics bằng `agent_metrics.py`.

Trong 15 câu Answerable, Agent trả lời được 14 câu. Câu Q17 bị từ chối dù bằng chứng có tồn tại trong nguồn tài liệu.

Hai câu Unanswerable là U01 và U02 đều được từ chối đúng.

### 3.3. Đánh giá AI Agent trên Test

Tập Test gồm:

- Q09, Q10, Q18, Q19, Q20.
- U03 và U04.

Sau khi chạy xong, mình thu được kết quả:

| Chỉ số | Kết quả Test |
|---|---:|
| Total Cases | 7 |
| Tool Use | 7/7 |
| Answered | 5/5 |
| Grounding Score 2 | 5 câu |
| Grounding trung bình | 2,00/2 |
| Tổng Grounding | 10/10 |
| Citation Validity | 7/8 |
| Answer Citation Coverage | 5/5 |
| Correct Abstention | 2/2 |
| False Abstention | 0/5 |
| Provider Errors trong trace cuối | 0 |

Đây là kết quả dựa trên các nhãn review của bài thực hành, không phải điểm chính thức do giảng viên chấm.

Cả 5 câu Answerable đều được Agent trả lời. Hai câu U03 và U04 được từ chối do không có đủ bằng chứng.

Trong lần chạy đầu tiên, U03 gặp lỗi từ OpenRouter vì mô hình sử dụng hết giới hạn output tokens mà chưa tạo được câu trả lời.

Mình chạy lại U03 với cấu hình giữ nguyên. Lần chạy sau, Agent trả về `abstain` và không còn lỗi hệ thống.

Kết quả Test cuối cùng được tổng hợp từ 7 trace đã hoàn thành.

### 3.4. So sánh Dev và Test

| Metric | Dev | Test |
|---|---:|---:|
| Total Cases | 17 | 7 |
| Answerable | 15 | 5 |
| Answered | 14/15 | 5/5 |
| Grounded Answer | 27/30 | 10/10 |
| Mean Grounding | 1,80/2 | 2,00/2 |
| Citation Validity | 18/18 | 7/8 |
| Citation Coverage | 14/14 | 5/5 |
| Correct Abstention | 2/2 | 2/2 |
| False Abstention | 1/15 | 0/5 |
| Tool Use | 17/17 | 7/7 |

Nhìn chung, Agent có kết quả tốt trên Test. Tuy nhiên, bộ câu hỏi chỉ có 7 trường hợp nên mình chưa thể kết luận rằng hệ thống sẽ luôn đạt kết quả tương tự khi sử dụng với dữ liệu khác.

### 3.5. Phân tích trước và sau khi cải thiện

Ở phiên bản Agent ban đầu, hệ thống trả lời được 12 trong tổng số 15 câu Answerable Dev.

Ba câu Q06, Q14 và Q17 đều bị từ chối.

Mình bắt đầu bằng việc điều chỉnh prompt để Agent có thể tìm kiếm thêm khi chưa có đủ bằng chứng.

Thay đổi này giúp Agent thử thêm các truy vấn, nhưng chưa giải quyết được toàn bộ các trường hợp thất bại.

Sau đó, mình điều chỉnh cách xây dựng context bằng cách ưu tiên kết quả tìm kiếm mới và bổ sung các chunks liền kề.

Kết quả được ghi nhận:

| Query ID | Trước cải thiện | Sau cải thiện | Nhận xét |
|---|---|---|---|
| Q06 | Abstain | Answered | Tìm được thông tin cần thiết |
| Q14 | Abstain | Answered | Bổ sung được bằng chứng từ chunk liền kề |
| Q17 | Abstain | Abstain | Chưa tìm được bằng chứng phù hợp |

Sau các thay đổi, số câu Answerable Dev được trả lời tăng từ **12/15 lên 14/15**.

### 3.6. Phân tích lỗi Q07

Q07 được Agent trả lời nhưng chưa đầy đủ.

Câu trả lời chỉ đề cập đến một phần nội dung và chưa bao quát các ý liên quan đến việc khám phá nhóm hoặc mẫu dữ liệu trong Unsupervised Learning.

Ngoài ra, phần citation sử dụng tài liệu Colliot thay vì nguồn UOW được yêu cầu trong benchmark.

Vì vậy, mình đánh giá Grounding của Q07 là 1/2 thay vì 2/2.

Trường hợp này cho thấy câu trả lời có citation không đồng nghĩa với việc đã trả lời đầy đủ và đúng nguồn.

### 3.7. Phân tích lỗi Q17

Q17 yêu cầu giải thích Overfitting dựa trên tài liệu Colliot.

Qua trace, mình nhận thấy Agent thực hiện ba lượt tìm kiếm nhưng không lấy được bằng chứng cần thiết trong các kết quả được chọn.

Do không có đủ context để trả lời, Agent quyết định từ chối.

Đây là một trường hợp False Abstention vì câu hỏi thật sự có thể trả lời dựa trên tài liệu gốc.

Mình xem đây là một hạn chế về Retrieval Ranking và Context Selection.

Trường hợp này cũng cho thấy Agent có thể hoạt động đúng theo nguyên tắc không tự tạo thông tin nhưng vẫn không trả lời được câu hỏi do bước Retrieval chưa tốt.

### 3.8. Phân tích citation Q19

Trong Test, Q19 hỏi về tiêu chí lựa chọn mô hình Linear Regression tốt nhất.

Agent trả lời rằng mô hình tốt nhất là mô hình giảm thiểu hàm chi phí `J(w1)`.

Nội dung câu trả lời phù hợp với tài liệu Colliot.

Tuy nhiên, citation `colliot_ml_2023_p022_paragraph_001` trỏ đến một chunk bị cắt ngay giữa từ `minimizes`.

Phần còn lại của câu nằm trong chunk kế tiếp.

Mặc dù context đã chứa cả hai chunks và câu trả lời đúng về mặt nội dung, citation được chọn chưa tự chứa đầy đủ bằng chứng.

Vì vậy, mình sử dụng cách đánh giá nghiêm ngặt và ghi nhận Citation Validity trên Test là 7/8.

### 3.9. Nhận xét về thời gian, chi phí và giới hạn

Trong bài thực hành này, các phương pháp Retrieval được chạy cục bộ.

TF-IDF có tốc độ nhanh nhất nhưng không phải lúc nào cũng tìm được đầy đủ bằng chứng.

Dense Retrieval và Hybrid Retrieval cần thêm thời gian để xử lý vector và xếp hạng kết quả.

Đối với Agent, mỗi câu hỏi cần gọi Language Model để tạo truy vấn và xây dựng câu trả lời. Vì vậy, thời gian xử lý thường cao hơn so với Retrieval độc lập.

Các trace Test cuối cùng ghi nhận 7 lượt tìm kiếm và 14 lần gọi Language Model. Chi phí được API báo cáo là 0 USD với mô hình miễn phí đang sử dụng.

Qua quá trình thực hiện, mình nhận thấy một số hạn chế:

- Dữ liệu PDF có lỗi mã hóa ký tự khi trích xuất.
- Một số chunks bị cắt giữa câu.
- Context Budget có thể làm mất một phần bằng chứng.
- Dense Retrieval không phải lúc nào cũng tốt hơn TF-IDF.
- Agent phụ thuộc vào khả năng phản hồi của OpenRouter.
- Bộ benchmark có số lượng câu hỏi tương đối nhỏ.
- Citation Validation tự động chưa kiểm tra đầy đủ ý nghĩa của nguồn được trích dẫn.

## 4. Thử thách nâng cao

Mình tập trung hoàn thành phần cơ bản của Day 7 trước.

Phần thử thách nâng cao yêu cầu xử lý tài liệu từ 100 trang trở lên, có bảng, hình ảnh và biểu đồ, đồng thời xây dựng bộ benchmark đánh giá riêng.

Trong kết quả hiện tại, mình chưa thực hiện đầy đủ phần này nên không báo cáo điểm thưởng hoặc tỷ lệ hoàn thành cho Advanced Challenge.

## 5. Khả năng tái lập (Reproducibility)

### 5.1. Cấu trúc dữ liệu và kết quả

Các file chính được sử dụng trong bài:

| File | Nội dung |
|---|---|
| `data/converted/chunks.json` | Các chunks được chuyển từ Day 6 |
| `data/converted/questions.json` | Bộ câu hỏi sau khi chuyển đổi |
| `data/benchmark_dev_reviewed.json` | Benchmark Dev đã kiểm tra |
| `results/unanswerable_review.csv` | Kết quả kiểm tra câu hỏi thiếu bằng chứng |
| `starter/agentic.py` | Chương trình AI Agent |
| `starter/agent_metrics.py` | Chương trình đánh giá Agent |
| `results/agent_metrics_dev.json` | Metrics trên Dev |
| `results/agent_metrics_test.json` | Metrics trên Test |
| `results/agent_test_combined.json` | Kết quả chi tiết của 7 câu Test |

Các kết quả Retrieval Dev được lưu trong những thư mục tương ứng với từng phương pháp.

### 5.2. Cách chạy Agent

Mình chạy chương trình bằng Python trên Windows PowerShell.

Ví dụ, chạy một câu hỏi Dev:

```powershell
py starter/agentic.py --id Q01
```

Chạy một câu hỏi Test:

```powershell
py starter/agentic.py --split test --id Q09
```

Chạy toàn bộ bảy câu Test:

```powershell
@('Q09','Q10','Q18','Q19','Q20','U03','U04') |
ForEach-Object {
    py starter/agentic.py --split test --id $_
}
```

Để đánh giá kết quả Test, mình sử dụng:

```powershell
py starter/agent_metrics.py `
  --cases results/agent_cases_test.json `
  --traces results/agent_traces_test.json `
  --reviews results/agent_reviews_test.json `
  --split test `
  --output results/agent_metrics_test.json
```

Các API Key được quản lý bằng file `.env` trên máy cá nhân và không được đưa vào repository.

Để tái lập đầy đủ, người chạy cần chuẩn bị môi trường Python, cài đặt các thư viện cần thiết, có quyền truy cập vào nguồn dữ liệu và cấu hình OpenRouter API Key của riêng mình.


#### 5.2.1. Tạo lại Vector Store từ Day 6

Để chạy lại hệ thống, mình sử dụng PostgreSQL 16 kết hợp pgvector từ môi trường Docker của Day 6.

Database được truy cập qua `127.0.0.1:55433`. Trong Day 7, mình tạo bảng `day7_minilm_chunks` riêng để lưu 181 embeddings, không thay đổi dữ liệu gốc của Day 6.

**Bước 1: Khởi động PostgreSQL**

Mở Docker Desktop và khởi động container PostgreSQL của Day 6.

Kiểm tra bằng lệnh:

```powershell
docker ps
```

**Bước 2: Thiết lập kết nối**

Trong PowerShell:

```powershell
$cid = "day6_student_starter-postgres-1"
$env:DAY7_DB_USER = (docker exec $cid printenv POSTGRES_USER).Trim()
$env:DAY7_DB_NAME = (docker exec $cid printenv POSTGRES_DB).Trim()
$env:DAY7_DB_PASSWORD = (docker exec $cid printenv POSTGRES_PASSWORD).Trim()
```

Tên container và port cần được điều chỉnh nếu chạy trên môi trường Docker khác.

**Bước 3: Tạo bảng lưu embeddings**

```powershell
docker exec $cid psql -U $env:DAY7_DB_USER -d $env:DAY7_DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;" -c "CREATE TABLE IF NOT EXISTS public.day7_minilm_chunks (chunk_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, page INTEGER, content TEXT NOT NULL, text_sha256 TEXT NOT NULL, embedding vector(384) NOT NULL);"
```

**Bước 4: Tạo và nạp embeddings**

Cài đặt các thư viện cần thiết:

```powershell
py -m pip install "psycopg[binary]" sentence-transformers numpy
```

Sau đó, đọc 181 chunks từ `data/converted/chunks.json`, sử dụng MiniLM `all-MiniLM-L6-v2` để tạo embeddings đã chuẩn hóa (normalized embeddings) và lưu vào bảng `day7_minilm_chunks`.

Mỗi record gồm Chunk ID, Source ID, Page, Content, SHA-256 và vector 384 chiều.

Lệnh tạo và nạp dữ liệu cần được chạy trước lần đầu sử dụng Dense Retrieval, Hybrid Retrieval hoặc Agent trên một database mới.


**Lệnh tạo và nạp 181 embeddings vào PostgreSQL**

Sau khi thiết lập các biến môi trường ở Bước 2 và tạo bảng ở Bước 3, mình chạy lệnh sau để tạo lại embeddings từ dữ liệu Day 6 đã chuyển đổi:

```powershell
py -c "import os,json,hashlib,psycopg; from pathlib import Path; from sentence_transformers import SentenceTransformer; cs=json.loads(Path('data/converted/chunks.json').read_text(encoding='utf-8-sig')); assert len(cs)==181; model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); vec=model.encode([c['text'] for c in cs],normalize_embeddings=True,convert_to_numpy=True); assert vec.shape==(181,384); rows=[(c['id'],c['source_id'],c['page'],c['text'],hashlib.sha256(c['text'].encode('utf-8')).hexdigest(),'['+','.join(format(float(v),'.9g') for v in a)+']') for c,a in zip(cs,vec)]; conn=psycopg.connect(host='127.0.0.1',port=55433,dbname=os.environ['DAY7_DB_NAME'],user=os.environ['DAY7_DB_USER'],password=os.environ['DAY7_DB_PASSWORD']); cur=conn.cursor(); cur.executemany('INSERT INTO public.day7_minilm_chunks (chunk_id,source_id,page,content,text_sha256,embedding) VALUES (%s,%s,%s,%s,%s,%s::vector) ON CONFLICT (chunk_id) DO UPDATE SET source_id=EXCLUDED.source_id,page=EXCLUDED.page,content=EXCLUDED.content,text_sha256=EXCLUDED.text_sha256,embedding=EXCLUDED.embedding',rows); conn.commit(); print('Stored vectors:',len(rows),'| Dimensions:',vec.shape[1]); cur.close(); conn.close()"
```

Kết quả mong đợi:

`Stored vectors: 181 | Dimensions: 384`

Lệnh chỉ thêm hoặc cập nhật dữ liệu trong bảng `day7_minilm_chunks`, không thay đổi bảng `day6_chunks`.

Sau khi chạy, mình sử dụng câu lệnh SQL ở Bước 5 để kiểm tra số lượng records và kích thước vector.


**Bước 5: Kiểm tra Vector Store**

```powershell
docker exec $cid psql -U $env:DAY7_DB_USER -d $env:DAY7_DB_NAME -c "SELECT COUNT(*) AS total, MIN(vector_dims(embedding)) AS dimensions FROM public.day7_minilm_chunks;"
```

Kết quả mong đợi:

- Total: 181
- Dimensions: 384

Khi Vector Store đã được chuẩn bị, các chương trình Dense Retrieval, Hybrid Retrieval và Agent sẽ đọc embeddings từ PostgreSQL.

Việc kết nối sử dụng các biến môi trường `DAY7_DB_USER`, `DAY7_DB_NAME` và `DAY7_DB_PASSWORD`.

Mình không đưa mật khẩu PostgreSQL hoặc API Key lên GitHub.


### 5.3. Nguồn tham khảo

- VinUni Day 7 Lab Repository: https://github.com/VinUni-AI20k/K5-L3-Day07-Retrieval-Context-Engineering
- Olivier Colliot, Introduction to Machine Learning: https://hal.science/hal-03957125v3
- Sentence Transformers, MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- OpenRouter Documentation: https://openrouter.ai/docs

## 6. Kết luận

Trong bài thực hành Day 7, mình đã sử dụng dữ liệu từ Day 6 để thử nghiệm ba phương pháp Retrieval gồm TF-IDF, Dense Retrieval và Hybrid Retrieval.

Qua các kết quả trên Dev, mình nhận thấy không có phương pháp nào tốt nhất ở mọi chỉ số. TF-IDF có tốc độ nhanh, Dense Retrieval có thể tìm được những đoạn văn gần nghĩa với câu hỏi, còn Hybrid Retrieval giúp kết hợp ưu điểm của cả hai phương pháp.

Mình cũng nhận thấy rằng việc tìm được đúng chunk chưa chắc đã đủ để Language Model trả lời. Nếu phần bằng chứng bị cắt hoặc không được đưa vào context, Agent vẫn có thể thất bại.

Sau khi tăng Context Budget và cải thiện cách chọn context, Agent trả lời được thêm hai câu hỏi Dev, nâng số câu Answerable được trả lời từ 12/15 lên 14/15.

Ở tập Test, Agent trả lời được cả 5 câu Answerable và từ chối đúng 2 câu Unanswerable. Tuy nhiên, phần kiểm tra citation vẫn phát hiện một trường hợp nguồn trích dẫn chưa chứa đầy đủ nội dung hỗ trợ câu trả lời.

Mình hiểu rõ hơn vai trò riêng của Retrieval, Context Engineering và Agent Evaluation. Mình cũng học được rằng không nên chỉ nhìn vào việc AI có trả lời được hay không, mà cần kiểm tra xem câu trả lời có đủ bằng chứng, trích dẫn có chính xác và hệ thống có biết từ chối khi thiếu thông tin hay không.
