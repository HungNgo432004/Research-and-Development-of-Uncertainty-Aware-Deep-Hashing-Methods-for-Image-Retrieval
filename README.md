# Nghiên cứu và phát triển phương pháp băm sâu nhận biết độ bất định cho tra cứu ảnh

Research and Development of Uncertainty-Aware Deep Hashing Methods for Image Retrieval

## Thành viên

| Họ và tên | Mã sinh viên |
|---|---|
| Ngô Thế Hưng | B22DCKH058 |
| Đào Trường Vũ | B22DCKH132 |
| Dương Trí Dũng | B22DCKH017 |

## Nội dung chính

Đồ án tập trung vào bài toán truy xuất ảnh quy mô lớn bằng Learning-to-Hash (L2H), trong đó mỗi ảnh được ánh xạ thành một mã hash nhị phân ngắn (16/32/64 bit) để so khớp bằng khoảng cách Hamming. Hạn chế cốt lõi của các phương pháp hashing tất định hiện có là hiện tượng "hòa điểm Hamming": vì mã hash K bit chỉ cho tối đa K+1 giá trị khoảng cách, hàng nghìn ảnh có thể cùng khoảng cách với truy vấn, khiến thứ tự trả về trong nhóm hòa gần như ngẫu nhiên.

Nhóm nghiên cứu và cài đặt lại phương pháp trong bài báo "Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing" (TMLR 2024), tách bạch rõ ba thành phần: ProbHash, định lượng uncertainty và HashUQ.

Trước khi triển khai code, nhóm chuẩn bị một báo cáo mô tả chính xác pipeline (train → MC Dropout inference → sinh binary code + uncertainty → retrieval ranking), công thức tính uncertainty, vai trò MC Dropout, và thuật toán retrieval. Nội dung thực hiện gồm hai phần:

- **(a) Tái hiện kết quả gốc:** cài đặt ProbHash, module t-test, thuật toán HashUQ/HashUQ-Bin, tái lập bảng kết quả trên các bộ dữ liệu chuẩn.
- **(b) Mở rộng để tạo đóng góp riêng:** đánh giá độ ổn định qua nhiều seed, thử backbone hiện đại hơn AlexNet, so sánh với baseline UQ đơn giản hơn, chi phí thấp hơn.

Demo cuối kỳ là một hệ thống retrieval đơn giản minh họa kết quả; trọng tâm đồ án vẫn là phương pháp và thực nghiệm.

## Cơ sở dữ liệu ban đầu

Sử dụng ba bộ dữ liệu chuẩn của lĩnh vực image retrieval, theo đúng thiết lập trong bài báo gốc:

| Bộ dữ liệu | Mô tả | Độ đo |
|---|---|---|
| ImageNet | Tập con 100 category được chọn để huấn luyện và đánh giá (bộ gốc có hơn 10 triệu ảnh, 20.000 synset) | mAP@1000 |
| MS COCO | 132.218 mẫu ảnh thuộc 80 category, mỗi ảnh có thể gắn nhiều nhãn | mAP@5000 |
| NUS-WIDE | 269.648 mẫu ảnh thu thập từ Flickr, sử dụng 21 concept phổ biến nhất trong 81 concept gốc (mỗi concept có ít nhất 5.000 ảnh) | mAP@5000 |

Mỗi bộ được chia thành tập train/validation/query/pool theo đúng phân chia trong bài báo. Backbone dùng chung là AlexNet với 3 lớp fully-connected (4096 chiều) thêm vào cuối, dropout rate 0.5 ở các lớp này (dùng cho cả regularization lúc train và MC Dropout lúc inference). Nếu triển khai hướng mở rộng sang dữ liệu thời trang, nhóm dự kiến bổ sung thêm bộ FashionIQ hoặc DeepFashion.

## Timeline

Kế hoạch 10 tuần, từ 24/9/2026 đến 1/12/2026.

### Tuần 1: 24/9 – 30/9

- Viết báo cáo mô tả pipeline (train → MC Dropout inference → sinh binary code + uncertainty → retrieval ranking)
- Viết công thức uncertainty (t-test), vai trò MC Dropout, thuật toán retrieval
- Chốt dataset chính để làm sâu trước
- Chuẩn bị tài nguyên tính toán (GPU/Colab)
- **Mốc:** nộp báo cáo pipeline cho cô trước 30/9

### Tuần 2: 1/10 – 7/10

- Dựng môi trường
- Clone code baseline GreedyHash (Label-Target)
- Clone code baseline CSQ (Center-Target)
- Viết data pipeline (train/val/query/pool split) cho dataset chính
- Chuẩn hoá backbone AlexNet + 3 FC (4096-d, dropout 0.5)
- **Mốc:** chạy baseline gốc, ghi lại mAP tham chiếu trước 7/10

### Tuần 3–4: 8/10 – 21/10

- Cài đặt closed-form ELBO cho Center-Target
- Cài đặt ELBO cho Label-Target
- Train cả 2 construction, kiểm tra hội tụ
- Viết eval script mAP@r dùng chung
- Chạy ablation straight-through vs closed-form
- **Mốc:** đạt mAP tương đương/gần baseline trước 21/10

### Tuần 5: 22/10 – 28/10

- Cài MC Dropout inference (N_sample forward pass, tính π_{n,k})
- Cài paired t-test → p-value P_k
- Tính tổng log P_k thành uncertainty score
- Vẽ boxplot mAP theo quantile uncertainty để kiểm định định tính
- Thiết kế cấu trúc lưu database (hash-code + uncertainty nén 1-2 bit)
- **Mốc:** module uncertainty hoạt động đúng trước 28/10

### Tuần 6: 29/10 – 4/11

- Cài thuật toán HashUQ (ranking D_Hamming + α·uncertainty)
- Cài HashUQ-Bin (uncertainty nhị phân hoá)
- Chạy full thực nghiệm trên dataset chính ở 3 bit-length (16/32/64)
- Tổng hợp bảng so sánh baseline / ProbHash / HashUQ / HashUQ-Bin
- Báo cáo tiến độ với cô ở mốc này
- **Mốc:** có bộ số liệu đầy đủ đầu tiên trước 4/11

### Tuần 7: 5/11 – 11/11

- Mở rộng pipeline sang dataset thứ 2
- Mở rộng pipeline sang dataset thứ 3
- Ưu tiên 1-2 bit-length nếu thời gian gấp
- Bắt đầu thí nghiệm multi-seed (3-5 seed) trên dataset chính
- **Mốc:** có kết quả sơ bộ trên cả 3 dataset trước 11/11

### Tuần 8: 12/11 – 18/11

- Thử backbone hiện đại hơn AlexNet (ResNet-18/MobileNet)
- So sánh mAP giữa AlexNet và backbone mới
- So sánh t-test với baseline UQ rẻ hơn (giảm N_sample)
- So sánh t-test với Entropy/Variance
- Bắt đầu dựng demo (giao diện query → top-k)
- Thêm toggle bật/tắt HashUQ trong demo
- **Mốc:** hoàn thành các thí nghiệm mở rộng trước 18/11

### Tuần 9: 19/11 – 25/11

- Hoàn thiện demo
- Tổng hợp bảng/biểu đồ cho cả 3 dataset
- Tổng hợp kết quả phần mở rộng
- Viết phần bài toán và phương pháp trong báo cáo
- Viết phần thực nghiệm và đóng góp mở rộng
- Viết phần hạn chế
- **Mốc:** có bản nháp báo cáo hoàn chỉnh trước 25/11

### Tuần 10: 26/11 – 1/12

- Xử lý phần việc bị trễ (buffer)
- Rà soát lại toàn bộ báo cáo
- Làm slide bảo vệ
- Luyện tập trình bày
- **Mốc:** nộp báo cáo cuối cùng trước 1/12

## Mã nguồn HashUQ

Implementation of paper: Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing. We implement our model based on [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch).

### Environment

Current setup (Python 3.11, torch 2.x, CUDA 11.8):

```
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python check_env.py --dataset imagenet   # checks packages, GPU, AlexNet weights, dataset files
```

Original environment from the paper:

```
torch==1.8.1+cu111
lap==0.4.0
matplotlib==3.4.2
torchvision==0.9.1+cu111
scipy==1.6.3
numpy==1.19.5
tqdm==4.61.0
pandas==1.2.4
Flask==3.0.0
Pillow==10.1.0
scikit_learn==1.3.1
```

### Dataset Preparation
Please find the instructions to setup datasets in [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch). Notice that the specfic split we use for NUS-WIDE dataset is NUS-WIDE-21. 
After downloading the dataset, please unzip the dataset at
```
../image_hashing_data/DATA_SET
```

where DATA_SET represent the specfic dataset we want to test, and need to be replaced by ``imagenet``, ``coco`` or ``nuswide_21``.

We include our dataset split in ```./data/```. The split is constructed by randomly select 10% of the data samples in the original training split to do the validation and model selection using ```Devide_TrainingSet_ImageNet```, ```Devide_TrainingSet_MSCOCO``` and ```Devide_TrainingSet_NUSWIDE```. Please move the txt files to the root folder of each dataset. 


### Training
To train our model with ``Center-Target'' construction on ImageNet, use the command:
```
python CSQ_HashUQ.py --grad_est cf --no-pairwise --no-tqdm --dataset imagenet --KL_regularization 1 --val --rt_st hamuct --sample_method MCD --sample 100
```

You can replace the argument ``imagenet`` to ``coco`` or ``nuswide_21`` to train and check the performance on other dataset. 

Extra arguments (both `CSQ_HashUQ.py` and `GreedyHash_HashUQ.py`):

| Argument | Default | Meaning |
|---|---|---|
| `--bit 16 32 64` | `16` | Hash lengths to train, one after another |
| `--seed N` | `0` | Random seed (also part of the save folder name, so runs with different seeds don't overwrite each other) |
| `--net NAME` | `alexnet` | `alexnet`, `resnet18`, `resnet34`, `resnet50`, ... (ResNet has no Dropout layer, so MC Dropout gives identical samples) |
| `--epoch N` | `100` | Training epochs (validation runs every 10 epochs for CSQ, every 100 for GreedyHash) |
| `--num_workers N` | `4` | DataLoader workers (use 0–4 on Windows/laptops) |
| `--uct_quant_levels 2 4` | `2 4` | HashUQ-Bin quantization levels evaluated together with HashUQ |

With `--rt_st hamuct`, each evaluation prints and saves Hamming, HashUQ and HashUQ-Bin mAP to
`saved_model/<run>/results_epoch_<N>.json`; the best-checkpoint evaluation is saved as `results_final.json`.
The database/query uncertainties are saved as `trn_uct.npy` / `tst_uct.npy` next to the hash codes.

Model will be automaticly saved at 
```
./saved_model/
```

### Testing
To test our model with ``Center-Target'' construction on ImageNet after trai, use the command:
```
python CSQ_HashUQ.py --grad_est cf --no-pairwise --no-tqdm --dataset imagenet --KL_regularization 1 --val --rt_st hamuct --sample_method MCD --sample 100 --no-train
```