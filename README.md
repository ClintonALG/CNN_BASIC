\# Nhận diện biển báo giao thông với CNN



Phân loại biển báo giao thông sử dụng CNN cơ bản trên dataset GTSRB (43 lớp).



\## Cấu trúc thư mục



```

LAB01/

├── config.py          # Cấu hình tham số

├── data\_loader.py     # Load và tiền xử lý dữ liệu

├── model.py           # Kiến trúc CNN

├── train.py           # Huấn luyện mô hình

├── evaluate.py        # Đánh giá trên test set

├── predict.py         # Dự đoán ảnh đơn lẻ

├── utils.py           # Hàm tiện ích

├── visualize.py       # Trực quan hóa dữ liệu

├── Data/

│   ├── Train.csv      # Nhãn huấn luyện

│   ├── Test.csv       # Nhãn kiểm tra

│   ├── Meta.csv       # Metadata các lớp

│   ├── Train/         # Ảnh huấn luyện (43 thư mục con 0/..42/)

│   ├── Test/          # Ảnh kiểm tra

│   └── Meta/          # Ảnh biểu tượng các lớp

└── saved\_models/      # Lưu model sau khi train

```



\## Yêu cầu



\- Python 3.x

\- PyTorch

\- OpenCV (`cv2`)

\- scikit-learn

\- pandas

\- matplotlib

\- numpy



\### 2. Huấn luyện



```bash

python train.py

```



\### 3. Đánh giá



```bash

python evaluate.py

```



\### 4. Dự đoán ảnh đơn lẻ



```bash

python predict.py path/to/image.jpg

```



\## Kiến trúc mô hình



```

Conv2d(3, 32) → ReLU → MaxPool(2)

Conv2d(32, 64) → ReLU → MaxPool(2)

Conv2d(64, 128) → ReLU → MaxPool(2)

FC(128\*4\*4, 256) → ReLU → Dropout(0.5)

FC(256, 43)

```



\## Tham số



\- Ảnh đầu vào: 32x32x3

\- Batch size: 64

\- Epochs: 20

\- Learning rate: 0.001

\- Optimizer: Adam

\- Loss: CrossEntropyLoss

