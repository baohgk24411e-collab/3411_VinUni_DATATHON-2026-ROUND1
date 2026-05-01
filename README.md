# VinUni Datathon Round 1 - Revenue & COGS Forecasting

## 📋 Tổng Quan Dự Án

Dự án này là một **pipeline phân tích và dự báo doanh thu (Revenue) và chi phí giá vốn (COGS)** cho [VinUni Datathon Round 1](). Dự án sử dụng kết hợp ba phương pháp phân tích:

1. **Phân tích Tương Quan & Hồi Quy Tuyến Tính** - Hiểu mối quan hệ giữa các biến
2. **Time Series Forecasting (SARIMA)** - Dự báo chuỗi thời gian
3. **Deep Learning & Machine Learning Ensemble** - Kết hợp XGBoost, Random Forest, LSTM và Temporal Fusion Transformer

---

## 📁 Cấu Trúc File

```
.
├── 3411_VinUni_DATATHON_ROUND1_EDA.ipynb                 # Phân tích dữ liệu sơ bộ (EDA)
├── 3411_VinUni_DATATHON_ROUND1_MACHINE___DEEP_LEARNING.ipynb  # Pipeline ML/DL chính
├── submission.csv                                         # File kết quả dự báo
├── Dashboard.pbix                                         # File dashboard
└── README.md                                              # File này
```
Vì file Dashboard vượt quá 25MB nên nhóm xin phép upload bằng đường dẫn drive (chung thư mục với báo cáo.pdf và submission.csv)
Link: https://drive.google.com/file/d/1H3vpRLbMipnX7K2_MRRWv7G0MhsC1gJ3/view?usp=drive_link
---

## 🔍 File 1: EDA (Exploratory Data Analysis)

**File:** `3411_VinUni_DATATHON_ROUND1_EDA.ipynb`

### Mục đích:
- Khám phá cấu trúc dữ liệu
- Phát hiện giá trị thiếu (missing values)
- Phân tích thống kê mô tả (descriptive statistics)
- Hình dung xu hướng dữ liệu

### Công cụ chính:
- `pandas` - Xử lý dữ liệu
- `numpy` - Tính toán số học
- `matplotlib` & `seaborn` - Hình dung dữ liệu
- `statsmodels` - Phân tích thống kê

---

## 🤖 File 2: Machine Learning & Deep Learning Pipeline

**File:** `3411_VinUni_DATATHON_ROUND1_MACHINE___DEEP_LEARNING.ipynb`

Pipeline chính được chia thành **3 giai đoạn**:

### **GIAI ĐOẠN 1: Correlation Analysis + Linear Regression**

#### 1.1 - Chuẩn bị dữ liệu
- Import thư viện cần thiết
- Đọc và merge dữ liệu CSV

#### 1.2 - Xử lý bảng (Table Processing)
- Làm sạch dữ liệu
- Xử lý giá trị thiếu
- Feature engineering cơ bản

#### 1.3 - Phân tích Tương Quan (Correlation Analysis)
- Tính ma trận tương quan (Correlation Matrix)
- Hình dung heatmap
- Nhận diện các biến có liên quan chặt chẽ

#### 1.4 - VIF (Variance Inflation Factor) Analysis
- Phát hiện **multicollinearity** (đa cộng tuyến)
- Loại bỏ các biến có VIF cao

#### 1.5 - So Sánh Mô Hình
- **Model A**: Sử dụng tất cả biến
- **Model B**: Sử dụng biến đã được lọc (VIF thấp)
- Chỉ số so sánh: **R² Hiệu Chỉnh (Adjusted R²)**
- Kết luận: Model B thường tốt hơn do giảm overfitting

---

### **GIAI ĐOẠN 2: Time Series Forecasting (SARIMA)**

#### 2.1 - ETL (Extract, Transform, Load)
- Tải dữ liệu
- Chuyển đổi định dạng thời gian
- Sắp xếp theo thứ tự chronological

#### 2.2 - Kiểm Tra Tính Dừng (Stationarity Testing)
- **ADF Test** (Augmented Dickey-Fuller): Kiểm tra căn đơn vị
- **KPSS Test**: Kiểm tra xu hướng
- **Kết quả**: Nếu chuỗi không dừng, ta phải sai phân (differencing)

#### 2.3 - Nhận Diện Tham Số p, q (ACF/PACF)
- **ACF Plot** (Autocorrelation Function): Xác định q (Moving Average)
- **PACF Plot** (Partial Autocorrelation Function): Xác định p (Autoregressive)
- Giúp chọn tham số cho ARIMA model

#### 2.4 - Lựa Chọn Mô Hình Bằng AIC (Grid Search)
- Grid search qua các giá trị (p, d, q) và (P, D, Q, s)
- Sử dụng **Akaike Information Criterion (AIC)** để đánh giá
- Chọn mô hình với AIC thấp nhất

#### 2.5 - Huấn Luyện SARIMAX (Seasonal ARIMA with Exogenous Variables)
- SARIMAX = SARIMA + các biến ngoại sinh (exogenous)
- Huấn luyện trên toàn bộ dữ liệu training

#### 2.6 - Chuẩn Đoán Phần Dư (Residual Diagnostics)
- Kiểm tra xem phần dư có phải white noise không
- Trực quan hóa ACF/PACF của phần dư
- Đảm bảo không còn tính tự tương quan

#### 2.7 - Dự Báo & So Sánh với Ridge Baseline
- Dự báo cho tập test
- So sánh với mô hình Ridge Regression (baseline)
- Tính MAE, RMSE, R²

#### 2.8 - Hình Dung Tổng Hợp
- Plot dữ liệu thực tế vs dự báo
- Trực quan hóa khoảng tin cậy (confidence intervals)

---

### **GIAI ĐOẠN 3: Deep Learning & Ensemble Pipeline**

#### 3.1 - Kiểm Tra Thư Viện
- Xác nhận cài đặt đầy đủ
- Import TensorFlow/PyTorch cho LSTM & TFT

#### 3.2 - ETL & Feature Engineering
- Chuẩn hóa dữ liệu (normalization/standardization)
- Tạo sliding windows cho time series deep learning
- Feature scaling cho các mô hình

#### 3.3 - XGBoost
- Gradient boosting model
- Tối ưu hóa hyperparameters
- Nhanh và chính xác

#### 3.4 - Random Forest
- Ensemble method dựa trên decision trees
- Giảm overfitting hiệu quả

#### 3.5 - LSTM (Long Short-Term Memory)
- Deep learning model cho time series
- Học các mối quan hệ dài hạn
- Sử dụng sliding windows

#### 3.6 - Temporal Fusion Transformer (TFT)
- State-of-the-art architecture cho time series
- Kết hợp attention mechanism
- Xử lý multiple variables đồng thời

#### 3.7 - SHAP Explainer (XGBoost)
- Giải thích mô hình XGBoost
- Xác định feature importance
- SHAP values cho interpretability

#### 3.8 - So Sánh & Chọn Mô Hình Winner
- Tính MAE, RMSE, MAPE cho tất cả mô hình
- So sánh thời gian huấn luyện
- Chọn mô hình tốt nhất dựa trên metrics

#### 3.9 - Final Submission
- Tạo submission file cuối cùng
- Format: Date, Revenue, COGS

---

## 📊 Dữ Liệu Input

### Cấu trúc dữ liệu:
```
Date       | Revenue    | COGS       | [Các biến khác...]
-----------|------------|------------|-------------------
2023-01-01 | 1805746   | 1483780.46 | ...
2023-01-02 | 1924399   | 1581277.68 | ...
...        | ...        | ...        | ...
```

### Thống kê:
- **Số dòng**: 548 ngày
- **Khoảng thời gian**: Từ 2023-01-01 đến hết năm 2023 (hoặc khoảng đó)
- **Cột chính**: Date, Revenue, COGS
- **Định dạng Revenue/COGS**: Float (currency in VND hoặc tương tự)

---

## 📈 Kết Quả Dự Báo

**File output:** `submission.csv`

### Format:
```csv
Date,Revenue,COGS
2023-01-01,1805745.8,1483780.46
2023-01-02,1924398.9,1581277.68
...
```

### Chất lượng kết quả:
- Dự báo được thực hiện bởi **ensemble của nhiều mô hình**
- Kết hợp kỳ vọng từ SARIMA, XGBoost, Random Forest, LSTM, TFT
- Các dự báo được calibrated để phản ánh xu hướng thực tế

---

## 🛠️ Công Cụ & Thư Viện

### Data Processing
```
pandas, numpy
```

### Visualization
```
matplotlib, seaborn, plotly
```

### Statistical Analysis
```
statsmodels (SARIMA, ADF, KPSS, ACF/PACF)
scikit-learn (preprocessing, ensemble methods)
```

### Machine Learning
```
xgboost
scikit-learn (RandomForest)
```

### Deep Learning
```
tensorflow/keras (LSTM)
pytorch (Temporal Fusion Transformer)
```

### Interpretability
```
shap (SHAP Explainer)
```

---

## 🚀 Cách Chạy Code

### Yêu cầu:
- Python 3.8+
- Jupyter Notebook hoặc Google Colab
- Các thư viện được liệt kê phía trên

### Bước 1: Cài đặt thư viện
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn xgboost shap torch tensorflow plotly
```

### Bước 2: Chuẩn bị dữ liệu
- Upload file CSV input (sales.csv, hoặc tên gốc)
- Đặt trong cùng thư mục với notebook

### Bước 3: Chạy EDA notebook
```
Mở: 3411_VinUni_DATATHON_ROUND1_EDA.ipynb
Chạy từ trên xuống dưới (Ctrl+Enter mỗi cell)
```

### Bước 4: Chạy ML/DL pipeline
```
Mở: 3411_VinUni_DATATHON_ROUND1_MACHINE___DEEP_LEARNING.ipynb
Chạy theo thứ tự: GIAI ĐOẠN 1 → GIAI ĐOẠN 2 → GIAI ĐOẠN 3
```

### Bước 5: Lấy kết quả
```
Output: submission.csv
```

---

## 📊 Mô Hình & Metrics

| Mô Hình | Loại | Ưu Điểm | Nhược Điểm |
|---------|------|--------|-----------|
| **Linear Regression** | Baseline | Dễ hiểu, nhanh | Không bắt được non-linearity |
| **SARIMA** | Time Series | Tốt cho chuỗi tuần hoàn | Yêu cầu stationary |
| **XGBoost** | Boosting | Mạnh, interpretable | Tuning phức tạp |
| **Random Forest** | Ensemble | Robust | Chậm hơn XGBoost |
| **LSTM** | Deep Learning | Bắt mối quan hệ dài hạn | Yêu cầu dữ liệu nhiều |
| **TFT** | Attention-based | State-of-the-art | Phức tạp, slow |

### Metrics Đánh Giá:
- **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình
- **RMSE** (Root Mean Squared Error): Căn bậc hai lỗi bình phương trung bình
- **MAPE** (Mean Absolute Percentage Error): Lỗi phần trăm tuyệt đối trung bình
- **R²**: Hệ số xác định (bao nhiêu phương sai được giải thích)

---

## 🔑 Những Khám Phá Chính

1. **Multicollinearity**: Một số biến có tương quan cao, VIF analysis giúp loại bỏ
2. **Seasonality**: Dữ liệu có tính mùa vụ rõ rệt → SARIMA thích hợp
3. **Non-linearity**: Mối quan hệ phức tạp → Deep learning models tốt hơn
4. **Ensemble Power**: Kết hợp nhiều mô hình cho dự báo tốt hơn

---

## 📝 Ghi Chú

- Pipeline được thiết kế để **có thể tái sử dụng** cho dữ liệu mới
- Tất cả hyperparameters đều có thể điều chỉnh
- Code có comment chi tiết để dễ theo dõi
- Phù hợp chạy trên Google Colab (có hỗ trợ GPU/TPU)


