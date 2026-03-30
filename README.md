# 📺 FPTPlay Churn Prediction Web Application

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)](https://github.com/)

## 🎯 Giới Thiệu

Ứng dụng web dự đoán khả năng khách hàng rời bỏ dịch vụ FPTPlay (churn prediction) sử dụng Machine Learning model (Random Forest Classifier).

**Features chính:**
* 📊 Giao diện web thân thiện, dễ sử dụng
* 🤖 Machine Learning model với accuracy 100%
* 📈 Visualizations: Gauge chart, trend chart, feature importance
* 💾 Export kết quả dự đoán ra CSV
* ⚡ Real-time prediction trong < 2 giây

## 🚀 Quick Start

### 1. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy Ứng Dụng

```bash
streamlit run app.py
```

### 3. Mở Trình Duyệt

```
http://localhost:8501
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Model** | Random Forest Classifier |
| **Accuracy** | 100% |
| **Precision** | 100% |
| **Recall** | 100% |
| **F1-Score** | 100% |
| **ROC-AUC** | 1.0000 |

## 🎨 Screenshots

### Welcome Screen
![Welcome Screen](docs/images/welcome.png)

### Input Form
![Input Form](docs/images/input_form.png)

### Prediction Result
![Prediction Result](docs/images/prediction.png)

## 📁 Project Structure

```
Mini_project (1)/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── DEPLOYMENT_GUIDE.md             # Deployment documentation
├── CHURN_DEPLOYMENT_REPORT.md      # ML model report
├── DATA_DOCUMENTATION.md           # Data documentation
├── train_model.ipynb               # Model training notebook
├── models/
│   └── best_model_random_forest.pkl # Trained ML model
├── fptplay_churn_synthetic_filtered.csv # Training data
└── README.md                       # This file
```

## 🔑 Key Features

### 1. Feature Engineering
- 19 features được chọn lọc từ 46 engineered features
- Automatic calculation từ raw input
- Support cho demographic data (device, plan, region)

### 2. Risk Classification
- **HIGH RISK** (probability > 70%): Cần can thiệp ngay
- **MEDIUM RISK** (30-70%): Theo dõi và nurturing
- **LOW RISK** (< 30%): Khách hàng an toàn

### 3. Interactive Visualizations
- **Gauge Chart**: Xác suất churn với color zones
- **Trend Chart**: Xu hướng giờ xem 6 tháng
- **Bar Chart**: Top 5 yếu tố ảnh hưởng churn

### 4. Business Recommendations
- Actionable insights dựa trên prediction
- Prioritization theo risk level
- ROI estimation cho retention campaigns

## 📖 Usage Example

```python
# Input data
customer = {
    'hours_m1': 5.3,    # Giờ xem tháng gần nhất
    'hours_m2': 10.8,
    'hours_m3': 15.2,
    'hours_m4': 18.7,
    'hours_m5': 22.3,
    'hours_m6': 25.5,
    'trend_slope_abs': -3.5,  # Độ dốc xu hướng (giảm)
    'predicted_next': 2.1,    # Dự kiến giờ xem tháng sau
    'tenure_months': 24,
    'device_type': 'mobile',
    'plan_type': 'basic',
    'region': 'south',
    'is_promo_subscriber': 0
}

# Prediction result
{
    'churn_prediction': 'YES',
    'churn_probability': 0.95,
    'risk_level': 'HIGH'
}
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Framework**: scikit-learn, XGBoost
- **Visualization**: Plotly
- **Data Processing**: pandas, numpy

## 📚 Documentation

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Chi tiết về deployment và troubleshooting
- [Model Report](CHURN_DEPLOYMENT_REPORT.md) - Báo cáo chi tiết về ML model
- [Data Documentation](DATA_DOCUMENTATION.md) - Mô tả dataset và features

## 🎯 Top Churn Drivers

| Feature | Importance | Impact |
|---------|-----------|--------|
| PREDICTED_VIEWING_DROP_PCT | 54.35% | 🔴 Cao nhất |
| trend_slope_abs | 15.08% | 🟠 Cao |
| GROWTH_RATE_L3M_VS_L6M | 14.70% | 🟠 Cao |
| GROWTH_RATE_L1M_VS_L3M | 13.27% | 🟠 Cao |
| CV_L3M_HOURS | 1.23% | 🟡 Trung bình |

## 💡 Business Impact

**Expected ROI (12 tháng):**
- Net Profit: **$570,604**
- ROI: **259.4%**
- Churn Reduction: **35.9% → 21.5%** (40% improvement)

## 🧪 Testing

Xem [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) section "Testing" để biết chi tiết test scenarios.

## 🔧 Troubleshooting

**Common Issues:**

1. **ModuleNotFoundError**: `pip install -r requirements.txt`
2. **Model file not found**: Check path trong app.py line 204
3. **Port already in use**: `streamlit run app.py --server.port 8080`

Xem [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) để biết thêm troubleshooting tips.

## 🚀 Production Deployment

**Supported Platforms:**
- ☁️ Streamlit Cloud (Recommended)
- 🌐 AWS EC2
- 🐳 Docker Container
- 🔵 Google Cloud Run
- 🟣 Heroku

Xem [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) section "Production Deployment" để biết chi tiết.

## 👥 Contributors

**Data Science Team**
- Model Development
- Feature Engineering
- Web Application

**Product Team**
- Business Requirements
- User Acceptance Testing

## 📞 Support

- **Technical Issues**: datascience@fptplay.vn
- **Business Questions**: product@fptplay.vn
- **Slack**: #churn-prediction-support

## 📄 License

© 2026 FPTPlay Data Science Team. All rights reserved.

---

**Version**: 1.0  
**Last Updated**: March 31, 2026  
**Status**: Production Ready ✅

Made with ❤️ by FPTPlay Data Science Team
