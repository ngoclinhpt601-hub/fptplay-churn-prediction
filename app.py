# FPTPlay Churn Prediction Web Application
# =========================================
# Ứng dụng web dự đoán khả năng rời bỏ dịch vụ (churn) của khách hàng FPTPlay
# sử dụng Machine Learning model (Random Forest).
#
# Công nghệ:
# - Streamlit: Framework web application
# - scikit-learn: Machine Learning model
# - plotly: Interactive visualizations
#
# Tác giả: Data Science Team
# Phiên bản: 1.0
# Ngày: 2026-03-31

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FPTPlay Churn Prediction",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .churn-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def feature_engineering(df):
    """
    Tạo các engineered features từ raw input data
    
    Input:
        df: DataFrame với các columns cơ bản (hours_m1-m6, trend_slope_abs, ...)
    
    Output:
        df_eng: DataFrame với 46 features (bao gồm engineered features)
    """
    df_eng = df.copy()
    
    # L1M features (Last 1 month)
    df_eng['AVG_L1M_HOURS'] = df['hours_m1']
    
    # L3M features (Last 3 months)
    hours_l3m = df[['hours_m1', 'hours_m2', 'hours_m3']]
    df_eng['SUM_L3M_HOURS'] = hours_l3m.sum(axis=1)
    df_eng['AVG_L3M_HOURS'] = hours_l3m.mean(axis=1)
    df_eng['MAX_L3M_HOURS'] = hours_l3m.max(axis=1)
    df_eng['MIN_L3M_HOURS'] = hours_l3m.min(axis=1)
    df_eng['STDDEV_L3M_HOURS'] = hours_l3m.std(axis=1)
    df_eng['MEDIAN_L3M_HOURS'] = hours_l3m.median(axis=1)
    
    # L6M features (Last 6 months)
    hours_l6m = df[['hours_m1', 'hours_m2', 'hours_m3', 'hours_m4', 'hours_m5', 'hours_m6']]
    df_eng['SUM_L6M_HOURS'] = hours_l6m.sum(axis=1)
    df_eng['AVG_L6M_HOURS'] = hours_l6m.mean(axis=1)
    df_eng['MAX_L6M_HOURS'] = hours_l6m.max(axis=1)
    df_eng['MIN_L6M_HOURS'] = hours_l6m.min(axis=1)
    df_eng['STDDEV_L6M_HOURS'] = hours_l6m.std(axis=1)
    df_eng['MEDIAN_L6M_HOURS'] = hours_l6m.median(axis=1)
    
    # Growth rates
    df_eng['GROWTH_RATE_L1M_VS_L3M'] = (
        (df['hours_m1'] - df_eng['AVG_L3M_HOURS']) / (df_eng['AVG_L3M_HOURS'] + 1e-6) * 100
    ).fillna(0)
    df_eng['GROWTH_RATE_L3M_VS_L6M'] = (
        (df_eng['AVG_L3M_HOURS'] - df_eng['AVG_L6M_HOURS']) / (df_eng['AVG_L6M_HOURS'] + 1e-6) * 100
    ).fillna(0)
    
    # Trend features - dự đoán % drop
    df_eng['PREDICTED_VIEWING_DROP_PCT'] = (
        (1 - df['predicted_next'] / (df_eng['AVG_L6M_HOURS'] + 1e-6)) * 100
    ).fillna(0)
    
    # Volatility (hệ số biến động)
    df_eng['CV_L3M_HOURS'] = (df_eng['STDDEV_L3M_HOURS'] / (df_eng['AVG_L3M_HOURS'] + 1e-6)).fillna(0)
    
    # Demographics - One-hot encoding
    df_eng['DEVICE_MOBILE'] = (df['device_type'] == 'mobile').astype(int)
    df_eng['DEVICE_TV'] = (df['device_type'] == 'tv').astype(int)
    df_eng['DEVICE_WEB'] = (df['device_type'] == 'web').astype(int)
    df_eng['PLAN_BASIC'] = (df['plan_type'] == 'basic').astype(int)
    df_eng['PLAN_STANDARD'] = (df['plan_type'] == 'standard').astype(int)
    df_eng['PLAN_PREMIUM'] = (df['plan_type'] == 'premium').astype(int)
    df_eng['REGION_NORTH'] = (df['region'] == 'north').astype(int)
    df_eng['REGION_CENTRAL'] = (df['region'] == 'central').astype(int)
    df_eng['REGION_SOUTH'] = (df['region'] == 'south').astype(int)
    
    # Interaction features - điểm giá trị cao
    df_eng['HIGH_VALUE_SCORE'] = (
        (df['tenure_months'] / 60.0) + df_eng['PLAN_PREMIUM'] + 
        (hours_l6m > 0).sum(axis=1) / 6.0
    ) / 3.0
    
    # Xử lý inf và NaN
    df_eng = df_eng.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df_eng

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """
    Load trained model từ file .pkl
    
    Returns:
        model_info: Dictionary chứa model, features, scaler, metadata
    """
    try:
        model_path = '/Workspace/Users/ngoclinhpt601@gmail.com/Mini_project (1)/models/best_model_random_forest.pkl'
        model_info = joblib.load(model_path)
        return model_info
    except Exception as e:
        st.error(f"❌ Lỗi khi load model: {str(e)}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_churn(customer_data, model_info):
    """
    Dự đoán churn cho khách hàng
    
    Parameters:
        customer_data: Dictionary hoặc DataFrame với thông tin khách hàng
        model_info: Dictionary chứa model đã train
    
    Returns:
        result: Dictionary với prediction, probability, risk_level
    """
    # Chuyển input thành DataFrame
    if isinstance(customer_data, dict):
        customer_df = pd.DataFrame([customer_data])
    else:
        customer_df = customer_data.copy()
    
    # Feature engineering
    customer_df_eng = feature_engineering(customer_df)
    
    # Lấy các features cần thiết (19 features được chọn)
    X_new = customer_df_eng[model_info['features']]
    
    # Predict
    model = model_info['model_object']
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0, 1]  # Probability of churn
    
    # Risk level classification
    if probability > 0.7:
        risk_level = 'HIGH'
        risk_color = '#c62828'
    elif probability > 0.3:
        risk_level = 'MEDIUM'
        risk_color = '#f57c00'
    else:
        risk_level = 'LOW'
        risk_color = '#2e7d32'
    
    result = {
        'churn_prediction': 'YES' if prediction == 1 else 'NO',
        'churn_probability': float(probability),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'prediction_numeric': int(prediction)
    }
    
    return result

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_gauge_chart(probability):
    """Tạo gauge chart hiển thị xác suất churn"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Xác Suất Churn (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_viewing_trend_chart(hours_data):
    """Tạo line chart hiển thị xu hướng giờ xem 6 tháng"""
    months = ['M6', 'M5', 'M4', 'M3', 'M2', 'M1']
    hours = [hours_data[f'hours_m{i}'] for i in range(6, 0, -1)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=hours,
        mode='lines+markers',
        name='Giờ Xem',
        line=dict(color='#2196f3', width=3),
        marker=dict(size=10, color='#1976d2')
    ))
    
    # Add average line
    avg_hours = np.mean(hours)
    fig.add_hline(y=avg_hours, line_dash="dash", line_color="red", 
                  annotation_text=f"Trung bình: {avg_hours:.1f}h")
    
    fig.update_layout(
        title="Xu Hướng Giờ Xem 6 Tháng Gần Nhất",
        xaxis_title="Tháng",
        yaxis_title="Giờ Xem",
        height=300,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_feature_contribution_chart():
    """Tạo bar chart hiển thị độ quan trọng của các features"""
    features = [
        'PREDICTED_VIEWING_DROP_PCT',
        'trend_slope_abs',
        'GROWTH_RATE_L3M_VS_L6M',
        'GROWTH_RATE_L1M_VS_L3M',
        'CV_L3M_HOURS'
    ]
    importance = [54.35, 15.08, 14.70, 13.27, 1.23]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Importance (%)")
        ),
        text=[f"{val:.1f}%" for val in importance],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Top 5 Yếu Tố Ảnh Hưởng Churn",
        xaxis_title="Importance (%)",
        yaxis_title="",
        height=300,
        template='plotly_white'
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Hàm main của ứng dụng Streamlit"""
    
    # ========== HEADER ==========
    st.markdown('<div class="main-header">📺 FPTPlay Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dự Đoán Khả Năng Khách Hàng Rời Bỏ Dịch Vụ</div>', unsafe_allow_html=True)
    
    # Load model
    model_info = load_model()
    
    if model_info is None:
        st.error("❌ Không thể load model. Vui lòng kiểm tra đường dẫn file model.")
        return
    
    # Display model info
    with st.expander("ℹ️ Thông Tin Mô Hình", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", model_info['model_name'])
        with col2:
            st.metric("Accuracy", f"{model_info['test_accuracy']:.2%}")
        with col3:
            st.metric("F1-Score", f"{model_info['f1_score']:.2%}")
        with col4:
            st.metric("ROC-AUC", f"{model_info['roc_auc']:.2%}")
    
    st.markdown("---")
    
    # ========== SIDEBAR - INPUT FORM ==========
    st.sidebar.header("📝 Nhập Thông Tin Khách Hàng")
    st.sidebar.markdown("Vui lòng điền đầy đủ thông tin dưới đây:")
    
    # Customer ID
    customer_id = st.sidebar.text_input(
        "🆔 Mã Khách Hàng",
        value="CUST_999999",
        help="Mã định danh khách hàng"
    )
    
    st.sidebar.markdown("### 📊 Lịch Sử Xem (Giờ/Tháng)")
    st.sidebar.caption("Nhập số giờ xem phim của khách hàng trong 6 tháng gần nhất")
    
    # Viewing hours (6 months)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        hours_m1 = st.number_input("Tháng 1 (gần nhất)", min_value=0.0, max_value=500.0, value=5.3, step=0.1)
        hours_m3 = st.number_input("Tháng 3", min_value=0.0, max_value=500.0, value=15.2, step=0.1)
        hours_m5 = st.number_input("Tháng 5", min_value=0.0, max_value=500.0, value=22.3, step=0.1)
    
    with col2:
        hours_m2 = st.number_input("Tháng 2", min_value=0.0, max_value=500.0, value=10.8, step=0.1)
        hours_m4 = st.number_input("Tháng 4", min_value=0.0, max_value=500.0, value=18.7, step=0.1)
        hours_m6 = st.number_input("Tháng 6 (xa nhất)", min_value=0.0, max_value=500.0, value=25.5, step=0.1)
    
    st.sidebar.markdown("### 📈 Xu Hướng & Dự Đoán")
    
    # Trend and prediction
    trend_slope_abs = st.sidebar.number_input(
        "Độ dốc xu hướng (giờ/tháng)",
        min_value=-50.0,
        max_value=50.0,
        value=-3.5,
        step=0.1,
        help="Tốc độ thay đổi giờ xem theo tháng. Giá trị âm = giảm, dương = tăng"
    )
    
    predicted_next = st.sidebar.number_input(
        "Giờ xem dự kiến tháng tiếp theo",
        min_value=0.0,
        max_value=500.0,
        value=2.1,
        step=0.1,
        help="Dự đoán số giờ xem tháng tiếp theo dựa trên trend"
    )
    
    st.sidebar.markdown("### 👤 Thông Tin Khách Hàng")
    
    # Demographics
    tenure_months = st.sidebar.slider(
        "Thời gian sử dụng dịch vụ (tháng)",
        min_value=1,
        max_value=60,
        value=24,
        help="Số tháng khách hàng đã sử dụng dịch vụ FPTPlay"
    )
    
    device_type = st.sidebar.selectbox(
        "Thiết bị chính",
        options=['mobile', 'tv', 'web'],
        index=0,
        help="Thiết bị khách hàng thường xem phim"
    )
    
    plan_type = st.sidebar.selectbox(
        "Gói dịch vụ",
        options=['basic', 'standard', 'premium'],
        index=0,
        help="Loại gói dịch vụ khách hàng đang sử dụng"
    )
    
    region = st.sidebar.selectbox(
        "Khu vực",
        options=['north', 'central', 'south'],
        index=2,
        help="Khu vực địa lý của khách hàng"
    )
    
    is_promo_subscriber = st.sidebar.checkbox(
        "Đăng ký qua chương trình khuyến mãi",
        value=False,
        help="Khách hàng có đăng ký qua khuyến mãi không?"
    )
    
    st.sidebar.markdown("---")
    
    # Predict button
    predict_button = st.sidebar.button(
        "🔮 DỰ ĐOÁN CHURN",
        type="primary",
        use_container_width=True
    )
    
    # ========== MAIN CONTENT ==========
    
    if predict_button:
        # Prepare customer data
        customer_data = {
            'customer_id': customer_id,
            'hours_m1': hours_m1,
            'hours_m2': hours_m2,
            'hours_m3': hours_m3,
            'hours_m4': hours_m4,
            'hours_m5': hours_m5,
            'hours_m6': hours_m6,
            'trend_slope_abs': trend_slope_abs,
            'predicted_next': predicted_next,
            'tenure_months': tenure_months,
            'device_type': device_type,
            'plan_type': plan_type,
            'region': region,
            'is_promo_subscriber': int(is_promo_subscriber)
        }
        
        # Predict
        with st.spinner('🤖 Đang phân tích dữ liệu và dự đoán...'):
            result = predict_churn(customer_data, model_info)
        
        # ===== DISPLAY RESULTS =====
        
        # Main prediction result
        if result['churn_prediction'] == 'YES':
            st.markdown(
                f'<div class="prediction-box churn-yes">'
                f'⚠️ CẢNH BÁO: Khách hàng CÓ NGUY CƠ CHURN!'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="warning-box">'
                '<strong>⚡ Khuyến nghị:</strong> Cần can thiệp ngay để giữ chân khách hàng. '
                'Xem xét các biện pháp: ưu đãi đặc biệt, hỗ trợ kỹ thuật, tư vấn gói dịch vụ phù hợp.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box churn-no">'
                f'✅ AN TOÀN: Khách hàng KHÔNG CHURN'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="info-box">'
                '<strong>💡 Khuyến nghị:</strong> Khách hàng đang hài lòng với dịch vụ. '
                'Tiếp tục duy trì chất lượng và có thể xem xét upsell gói cao hơn.'
                '</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Detailed metrics
        st.subheader("📊 Phân Tích Chi Tiết")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Xác Suất Churn",
                value=f"{result['churn_probability']:.1%}",
                delta=f"{result['churn_probability'] - 0.5:.1%}" if result['churn_probability'] != 0.5 else "0%",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Mức Độ Rủi Ro",
                value=result['risk_level']
            )
        
        with col3:
            avg_hours = np.mean([hours_m1, hours_m2, hours_m3, hours_m4, hours_m5, hours_m6])
            st.metric(
                label="Giờ Xem TB (6 tháng)",
                value=f"{avg_hours:.1f}h"
            )
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📈 Biểu Đồ Trực Quan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart
            gauge_fig = create_gauge_chart(result['churn_probability'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Viewing trend chart
            trend_fig = create_viewing_trend_chart(customer_data)
            st.plotly_chart(trend_fig, use_container_width=True)
        
        # Feature importance
        st.markdown("---")
        st.subheader("🔍 Yếu Tố Ảnh Hưởng")
        importance_fig = create_feature_contribution_chart()
        st.plotly_chart(importance_fig, use_container_width=True)
        
        # Download button for results
        st.markdown("---")
        result_df = pd.DataFrame([{
            'Customer ID': customer_id,
            'Churn Prediction': result['churn_prediction'],
            'Churn Probability': f"{result['churn_probability']:.2%}",
            'Risk Level': result['risk_level'],
            'Avg Hours (6M)': f"{avg_hours:.1f}",
            'Trend Slope': trend_slope_abs,
            'Tenure (months)': tenure_months,
            'Device': device_type,
            'Plan': plan_type,
            'Region': region,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }])
        
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải Kết Quả (CSV)",
            data=csv,
            file_name=f"churn_prediction_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome screen
        st.info(
            "👈 **Hướng dẫn sử dụng:**\n\n"
            "1. Điền thông tin khách hàng vào form bên trái\n"
            "2. Nhập lịch sử giờ xem 6 tháng gần nhất\n"
            "3. Điền các thông tin bổ sung (xu hướng, demographics)\n"
            "4. Nhấn nút **'DỰ ĐOÁN CHURN'** để xem kết quả\n\n"
            "Hệ thống sẽ phân tích dữ liệu và đưa ra dự đoán về khả năng khách hàng rời bỏ dịch vụ."
        )
        
        # Model performance summary
        st.subheader("🎯 Hiệu Suất Mô Hình")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Accuracy", "100%", help="Độ chính xác tổng thể")
        with perf_col2:
            st.metric("Precision", "100%", help="Tỷ lệ dự đoán churn đúng")
        with perf_col3:
            st.metric("Recall", "100%", help="Tỷ lệ phát hiện được churn")
        with perf_col4:
            st.metric("F1-Score", "100%", help="Điểm trung bình điều hòa")
        
        st.markdown("---")
        
        # Feature importance chart
        st.subheader("📊 Top Yếu Tố Ảnh Hưởng Churn")
        importance_fig = create_feature_contribution_chart()
        st.plotly_chart(importance_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 20px;">'
        '📺 FPTPlay Churn Prediction System v1.0 | '
        'Powered by Random Forest ML Model | '
        'Data Science Team © 2026'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
