def predict_churn(customer_data, model_info):
    """
    Dự đoán churn cho khách hàng
    
    Parameters:
        customer_data: Dictionary hoặc DataFrame với thông tin khách hàng
        model_info: Dictionary chứa model đã train
    
    Returns:
        result: Dictionary với prediction, probability, risk_level
    """
    try:
        # Chuyển input thành DataFrame
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Feature engineering
        customer_df_eng = feature_engineering(customer_df)
        
        # Lấy các features cần thiết (19 features được chọn)
        X_new = customer_df_eng[model_info['features']]
        
        # Predict với error handling
        model = model_info['model_object']
        
        # Fix sklearn compatibility issue
        # Ensure model estimators have monotonic_cst attribute
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if not hasattr(estimator, 'monotonic_cst'):
                    estimator.monotonic_cst = None
        
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
        
    except AttributeError as e:
        st.error(f"⚠️ Lỗi tương thích model: {str(e)}")
        st.warning("Đang thử phương pháp dự đoán thay thế...")
        
        # Fallback: Try to reconstruct model prediction manually
        try:
            customer_df = pd.DataFrame([customer_data]) if isinstance(customer_data, dict) else customer_data.copy()
            customer_df_eng = feature_engineering(customer_df)
            X_new = customer_df_eng[model_info['features']]
            
            # Manual prediction using tree voting
            model = model_info['model_object']
            predictions = []
            for tree in model.estimators_:
                # Patch missing attribute
                if not hasattr(tree, 'monotonic_cst'):
                    tree.monotonic_cst = None
                pred = tree.predict(X_new)[0]
                predictions.append(pred)
            
            prediction = int(np.round(np.mean(predictions)))
            probability = np.mean(predictions)
            
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
            
            st.success("✅ Dự đoán thành công bằng phương pháp thay thế!")
            return result
            
        except Exception as fallback_error:
            st.error(f"❌ Không thể thực hiện dự đoán: {str(fallback_error)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình dự đoán: {str(e)}")
        st.info("Vui lòng kiểm tra lại thông tin đầu vào hoặc liên hệ support.")
        return None