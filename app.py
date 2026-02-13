"""
HR ATTRITION PREDICTION - WEB APP
==================================
Simple Streamlit interface for the HR Analytics model

Run with: streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import json

# Page config
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# API URL
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .risk-medium {
        background-color: #ffaa00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .risk-low {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 HR Attrition Risk Predictor</p>', unsafe_allow_html=True)
st.markdown("**Predict employee attrition risk using machine learning**")

# Check API connection
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool predicts which employees are at risk of leaving using:
    - **Random Forest** (60% weight)
    - **XGBoost** (40% weight)
    - **SMOTE-balanced training**
    - **Probability calibration**
    
    **Performance:**
    - 83% recall (catches 83% of leavers)
    - 71% F1-score
    - 87% ROC-AUC
    """)
    
    api_status = check_api()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.warning("Start API: `python start_api.py`")

# Main content tabs
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])

# TAB 1: Single Prediction
with tab1:
    st.header("Predict Single Employee Risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=18, max_value=70, value=35)
        department = st.selectbox(
            "Department",
            ["finance", "hr", "it", "operations", "sales", "unknown"]
        )
        years = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
    
    with col2:
        st.subheader("Performance Metrics")
        satisfaction = st.slider("Satisfaction Score", 0.0, 1.0, 0.5, 0.05)
        evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.05)
        num_projects = st.number_input("Number of Projects", min_value=1, max_value=20, value=5)
        monthly_hours = st.number_input("Avg Monthly Hours", min_value=80, max_value=350, value=180)
    
    if st.button("🔮 Predict Risk", type="primary"):
        if not api_status:
            st.error("API is offline. Start it with: `python start_api.py`")
        else:
            # Calculate engineered features
            hours_per_project = monthly_hours / num_projects
            performance_ratio = evaluation / satisfaction if satisfaction > 0 else 1.0
            
            # Tenure category
            if years <= 2:
                tenure_category = 0
            elif years <= 5:
                tenure_category = 1
            else:
                tenure_category = 2
            
            # High risk flag
            high_risk = 1 if (hours_per_project > 40 and satisfaction < 0.5) else 0
            
            # Department encoding
            dept_encoding = {
                'hr': [1, 0, 0, 0, 0],
                'it': [0, 1, 0, 0, 0],
                'operations': [0, 0, 1, 0, 0],
                'sales': [0, 0, 0, 1, 0],
                'unknown': [0, 0, 0, 0, 1],
                'finance': [0, 0, 0, 0, 0]
            }
            dept_enc = dept_encoding[department]
            
            # Create payload
            payload = {
                "Age": age,
                "SatisfactionScore": satisfaction,
                "LastEvaluationScore": evaluation,
                "NumProjects": num_projects,
                "AvgMonthlyHours": monthly_hours,
                "YearsAtCompany": years,
                "HoursPerProject": hours_per_project,
                "PerformanceRatio": performance_ratio,
                "TenureCategory_encoded": tenure_category,
                "Department_hr": dept_enc[0],
                "Department_it": dept_enc[1],
                "Department_operations": dept_enc[2],
                "Department_sales": dept_enc[3],
                "Department_unknown": dept_enc[4],
                "High_Risk_Employee": high_risk
            }
            
            # Make prediction
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    risk = result['risk_assessment']
                    impact = result['business_impact']
                    
                    # Display results
                    st.markdown("---")
                    st.header("Prediction Results")
                    
                    # Risk level
                    prob = risk['calibrated_probability']
                    if prob > 0.7:
                        st.markdown(f'<div class="risk-high">🔴 CRITICAL RISK: {prob*100:.1f}%</div>', unsafe_allow_html=True)
                    elif prob > 0.3:
                        st.markdown(f'<div class="risk-medium">🟡 MEDIUM RISK: {prob*100:.1f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low">🟢 LOW RISK: {prob*100:.1f}%</div>', unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Random Forest", f"{risk['random_forest_probability']*100:.1f}%")
                    with col2:
                        st.metric("XGBoost", f"{risk['xgboost_probability']*100:.1f}%")
                    with col3:
                        st.metric("Ensemble", f"{prob*100:.1f}%")
                    
                    # Business impact
                    st.subheader("💰 Business Impact")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Replacement Cost", impact['estimated_attrition_cost'])
                        st.metric("Intervention ROI", impact['intervention_roi'])
                    with col2:
                        st.metric("Priority Level", impact['priority_level'])
                        st.metric("Potential Savings", impact['expected_savings'])
                    
                    # Recommendations
                    st.subheader("📋 Recommended Actions")
                    for rec in result['recommendations']:
                        st.write(f"• {rec}")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 2: Batch Prediction
with tab2:
    st.header("Batch Prediction from CSV")
    
    st.info("""
    Upload a CSV file with these columns:
    - EmployeeID, Age, Department, SatisfactionScore
    - LastEvaluationScore, NumProjects, AvgMonthlyHours, YearsAtCompany
    """)
    
    # Download sample template
    sample_data = {
        'EmployeeID': ['EMP001', 'EMP002', 'EMP003'],
        'Age': [35, 28, 42],
        'Department': ['finance', 'hr', 'it'],
        'SatisfactionScore': [0.45, 0.85, 0.30],
        'LastEvaluationScore': [0.75, 0.80, 0.65],
        'NumProjects': [6, 4, 8],
        'AvgMonthlyHours': [220, 160, 280],
        'YearsAtCompany': [3, 2, 5]
    }
    sample_df = pd.DataFrame(sample_data)
    
    st.download_button(
        "📥 Download Sample Template",
        sample_df.to_csv(index=False),
        "employee_template.csv",
        "text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Employee CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} employees")
        st.dataframe(df.head())
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            if not api_status:
                st.error("API is offline. Start it with: `python start_api.py`")
            else:
                st.info("Processing employees...")
                results = []
                
                progress_bar = st.progress(0)
                for idx, row in df.iterrows():
                    # Prepare features
                    hours_per_project = row['AvgMonthlyHours'] / row['NumProjects'] if row['NumProjects'] > 0 else 0
                    performance_ratio = row['LastEvaluationScore'] / row['SatisfactionScore'] if row['SatisfactionScore'] > 0 else 1.0
                    tenure_category = 0 if row['YearsAtCompany'] <= 2 else (1 if row['YearsAtCompany'] <= 5 else 2)
                    dept = row['Department'].lower().strip()
                    
                    payload = {
                        "Age": int(row['Age']),
                        "SatisfactionScore": float(row['SatisfactionScore']),
                        "LastEvaluationScore": float(row['LastEvaluationScore']),
                        "NumProjects": int(row['NumProjects']),
                        "AvgMonthlyHours": int(row['AvgMonthlyHours']),
                        "YearsAtCompany": int(row['YearsAtCompany']),
                        "HoursPerProject": hours_per_project,
                        "PerformanceRatio": performance_ratio,
                        "TenureCategory_encoded": tenure_category,
                        "Department_hr": 1 if dept == 'hr' else 0,
                        "Department_it": 1 if dept == 'it' else 0,
                        "Department_operations": 1 if dept == 'operations' else 0,
                        "Department_sales": 1 if dept == 'sales' else 0,
                        "Department_unknown": 1 if dept == 'unknown' else 0,
                        "High_Risk_Employee": 0
                    }
                    
                    try:
                        response = requests.post(f"{API_URL}/predict", json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            results.append({
                                'EmployeeID': row['EmployeeID'],
                                'Risk_Level': result['risk_assessment']['risk_level'],
                                'Risk_%': round(result['risk_assessment']['calibrated_probability'] * 100, 1),
                                'Action': result['risk_assessment']['recommended_action'],
                                'Estimated_Cost': result['business_impact']['estimated_attrition_cost']
                            })
                    except:
                        pass
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                # Display results
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Risk_%', ascending=False)
                
                st.success(f"✅ Processed {len(results)} employees")
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Employees", len(results))
                with col2:
                    high_risk = len(results_df[results_df['Risk_%'] > 60])
                    st.metric("High Risk (>60%)", high_risk)
                with col3:
                    avg_risk = results_df['Risk_%'].mean()
                    st.metric("Average Risk", f"{avg_risk:.1f}%")
                
                # Results table
                st.subheader("Results (sorted by risk)")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                st.download_button(
                    "📥 Download Results",
                    results_df.to_csv(index=False),
                    "attrition_predictions.csv",
                    "text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with FastAPI + Streamlit | Random Forest + XGBoost Ensemble | 83% Recall</p>
</div>
""", unsafe_allow_html=True)
