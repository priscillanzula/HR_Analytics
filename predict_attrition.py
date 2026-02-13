"""
HR ATTRITION PREDICTION TOOL
=============================
Simple script to predict if an employee is at risk of leaving.

Usage:
    python predict_attrition.py

Author: Your Name
Date: January 2026
"""

import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the trained Random Forest model"""
    try:
        model = joblib.load('deployment/models/random_forest_smote.pkl')
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("❌ Error: Model file not found. Run the training notebook first.")
        return None

def get_employee_data():
    """
    Collect employee information from HR
    Returns a DataFrame with one employee's data
    """
    print("\n" + "="*50)
    print("ENTER EMPLOYEE INFORMATION")
    print("="*50)
    
    # Basic info
    age = int(input("Age: "))
    dept = input("Department (finance/hr/it/operations/sales/unknown): ").lower().strip()
    
    # Performance metrics
    satisfaction = float(input("Satisfaction Score (0.0 to 1.0): "))
    evaluation = float(input("Last Evaluation Score (0.0 to 1.0): "))
    
    # Workload
    num_projects = int(input("Number of Projects: "))
    monthly_hours = int(input("Average Monthly Hours: "))
    
    # Tenure
    years = int(input("Years at Company: "))
    
    # Calculate engineered features
    hours_per_project = monthly_hours / num_projects if num_projects > 0 else 0
    performance_ratio = evaluation / satisfaction if satisfaction > 0 else 0
    
    # Tenure category
    if years <= 2:
        tenure_category = 0  # New
    elif years <= 5:
        tenure_category = 1  # Mid
    else:
        tenure_category = 2  # Experienced
    
    # High risk flag
    median_hours_per_project = 40  # From your analysis
    median_satisfaction = 0.5  # From your analysis
    high_risk = 1 if (hours_per_project > median_hours_per_project and satisfaction < median_satisfaction) else 0
    
    # Create department dummy variables
    dept_hr = 1 if dept == 'hr' else 0
    dept_it = 1 if dept == 'it' else 0
    dept_operations = 1 if dept == 'operations' else 0
    dept_sales = 1 if dept == 'sales' else 0
    dept_unknown = 1 if dept == 'unknown' else 0
    
    # Create DataFrame with exact feature order from training
    employee_data = pd.DataFrame({
        'Age': [age],
        'SatisfactionScore': [satisfaction],
        'LastEvaluationScore': [evaluation],
        'NumProjects': [num_projects],
        'AvgMonthlyHours': [monthly_hours],
        'YearsAtCompany': [years],
        'HoursPerProject': [hours_per_project],
        'PerformanceRatio': [performance_ratio],
        'TenureCategory_encoded': [tenure_category],
        'High_Risk_Employee': [high_risk],
        'Department_hr': [dept_hr],
        'Department_it': [dept_it],
        'Department_operations': [dept_operations],
        'Department_sales': [dept_sales],
        'Department_unknown': [dept_unknown]
    })
    
    return employee_data

def predict_risk(model, employee_data):
    """
    Predict attrition risk for an employee
    Returns risk level and probability
    """
    # Get prediction probability
    risk_probability = model.predict_proba(employee_data)[0][1]
    
    # Get binary prediction
    will_leave = model.predict(employee_data)[0]
    
    # Determine risk level
    if risk_probability < 0.3:
        risk_level = "LOW RISK"
        color = "🟢"
    elif risk_probability < 0.6:
        risk_level = "MEDIUM RISK"
        color = "🟡"
    else:
        risk_level = "HIGH RISK"
        color = "🔴"
    
    return will_leave, risk_probability, risk_level, color

def display_results(will_leave, risk_probability, risk_level, color, employee_data):
    """Display prediction results to HR"""
    print("\n" + "="*50)
    print("ATTRITION RISK ASSESSMENT")
    print("="*50)
    print(f"\n{color} RISK LEVEL: {risk_level}")
    print(f"   Probability of Leaving: {risk_probability*100:.1f}%")
    print(f"   Prediction: {'WILL LEAVE' if will_leave == 1 else 'WILL STAY'}")
    
    print("\n" + "-"*50)
    print("KEY METRICS:")
    print("-"*50)
    print(f"  Satisfaction Score: {employee_data['SatisfactionScore'].values[0]:.2f}")
    print(f"  Hours Per Project: {employee_data['HoursPerProject'].values[0]:.1f}")
    print(f"  Number of Projects: {employee_data['NumProjects'].values[0]}")
    print(f"  High Risk Flag: {'YES' if employee_data['High_Risk_Employee'].values[0] == 1 else 'NO'}")
    
    print("\n" + "="*50)
    print("RECOMMENDED ACTIONS:")
    print("="*50)
    
    if risk_probability >= 0.6:
        print("🔴 URGENT - Schedule 1-on-1 meeting immediately")
        print("   • Review workload and redistribute projects")
        print("   • Discuss career development opportunities")
        print("   • Consider retention bonus or promotion")
    elif risk_probability >= 0.3:
        print("🟡 MODERATE - Monitor closely")
        print("   • Check in during next performance review")
        print("   • Ensure workload is manageable")
        print("   • Offer training or new challenges")
    else:
        print("🟢 LOW RISK - Continue standard engagement")
        print("   • Maintain current support level")
        print("   • Regular check-ins as scheduled")

def main():
    """Main function to run the prediction tool"""
    print("\n" + "="*60)
    print("         HR ATTRITION PREDICTION TOOL")
    print("="*60)
    print("Predict which employees are at risk of leaving\n")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    while True:
        # Get employee data
        employee_data = get_employee_data()
        
        # Make prediction
        will_leave, risk_prob, risk_level, color = predict_risk(model, employee_data)
        
        # Display results
        display_results(will_leave, risk_prob, risk_level, color, employee_data)
        
        # Ask if user wants to predict another employee
        print("\n" + "="*50)
        another = input("\nPredict another employee? (yes/no): ").lower().strip()
        if another not in ['yes', 'y']:
            print("\nThank you for using the HR Attrition Prediction Tool!")
            break

if __name__ == "__main__":
    main()
