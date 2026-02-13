"""
HR ATTRITION API CLIENT
Simple tool to interact with the FastAPI deployment
Usage: python api_client.py
"""

import requests
import json

API_URL = "http://localhost:8000"


def test_connection():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print("❌ API returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API")
        print("   Make sure the server is running: python start_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def get_model_details():
    """Get information about the models"""
    try:
        response = requests.get(f"{API_URL}/model-details")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting model details: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def predict_single_employee():
    """Interactive prediction for one employee"""
    print("\n" + "="*60)
    print("ENTER EMPLOYEE INFORMATION")
    print("="*60)

    try:
        # Get basic info
        age = int(input("\nAge: "))
        dept = input(
            "Department (hr/it/operations/sales/finance/unknown): ").lower().strip()

        # Performance metrics
        satisfaction = float(input("Satisfaction Score (0.0 to 1.0): "))
        evaluation = float(input("Last Evaluation Score (0.0 to 1.0): "))

        # Workload
        num_projects = int(input("Number of Projects: "))
        monthly_hours = int(input("Average Monthly Hours: "))
        years = int(input("Years at Company: "))

        # Calculate engineered features
        hours_per_project = monthly_hours / num_projects if num_projects > 0 else 0
        performance_ratio = evaluation / satisfaction if satisfaction > 0 else 1.0

        # Tenure category (0=New, 1=Mid, 2=Experienced)
        if years <= 2:
            tenure_category = 0
        elif years <= 5:
            tenure_category = 1
        else:
            tenure_category = 2

        # High risk flag
        median_hours_per_project = 40
        median_satisfaction = 0.5
        high_risk = 1 if (hours_per_project > median_hours_per_project and
                          satisfaction < median_satisfaction) else 0

        # Department encoding
        dept_mapping = {
            'hr': 'Department_hr',
            'it': 'Department_it',
            'operations': 'Department_operations',
            'sales': 'Department_sales',
            'finance': 'Department_finance',
            'unknown': 'Department_unknown'
        }

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
            "Department_hr": 1 if dept == 'hr' else 0,
            "Department_it": 1 if dept == 'it' else 0,
            "Department_operations": 1 if dept == 'operations' else 0,
            "Department_sales": 1 if dept == 'sales' else 0,
            "Department_unknown": 1 if dept == 'unknown' else 0,
            "High_Risk_Employee": high_risk
        }

        # Make API request
        print("\n⏳ Analyzing employee data...")
        response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            display_prediction(result)
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)

    except ValueError as e:
        print(f"❌ Invalid input: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def display_prediction(result):
    """Display prediction results nicely"""
    risk = result['risk_assessment']
    impact = result['business_impact']

    # Determine color emoji
    prob = risk['calibrated_probability']
    if prob > 0.7:
        emoji = "🔴"
    elif prob > 0.3:
        emoji = "🟡"
    else:
        emoji = "🟢"

    print("\n" + "="*60)
    print("ATTRITION RISK ASSESSMENT")
    print("="*60)

    print(f"\n{emoji} RISK LEVEL: {risk['risk_level']}")
    print(f"   Attrition Probability: {prob*100:.1f}%")
    print(f"   Random Forest: {risk['random_forest_probability']*100:.1f}%")
    print(f"   XGBoost: {risk['xgboost_probability']*100:.1f}%")

    print("\n" + "-"*60)
    print("BUSINESS IMPACT")
    print("-"*60)
    print(f"  Estimated Cost if Lost: {impact['estimated_attrition_cost']}")
    print(f"  Intervention ROI: {impact['intervention_roi']}")
    print(f"  Priority: {impact['priority_level']}")
    print(f"  Potential Savings: {impact['expected_savings']}")

    print("\n" + "-"*60)
    print("RECOMMENDED ACTIONS")
    print("-"*60)
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")

    print("\n" + "="*60)


def batch_predict_from_file(csv_file):
    """Predict from CSV file (simplified version)"""
    import pandas as pd

    print(f"\n⏳ Loading data from {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"   Found {len(df)} employees")

        # Process each employee (showing first 5 for demo)
        results = []
        for idx, row in df.head(5).iterrows():
            # Prepare features (similar to single prediction)
            hours_per_project = row['AvgMonthlyHours'] / \
                row['NumProjects'] if row['NumProjects'] > 0 else 0
            performance_ratio = row['LastEvaluationScore'] / \
                row['SatisfactionScore'] if row['SatisfactionScore'] > 0 else 1.0

            tenure_category = 0 if row['YearsAtCompany'] <= 2 else (
                1 if row['YearsAtCompany'] <= 5 else 2)

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

            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                results.append({
                    'EmployeeID': row.get('EmployeeID', f'EMP{idx:03d}'),
                    'Risk_Level': result['risk_assessment']['risk_level'],
                    'Probability_%': round(result['risk_assessment']['calibrated_probability'] * 100, 1),
                    'Action': result['risk_assessment']['recommended_action']
                })

        # Display summary
        print("\n" + "="*60)
        print("BATCH PREDICTION RESULTS (First 5 Employees)")
        print("="*60)

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        print("\n✅ Complete! For full batch processing, see batch_predict.py")

    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main_menu():
    """Main menu"""
    print("\n" + "="*60)
    print("      HR ATTRITION API CLIENT")
    print("="*60)

    # Test connection
    if not test_connection():
        return

    # Get model info
    print("\n📊 Model Information:")
    details = get_model_details()
    if details:
        print(f"   Strategy: {details['training_strategy']}")
        print(f"   Best Model: {details['models'][0]['name']}")
        print(
            f"   F1-Score: {details['models'][0]['performance']['f1_score']}")
        print(f"   Recall: {details['models'][0]['performance']['recall']}")

    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Predict single employee")
        print("2. Batch predict from CSV")
        print("3. View model details")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == '1':
            predict_single_employee()
        elif choice == '2':
            csv_file = input(
                "Enter CSV file path (or 'sample_employee_data.csv'): ").strip()
            if not csv_file:
                csv_file = 'sample_employee_data.csv'
            batch_predict_from_file(csv_file)
        elif choice == '3':
            details = get_model_details()
            if details:
                print("\n" + json.dumps(details, indent=2))
        elif choice == '4':
            print("\n✅ Goodbye!")
            break
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main_menu()
