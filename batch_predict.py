"""
BATCH ATTRITION PREDICTION
Predict attrition risk for multiple employees from a CSV file.

Usage:
    python batch_predict.py employee_data.csv

Output:
    Creates 'attrition_predictions.csv' with risk scores and recommendations
"""

import joblib
import pandas as pd
import numpy as np
import sys


def load_model():
    """Load the trained model"""
    model_path = 'deployment/models/random_forest_smote.pkl'
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}")
        print("Make sure you're running this from the HR_Analytics root folder")
        return None


def prepare_features(df):
    """
    Prepare features for prediction
    Expects CSV with columns: EmployeeID, Age, Department, SatisfactionScore, 
    LastEvaluationScore, NumProjects, AvgMonthlyHours, YearsAtCompany
    """
    print(f"Processing {len(df)} employees...")

    # Calculate engineered features
    df['HoursPerProject'] = df['AvgMonthlyHours'] / df['NumProjects']
    df['HoursPerProject'] = df['HoursPerProject'].replace(
        [np.inf, -np.inf], 40)

    df['PerformanceRatio'] = df['LastEvaluationScore'] / df['SatisfactionScore']
    df['PerformanceRatio'] = df['PerformanceRatio'].replace(
        [np.inf, -np.inf], 1.0)

    # Tenure category
    def categorize_tenure(years):
        if years <= 2:
            return 0  # New
        elif years <= 5:
            return 1  # Mid
        else:
            return 2  # Experienced

    df['TenureCategory_encoded'] = df['YearsAtCompany'].apply(
        categorize_tenure)

    # High risk flag
    median_hours_per_project = 40
    median_satisfaction = 0.5
    df['High_Risk_Employee'] = ((df['HoursPerProject'] > median_hours_per_project) &
                                (df['SatisfactionScore'] < median_satisfaction)).astype(int)

    # Department dummy variables
    df['Department'] = df['Department'].str.lower().str.strip()
    df['Department_hr'] = (df['Department'] == 'hr').astype(int)
    df['Department_it'] = (df['Department'] == 'it').astype(int)
    df['Department_operations'] = (
        df['Department'] == 'operations').astype(int)
    df['Department_sales'] = (df['Department'] == 'sales').astype(int)
    df['Department_unknown'] = (df['Department'] == 'unknown').astype(int)

    # CRITICAL: Features must be in EXACT order from training
    # This is the order your model was trained on
    feature_columns = [
        'Age',
        'SatisfactionScore',
        'LastEvaluationScore',
        'NumProjects',
        'AvgMonthlyHours',
        'YearsAtCompany',
        'HoursPerProject',
        'PerformanceRatio',
        'TenureCategory_encoded',
        'High_Risk_Employee',
        'Department_hr',
        'Department_it',
        'Department_operations',
        'Department_sales',
        'Department_unknown'
    ]

    return df, feature_columns


def predict_batch(model, df, feature_columns):
    """Make predictions for all employees"""
    X = df[feature_columns]

    # Get predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Add to dataframe
    df['Attrition_Prediction'] = predictions
    df['Risk_Probability'] = probabilities

    # Risk level
    df['Risk_Level'] = pd.cut(
        df['Risk_Probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )

    # Recommendation
    def get_recommendation(risk_level):
        if risk_level == 'HIGH':
            return 'URGENT: Schedule 1-on-1, review workload, discuss retention'
        elif risk_level == 'MEDIUM':
            return 'MONITOR: Check in during next review, ensure manageable workload'
        else:
            return 'STANDARD: Continue regular engagement'

    df['Recommended_Action'] = df['Risk_Level'].apply(get_recommendation)

    return df


def save_results(df):
    """Save predictions to CSV"""
    output_columns = [
        'EmployeeID', 'Age', 'Department', 'SatisfactionScore',
        'YearsAtCompany', 'NumProjects', 'AvgMonthlyHours',
        'Risk_Level', 'Risk_Probability', 'Attrition_Prediction',
        'Recommended_Action'
    ]

    output_df = df[output_columns].copy()
    output_df['Risk_Probability'] = (
        output_df['Risk_Probability'] * 100).round(1)
    output_df = output_df.rename(
        columns={'Risk_Probability': 'Risk_Probability_%'})

    # Sort by risk probability (highest first)
    output_df = output_df.sort_values('Risk_Probability_%', ascending=False)

    output_file = 'attrition_predictions.csv'
    output_df.to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")

    return output_df


def display_summary(df):
    """Display summary statistics"""
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nTotal Employees Analyzed: {len(df)}")
    print(f"\nRisk Distribution:")
    print(df['Risk_Level'].value_counts().to_string())

    print(f"\nTop 5 Highest Risk Employees:")
    print("="*60)
    top_5 = df.nlargest(5, 'Risk_Probability')[
        ['EmployeeID', 'Department', 'Risk_Probability', 'Risk_Level']
    ]
    top_5['Risk_Probability'] = (
        top_5['Risk_Probability'] * 100).round(1).astype(str) + '%'
    print(top_5.to_string(index=False))

    print("\n" + "="*60)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("      BATCH ATTRITION PREDICTION")
    print("="*60)

    # Check for input file
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide input CSV file")
        print("Usage: python batch_predict.py employee_data.csv")
        print("\nExpected CSV columns:")
        print("  - EmployeeID, Age, Department, SatisfactionScore,")
        print("  - LastEvaluationScore, NumProjects, AvgMonthlyHours, YearsAtCompany")
        return

    input_file = sys.argv[1]

    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"\n✅ Loaded {len(df)} employees from {input_file}")
    except FileNotFoundError:
        print(f"\n❌ Error: File '{input_file}' not found")
        return

    # Load model
    model = load_model()
    if model is None:
        return

    # Prepare features
    df, feature_columns = prepare_features(df)

    # Make predictions
    df = predict_batch(model, df, feature_columns)

    # Save results
    output_df = save_results(df)

    # Display summary
    display_summary(df)

    print("\n✅ Batch prediction complete!")


if __name__ == "__main__":
    main()
