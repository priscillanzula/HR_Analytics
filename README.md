# 🎯 HR Analytics: Employee Attrition Prediction

A machine learning project to predict which employees are at risk of leaving, helping HR teams take proactive retention actions.

---

## 📋 Project Overview

**Business Problem**: Employee turnover costs companies thousands in recruitment, training, and lost productivity. Most HR teams react AFTER employees quit.

**Solution**: Use machine learning to identify at-risk employees BEFORE they leave, enabling targeted retention strategies.

**Key Features**:
- Predicts attrition risk with 71% accuracy (F1-score 0.709)
- Identifies top drivers: workload, satisfaction, tenure
- Provides actionable recommendations for HR intervention
- Works for individual employees or batch predictions

---

## 🎬 Quick Demo

### Option 1: REST API (Recommended - Production Ready)
```bash
# Start the API server
python start_api.py

# In another terminal, use the client
python api_client.py
```
Professional FastAPI deployment with Swagger docs at `http://localhost:8000/docs`

### Option 2: Command Line Tools
```bash
# Single employee prediction
python predict_attrition.py

# Batch predictions
python batch_predict.py sample_employee_data.csv
```

---

## 📊 Key Findings

From analyzing 29,524 employees:

1. **33% attrition rate** across the company
2. **Top risk factors**:
   - High workload (>6 projects or 220+ hours/month)
   - Low satisfaction (<0.4 score)
   - Mid-tenure employees (2-5 years) most vulnerable
   - High hours-per-project ratio indicates burnout

3. **Department impact**: 
   - Finance & Sales: 33.5% attrition
   - Operations: 31.7% attrition
   - Minimal variation suggests company-wide issue

4. **High-risk employees**: 4-6x more likely to leave when flagged by our feature engineering

---

## 🛠️ Technical Stack

- **Python 3.8+**
- **API Framework**: FastAPI + Uvicorn (REST API deployment)
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Deployment**: 
  - FastAPI REST API with Swagger documentation
  - Simple Python scripts for direct use
- **Models**: Random Forest (primary), XGBoost (ensemble)
- **Techniques**: SMOTE for class imbalance, ensemble predictions, probability calibration

---

## 📁 Project Structure

```
HR_Analytics/
├── notebooks/
│   ├── hr_dataset_cleaning.ipynb          # Data cleaning & preprocessing
│   ├── feature_engineering.ipynb          # Feature creation
│   ├── HR_Analytics_HR_EDA.ipynb          # Exploratory analysis
│   └── HR_Analytics_Modelling.ipynb       # Model training & evaluation
├── deployment/
│   ├── api_smote.py                       # FastAPI REST API (main deployment)
│   ├── random_forest_smote.pkl            # Trained Random Forest model
│   ├── xgboost_smote.pkl                  # Trained XGBoost model
│   ├── preprocessor.pkl                   # Feature preprocessing pipeline
│   ├── metadata.pkl                       # Model metadata & performance
│   └── smote_config.pkl                   # SMOTE configuration
├── start_api.py                           # API server launcher
├── api_client.py                          # Interactive API client
├── predict_attrition.py                   # Single employee prediction (CLI)
├── batch_predict.py                       # Batch prediction script (CLI)
├── sample_employee_data.csv               # Example data template
├── DEPLOYMENT_GUIDE.md                    # Setup instructions
└── README.md                              # This file
```

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/HR_Analytics.git
cd HR_Analytics
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib xgboost imbalanced-learn matplotlib seaborn fastapi uvicorn pydantic requests
```

### 3. Choose Your Deployment Method

**Option A: FastAPI REST API (Recommended)**
```bash
# Start API server
python start_api.py

# Access interactive docs at http://localhost:8000/docs
# Or use the client: python api_client.py
```

**Option B: Command Line Tools**
```bash
# Single employee
python predict_attrition.py

# Batch processing
python batch_predict.py sample_employee_data.csv
```

Full setup guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

---

## 📈 Model Performance

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| **Random Forest** | **0.67** | **0.75** | **0.71** | **0.87** |
| XGBoost | 0.69 | 0.73 | 0.71 | 0.87 |
| Logistic Regression | 0.62 | 0.74 | 0.67 | 0.83 |

**Why Random Forest?**
- Best F1-score (balances precision and recall)
- High recall (75%) - catches most employees at risk
- Good ROC-AUC (0.87) - strong discrimination ability

---

## 💡 Feature Importance

Top predictors of attrition (from Random Forest):

1. **NumProjects** (24.4%) - Number of concurrent projects
2. **AvgMonthlyHours** (23.0%) - Total work hours
3. **SatisfactionScore** (20.3%) - Employee satisfaction
4. **HoursPerProject** (8.8%) - Workload intensity
5. **PerformanceRatio** (6.3%) - Performance vs. satisfaction

---

## 🎯 Business Impact

### For HR Teams:
- **Identify high-risk employees** 1-3 months before they quit
- **Prioritize interventions** on employees with 60%+ risk
- **Reduce turnover costs** by focusing on fixable issues (workload, satisfaction)

### Recommended Actions (Auto-generated):
- 🔴 **HIGH RISK**: Schedule 1-on-1, review workload, discuss retention
- 🟡 **MEDIUM RISK**: Monitor closely, ensure manageable workload
- 🟢 **LOW RISK**: Continue standard engagement

---

## 📝 Data Requirements

Your employee CSV file needs these columns:

| Column | Description | Example |
|--------|-------------|---------|
| EmployeeID | Unique identifier | EMP001 |
| Age | Employee age | 35 |
| Department | finance, hr, it, operations, sales, unknown | finance |
| SatisfactionScore | 0.0 to 1.0 | 0.65 |
| LastEvaluationScore | 0.0 to 1.0 | 0.82 |
| NumProjects | Current projects | 5 |
| AvgMonthlyHours | Average hours/month | 200 |
| YearsAtCompany | Tenure in years | 3 |

See [sample_employee_data.csv](sample_employee_data.csv) for template.

---

## 📊 Sample Output

```
ATTRITION RISK ASSESSMENT
==================================================

🔴 RISK LEVEL: HIGH RISK
   Probability of Leaving: 78.5%
   Prediction: WILL LEAVE

KEY METRICS:
  Satisfaction Score: 0.32
  Hours Per Project: 62.3
  Number of Projects: 8
  High Risk Flag: YES

RECOMMENDED ACTIONS:
🔴 URGENT - Schedule 1-on-1 meeting immediately
   • Review workload and redistribute projects
   • Discuss career development opportunities
   • Consider retention bonus or promotion
```

---

## 🔄 Future Improvements

1. **Add more features**: Salary, promotion history, manager ratings
2. **Time-series analysis**: Predict WHEN employee will leave
3. **Web dashboard**: Interactive visualization for HR
4. **A/B testing**: Measure impact of interventions
5. **Model retraining**: Automate monthly updates

---

## 👨‍💻 Author

**Your Name**  
Data Scientist | Machine Learning Engineer

- 📧 Email: your.email@example.com
- 💼 LinkedIn: [your-linkedin](https://linkedin.com/in/yourprofile)
- 🔗 Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset: HR Analytics (modified for privacy)
- Inspiration: Reducing employee turnover through data science
- Tools: scikit-learn, pandas, XGBoost

---

## 📞 Support

Questions or feedback? 
- Open an [issue](https://github.com/yourusername/HR_Analytics/issues)
- Email: your.email@example.com

---

**⭐ If this project helped you, please give it a star!**
