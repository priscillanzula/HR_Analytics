# HR Attrition Prediction Tool - Setup Guide

## Quick Start (5 minutes)

### Step 1: Install Required Software
```bash
# Install Python packages
pip install pandas numpy scikit-learn joblib xgboost imbalanced-learn matplotlib seaborn fastapi uvicorn pydantic requests
```

### Step 2: Verify Model Files
Make sure you have these files in your deployment folder:
- `api_smote.py` (the FastAPI server)
- `random_forest_smote.pkl` (trained model)
- `xgboost_smote.pkl` (backup model)
- `metadata.pkl` (model information)
- `preprocessor.pkl` (feature preprocessing)

If missing, run the `HR_Analytics_Modelling.ipynb` notebook first.

### Step 3: Test the Tool

**Option A: REST API (Recommended for Production)**
```bash
# Terminal 1: Start API server
python start_api.py

# Terminal 2: Use interactive client
python api_client.py
```
Access API documentation at: `http://localhost:8000/docs`

**Option B: Command Line (Quick Testing)**
```bash
# Single employee
python predict_attrition.py

# Batch processing
python batch_predict.py sample_employee_data.csv
```

---

## How to Use for Your HR Team

### REST API (Production Use)

**Starting the API:**
```bash
python start_api.py
```
Server runs at `http://localhost:8000`

**Using the API:**

1. **Interactive Documentation**:
   - Open browser: `http://localhost:8000/docs`
   - Test endpoints directly in browser
   - See request/response formats

2. **Python Client**:
   ```bash
   python api_client.py
   ```
   Interactive menu for predictions

3. **Direct API Calls** (for integration):
   ```python
   import requests
   
   payload = {
       "Age": 35,
       "SatisfactionScore": 0.45,
       "LastEvaluationScore": 0.75,
       # ... other fields
   }
   
   response = requests.post(
       "http://localhost:8000/predict",
       json=payload
   )
   result = response.json()
   ```

### Command Line Tools (Quick Use)

### Individual Prediction
1. Run: `python predict_attrition.py`
2. Enter employee details when prompted
3. Get instant risk assessment and recommendations

### Batch Prediction (Recommended for Monthly Reviews)
1. Prepare a CSV file with employee data (see `sample_employee_data.csv` for format)
2. Run: `python batch_predict.py your_employee_data.csv`
3. Open `attrition_predictions.csv` - employees sorted by risk
4. Focus on HIGH RISK employees first

### Required CSV Columns
- EmployeeID
- Age
- Department (finance, hr, it, operations, sales, or unknown)
- SatisfactionScore (0.0 to 1.0)
- LastEvaluationScore (0.0 to 1.0)
- NumProjects
- AvgMonthlyHours
- YearsAtCompany

---

## Understanding the Results

### Risk Levels
- 🟢 **LOW RISK** (0-30%): Employee likely to stay
- 🟡 **MEDIUM RISK** (30-60%): Monitor closely
- 🔴 **HIGH RISK** (60-100%): Immediate action needed

### Key Warning Signs (from model)
1. **High workload**: >6 projects or >220 hours/month
2. **Low satisfaction**: <0.4 score
3. **Mid-tenure burnout**: 2-5 years at company
4. **High hours per project**: Doing too much work per task

---

## Troubleshooting

**Error: Model file not found**
→ Run the modeling notebook to train and save the model first

**Error: CSV file not found**
→ Check the file path, make sure it's in the same folder

**Error: Missing columns**
→ Your CSV must have ALL 8 required columns (see above)

---

## Next Steps

1. **Monthly Reviews**: Run batch predictions on all employees
2. **Focus Efforts**: Start with HIGH RISK employees
3. **Track Results**: Did interventions reduce attrition?
4. **Retrain Model**: Every 6 months with new data

---

## Support
For questions, contact: [Your Email]
Project GitHub: [Your Repository URL]
