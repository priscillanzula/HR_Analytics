from faker import Faker
import pandas as pd
import random
import numpy as np

fake = Faker()
data = []

# Generate 30,000 records
for _ in range(30000):
    # Introduce some missing values and typos
    age = random.choice([random.randint(22, 60), None])  # ~missing ages
    department = random.choice(["Sales", "IT", "HR", "Operations", "Finance", "Hr", "finanCe", None])
    satisfaction = round(random.uniform(0, 1), 2)
    last_eval = round(random.uniform(0, 1), 2)
    projects = random.choice([random.randint(2, 7), None])
    hours = random.choice([random.randint(150, 310), 999])  # 999 as an outlier
    years = random.choice([random.randint(1, 10), None])
    left = random.choice([0, 1, None])  # some missing target values
    
    data.append({
        "Employee_ID": fake.random_int(min=1000, max=999999),  # allow duplicates
        "Age": age,
        "Department": department,
        "Satisfaction_Level": satisfaction,
        "Last_Evaluation": last_eval,
        "Projects": projects,
        "Average_Monthly_Hours": hours,
        "Years_at_Company": years,
        "Left": left
    })

# Introduce some exact duplicates
for _ in range(100):
    data.append(random.choice(data))

# Create DataFrame
df_hr = pd.DataFrame(data)

# Save to CSV
df_hr.to_csv("synthetic_hr_data_messy.csv", index=False)

print("Messy dataset created with 30,000+ rows ✅")
