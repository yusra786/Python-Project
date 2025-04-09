import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\yusra\Downloads\Pyhton project\heart_attack_predictions.csv')
# Initial Data Exploration
print("Initial Data Exploration")
print("First 5 rows of the dataset:")
print(df.head())
print("\nSummary of the dataset:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
# Check for missing values
print("\nMissing values in each column:")
missing_data = df.isnull().sum()
print(missing_data)

# Look at unique values in each column
print("\nUnique values in each column:")
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].unique())

# Groupby analysis

# Average age by gender
avg_age_gender = df.groupby('Gender')['Age'].mean()
print("\nAverage Age for each Gender:")
print(avg_age_gender)

# Average cholesterol by smoking history
avg_chol_by_smoking = df.groupby('Smoking_History')['Cholesterol_Level'].mean()
print("\nAverage Cholesterol Level by Smoking History:")
print(avg_chol_by_smoking)

# Average blood pressure by heart disease risk level
avg_bp_by_risk = df.groupby('Heart_Disease_Risk')['Blood_Pressure'].mean()
print("\nAverage Blood Pressure by Heart Disease Risk:")
print(avg_bp_by_risk)

# Count of outcomes (Survived vs Died)
outcome_counts = df['Heart_Attack_Outcome'].value_counts()
print("\nHeart Attack Outcome Counts:")
print(outcome_counts)

# Average age by education level
avg_age_by_edu = df.groupby('Education_Level')['Age'].mean()
print("\nAverage Age by Education Level:")
print(avg_age_by_edu)

# Count of heart attack outcomes by income level
outcome_by_income = df.groupby('Income_Level')['Heart_Attack_Outcome'].value_counts()
print("\nHeart Attack Outcome Counts by Income Level:")
print(outcome_by_income)

# Replace 'Male' with 'M' and 'Female' with 'F' in the 'Gender' column using loc
df.loc[df['Gender'] == 'Male', 'Gender'] = 'M'
df.loc[df['Gender'] == 'Female', 'Gender'] = 'F'


