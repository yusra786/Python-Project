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

# Add a new column:
#1. Adding Is_Senior_Citizen coloumn
df['Is_Senior_Citizen'] = df['Age'] >= 60

print("\nAdded 'Is_Senior_Citizen' column (True if Age >= 60):")
print(df[['Age', 'Is_Senior_Citizen']].head())

#2. Adding BMI Column
df['BMI'] = df.apply(lambda row: (row['Cholesterol_Level'] / 2 + row['Blood_Pressure'] / 3) / (1.75 ** 2), axis=1)

#3. Adding BMI Category Column
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 24.9:
        return 'Normal weight'
    elif bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

print(df[['Cholesterol_Level', 'Blood_Pressure', 'BMI', 'BMI_Category']].head())

# Descriptive Statistics 
print("\n Cholesterol Level")
print("Mean:", df["Cholesterol_Level"].mean())
print("Median:", df["Cholesterol_Level"].median())
print("Mode:", df["Cholesterol_Level"].mode()[0])
print("Count:", df["Cholesterol_Level"].count())
print("Max:", df["Cholesterol_Level"].max())
print("Min:", df["Cholesterol_Level"].min())

print("\n Blood Pressure")
print("Mean:", df["Blood_Pressure"].mean())
print("Median:", df["Blood_Pressure"].median())
print("Mode:", df["Blood_Pressure"].mode()[0])
print("Count:", df["Blood_Pressure"].count())
print("Max:", df["Blood_Pressure"].max())
print("Min:", df["Blood_Pressure"].min())

print("\n Age ")
print("Mean:", df["Age"].mean())
print("Median:", df["Age"].median())
print("Mode:", df["Age"].mode()[0])
print("Count:", df["Age"].count())
print("Max:", df["Age"].max())
print("Min:", df["Age"].min())



