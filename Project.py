import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\yusra\Downloads\heart_attack_predictions (1).csv")
#Step 1:Initial Data Exploration
print("Initial Data Exploration")
print("First 5 rows of the dataset:")
print(df.head())
print("\nSummary of the dataset:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
#Step 2: Check for missing values
print("\nMissing values in each column:")
missing_data = df.isnull().sum()
print(missing_data)

#Step 3: Data Cleaning
print("\n Data Cleaning")
df_cleaned = df.dropna()
print(f"Dropped rows with missing values. New dataset shape: {df_cleaned.shape}")

#Step 4: Look at unique values in each column
print("\nUnique values in each column:")
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].unique())

#Step 5: EDA and Statistical Analysis
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
# Adding Is_Senior_Citizen coloumn
df['Is_Senior_Citizen'] = df['Age'] >= 60

print("\nAdded 'Is_Senior_Citizen' column (True if Age >= 60):")
print(df[['Age', 'Is_Senior_Citizen']].head())



#Step 6: Descriptive Statistics 
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

# Step 7: Visualizing the Distribution of the Target Variable
plt.figure(figsize=(8, 5))
sns.countplot(x='Heart_Attack_Outcome', data=df_cleaned)
plt.title('Distribution of Target Variable (Heart Disease)')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.grid()
plt.show()
print("Observation: The distribution of the target variable shows the number of patients with and without heart disease.")

# Step 8: Visualizing Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Age'], bins=20, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()
print("Observation: The age distribution of patients is visualized.")

# Step 9: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Cholesterol_Level', hue='Heart_Attack_Outcome', data=df_cleaned)
plt.title('Scatter Plot of Age vs Cholesterol Level')
plt.xlabel('Age')
plt.ylabel('Cholesterol Level')
plt.grid()
plt.show()
print("Observation: The scatter plot shows the relationship between Age and Cholesterol Level, colored by Heart Attack Outcome.")

# Step 10: Line Plot 
# Calculate average cholesterol level by age
avg_chol_by_age = df_cleaned.groupby('Age')['Cholesterol_Level'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Cholesterol_Level', data=avg_chol_by_age, marker='o')
plt.title('Line Plot of Average Cholesterol Level by Age')
plt.xlabel('Age')
plt.ylabel('Average Cholesterol Level')
plt.grid()
plt.show()
print("Observation: The line plot shows the trend of average Cholesterol Level as Age increases.")

# Step 11: Correlation Matrix
print("\n Correlation Matrix ")

numeric_df = df_cleaned.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
print("Observation: The correlation matrix shows relationships between numeric features.")

# Step 12: Boxplots for Continuous Variables

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_cleaned[['Age', 'Cholesterol_Level', 'Max_Heart_Rate_Achieved', 'Thalassemia']], orient='h')
plt.title('Boxplots of Continuous Variables')
plt.xlabel('Value')
plt.ylabel('Variables')
plt.show()
print("Observation: Boxplots help identify outliers in continuous variables.")

# Step 13: Pairplot for Feature Relationships
sns.pairplot(df_cleaned, hue='Heart_Attack_Outcome', vars=['Age', 'Cholesterol_Level', 'Max_Heart_Rate_Achieved', 'Triglycerides'])
plt.title('Pairplot of Selected Features')
plt.show()

# Step 14: Outlier Detection using IQR

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(f"{column}: {len(outliers)} outliers")
    return outliers
numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

outliers_dict = {}
for col in numerical_columns:
    outliers = detect_outliers_iqr(df_cleaned, col)
    outliers_dict[col] = outliers
    
# Step 15: Final Summary of EDA
print("\n Final Summary of EDA ")
print("Number of records after cleaning:", outliers.shape[0])
print("Number of features:", outliers.shape[1])
print("EDA completed successfully.")

