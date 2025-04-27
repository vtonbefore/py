# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For built-in Iris dataset

# Task 1: Load and Explore the Dataset
# -----------------------------------
# Load the Iris dataset (built into seaborn)
df = sns.load_dataset('iris')

# Display first 5 rows
print("First 5 rows of the Iris dataset:")
print(df.head())

# Check data types and missing values
print("\nData types and missing values:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean data (no missing values in Iris, but included for generality)
df_cleaned = df.dropna()  # Or use df.fillna(0) if needed

# Task 2: Basic Data Analysis
# ---------------------------
# Basic statistics
print("\nBasic statistics:")
print(df_cleaned.describe())

# Group by species and compute mean
print("\nMean values per species:")
print(df_cleaned.groupby('species').mean())

# Observation
print("\nObservation: Setosa has smaller petals but wider sepals compared to other species.")

# Task 3: Data Visualization
# -------------------------
plt.figure(figsize=(15, 10))

# 1. Line chart (sepal_length over index)
plt.subplot(2, 2, 1)
plt.plot(df_cleaned['sepal_length'], color='blue')
plt.title('Trend of Sepal Length (by Index)')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')

# 2. Bar chart (mean petal_length per species)
plt.subplot(2, 2, 2)
df_cleaned.groupby('species')['petal_length'].mean().plot(kind='bar', color='green')
plt.title('Average Petal Length by Species')
plt.ylabel('Petal Length (cm)')

# 3. Histogram (distribution of sepal_width)
plt.subplot(2, 2, 3)
plt.hist(df_cleaned['sepal_width'], bins=15, color='orange')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')

# 4. Scatter plot (sepal_length vs. petal_length)
plt.subplot(2, 2, 4)
plt.scatter(df_cleaned['sepal_length'], df_cleaned['petal_length'], color='red')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

plt.tight_layout()
plt.show()