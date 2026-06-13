import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_mock_data(filepath='raw_data.csv'):
    # Generate some synthetic data to simulate a real-world scenario
    np.random.seed(42)
    n_samples = 1000

    data = {
        'ID': range(1, n_samples + 1),
        'Age': np.random.normal(35, 10, n_samples),
        'Income': np.random.normal(60000, 15000, n_samples),
        'Department': np.random.choice(['IT', 'HR', 'Marketing', 'Sales', 'Finance'], n_samples),
        'Performance_Score': np.random.uniform(1, 100, n_samples)
    }

    df = pd.DataFrame(data)

    # 1. Introduce Missing Values
    df.loc[np.random.choice(df.index, 50, replace=False), 'Age'] = np.nan
    df.loc[np.random.choice(df.index, 100, replace=False), 'Income'] = np.nan

    # 2. Introduce Outliers
    df.loc[0:10, 'Income'] = df['Income'][0:11] * 5
    df.loc[990:1000, 'Age'] = 150

    # 3. Introduce Duplicates
    duplicates = df.sample(20)
    df = pd.concat([df, duplicates], ignore_index=True)

    df.to_csv(filepath, index=False)
    print(f"Mock data generated and saved to {filepath}")
    return filepath

def clean_data(df):
    print("Starting Data Cleaning Process...")

    # Step 1: Handle Duplicates
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Removed {initial_shape[0] - df.shape[0]} duplicates.")

    # Step 2: Handle Missing Values
    # Impute Age and Income with their respective medians to avoid outlier skew
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Income'] = df['Income'].fillna(df['Income'].median())
    print("Filled missing values in 'Age' and 'Income' with medians.")

    # Step 3: Handle Outliers (using the Interquartile Range / IQR method)
    for col in ['Age', 'Income']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"Removed {outlier_count} outliers from {col}.")

    print("Data Cleaning Complete.")
    return df

def visualize_data(df, output_dir='visualizations'):
    # Create directory for saving the dashboard charts
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sns.set_theme(style="whitegrid")

    # Chart 1: Distribution of Income
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Income'], kde=True, color='teal')
    plt.title('Income Distribution after Cleaning')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_income_distribution.png")
    plt.close()

    # Chart 2: Boxplot of Age by Department
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Department', y='Age', data=df, hue='Department', legend=False, palette='Set2')
    plt.title('Age Distribution by Department')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_age_by_department_boxplot.png")
    plt.close()

    # Chart 3: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['Age', 'Income', 'Performance_Score']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_correlation_heatmap.png")
    plt.close()

    print(f"Visualizations saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    # Generate the raw messy dataset
    raw_file = generate_mock_data()
    
    # Load raw data
    raw_df = pd.read_csv(raw_file)
    print(f"\nRaw Data Overview:\n{raw_df.head()}\n")
    
    # Clean data
    cleaned_df = clean_data(raw_df)
    
    # Save cleaned data
    cleaned_df.to_csv('cleaned_data.csv', index=False)
    print(f"\nCleaned data (Shape: {cleaned_df.shape}) saved to 'cleaned_data.csv'.")
    
    # Visualize insights
    visualize_data(cleaned_df)
    print("\nProject workflow completed successfully!")
