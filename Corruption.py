# Test Score Analysis for Potential Corruption - MEDICINA Profession

This notebook analyzes test scores for the "MEDICINA" profession across different districts to identify potential signs of corruption or test leaks.

## Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene, f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
```

## Loading and Preparing the Data

```python
# Load the Excel file
# Replace 'your_file.xlsx' with your actual file path
df = pd.read_excel('your_file.xlsx')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Number of records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Display the first 5 rows
print("\nSample Data:")
display(df.head())

# Filter for MEDICINA profession only
medicina_df = df[df['PROFESION'] == 'MEDICINA'].copy()
print(f"\nNumber of MEDICINA records: {len(medicina_df)}")

# Check for missing values
print("\nMissing Values:")
print(medicina_df.isnull().sum())

# Basic statistics for MEDICINA test scores
print("\nBasic Statistics for MEDICINA Test Scores:")
display(medicina_df['NOTA'].describe())
```

## Statistical Analysis for Corruption Detection

### 1. Distribution Analysis by Region

```python
# Number of candidates per region
plt.figure(figsize=(12, 6))
region_counts = medicina_df['REGION DE EVALUACION'].value_counts()
sns.barplot(x=region_counts.index, y=region_counts.values)
plt.title('Number of MEDICINA Candidates by Region', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Number of Candidates', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Distribution of scores by region - Box Plot
plt.figure(figsize=(14, 8))
sns.boxplot(x='REGION DE EVALUACION', y='NOTA', data=medicina_df)
plt.axhline(y=medicina_df['NOTA'].mean(), color='r', linestyle='--', label=f'Overall Mean: {medicina_df["NOTA"].mean():.2f}')
plt.title('Distribution of MEDICINA Test Scores by Region', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Test Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Violin plots for better distribution visualization
plt.figure(figsize=(14, 8))
sns.violinplot(x='REGION DE EVALUACION', y='NOTA', data=medicina_df, inner='box')
plt.title('Violin Plot of MEDICINA Test Scores by Region', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Test Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### 2. Identifying Outlier Regions

```python
# Calculate z-scores for each region's mean score
region_stats = medicina_df.groupby('REGION DE EVALUACION')['NOTA'].agg(['mean', 'std', 'count']).reset_index()
overall_mean = medicina_df['NOTA'].mean()
overall_std = medicina_df['NOTA'].std()

# Add z-score column
region_stats['z_score'] = (region_stats['mean'] - overall_mean) / overall_std

# Sort by z-score to identify potential outliers
region_stats_sorted = region_stats.sort_values('z_score', ascending=False)
display(region_stats_sorted)

# Plot z-scores
plt.figure(figsize=(12, 6))
bars = plt.bar(region_stats_sorted['REGION DE EVALUACION'], region_stats_sorted['z_score'])
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=1.96, color='r', linestyle='--', label='95% Confidence Threshold')
plt.axhline(y=-1.96, color='r', linestyle='--')
plt.title('Z-Scores of Mean MEDICINA Test Scores by Region', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Z-Score', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Color bars based on significance
for i, bar in enumerate(bars):
    if abs(region_stats_sorted['z_score'].iloc[i]) > 1.96:
        bar.set_color('red')
    else:
        bar.set_color('blue')

plt.legend()
plt.tight_layout()
plt.show()
```

### 3. Statistical Tests for Detecting Anomalies

```python
# Test for normality in each region (Shapiro-Wilk test)
print("Shapiro-Wilk Test for Normality by Region:")
normality_results = []

for region in medicina_df['REGION DE EVALUACION'].unique():
    region_scores = medicina_df[medicina_df['REGION DE EVALUACION'] == region]['NOTA']
    if len(region_scores) >= 3:  # Shapiro requires at least 3 data points
        stat, p = shapiro(region_scores)
        normality_results.append({
            'Region': region,
            'Statistic': stat,
            'p-value': p,
            'Normal Distribution': p > 0.05
        })

normality_df = pd.DataFrame(normality_results)
display(normality_df)

# Test for homogeneity of variances (Levene's test)
region_groups = [medicina_df[medicina_df['REGION DE EVALUACION'] == region]['NOTA'] 
                for region in medicina_df['REGION DE EVALUACION'].unique() 
                if len(medicina_df[medicina_df['REGION DE EVALUACION'] == region]) >= 2]

if len(region_groups) >= 2:
    stat, p = levene(*region_groups)
    print(f"\nLevene's Test for Homogeneity of Variances:")
    print(f"Statistic: {stat:.4f}, p-value: {p:.4f}")
    print(f"Equal variances: {p > 0.05}")

# One-way ANOVA or Kruskal-Wallis test
if normality_df['Normal Distribution'].all() and p > 0.05:
    # Use ANOVA for normally distributed data with equal variances
    f_stat, p_anova = f_oneway(*region_groups)
    print("\nOne-way ANOVA:")
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_anova:.4f}")
    print(f"Significant difference between regions: {p_anova < 0.05}")
    
    if p_anova < 0.05:
        # Post-hoc test (Tukey's HSD)
        tukey = pairwise_tukeyhsd(endog=medicina_df['NOTA'], 
                                 groups=medicina_df['REGION DE EVALUACION'],
                                 alpha=0.05)
        print("\nTukey's HSD Post-hoc Test:")
        print(tukey)
else:
    # Use Kruskal-Wallis for non-normal data or unequal variances
    h_stat, p_kruskal = kruskal(*region_groups)
    print("\nKruskal-Wallis Test:")
    print(f"H-statistic: {h_stat:.4f}, p-value: {p_kruskal:.4f}")
    print(f"Significant difference between regions: {p_kruskal < 0.05}")
```

### 4. Score Distribution Analysis

```python
# Histogram of all MEDICINA scores
plt.figure(figsize=(12, 6))
sns.histplot(medicina_df['NOTA'], kde=True, bins=20)
plt.title('Distribution of All MEDICINA Test Scores', fontsize=15)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(x=medicina_df['NOTA'].mean(), color='r', linestyle='--', 
           label=f'Mean: {medicina_df["NOTA"].mean():.2f}')
plt.axvline(x=medicina_df['NOTA'].median(), color='g', linestyle='-.', 
           label=f'Median: {medicina_df["NOTA"].median():.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# QQ Plot to check for normality in overall distribution
plt.figure(figsize=(10, 6))
sm.qqplot(medicina_df['NOTA'], line='s')
plt.title('Q-Q Plot of MEDICINA Test Scores', fontsize=15)
plt.tight_layout()
plt.show()

# Histograms for top 3 regions with most candidates
top_regions = region_counts.nlargest(3).index
plt.figure(figsize=(15, 5))

for i, region in enumerate(top_regions):
    plt.subplot(1, 3, i+1)
    region_data = medicina_df[medicina_df['REGION DE EVALUACION'] == region]
    sns.histplot(region_data['NOTA'], kde=True, bins=15)
    plt.title(f'Scores in {region}', fontsize=12)
    plt.xlabel('Score', fontsize=10)
    if i == 0:
        plt.ylabel('Frequency', fontsize=10)
    else:
        plt.ylabel('')

plt.tight_layout()
plt.show()
```

### 5. Analyzing Score Clustering

```python
# Create a histogram for each region
regions = medicina_df['REGION DE EVALUACION'].unique()
num_regions = len(regions)
rows = (num_regions + 2) // 3  # Ceiling division to determine number of rows

plt.figure(figsize=(15, rows * 4))

for i, region in enumerate(regions):
    plt.subplot(rows, 3, i+1)
    region_data = medicina_df[medicina_df['REGION DE EVALUACION'] == region]
    sns.histplot(region_data['NOTA'], kde=True, bins=15)
    plt.title(f'Scores in {region} (n={len(region_data)})', fontsize=12)
    plt.xlabel('Score', fontsize=10)
    
    # Calculate skewness and kurtosis
    skewness = region_data['NOTA'].skew()
    kurtosis = region_data['NOTA'].kurtosis()
    
    # Add text with skewness and kurtosis
    plt.annotate(f'Skew: {skewness:.2f}\nKurt: {kurtosis:.2f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.tight_layout()
plt.show()

# Look for unusual clusters or peaks in scores
# Create a kernel density estimate for each region
plt.figure(figsize=(12, 6))

for region in regions:
    region_data = medicina_df[medicina_df['REGION DE EVALUACION'] == region]
    sns.kdeplot(region_data['NOTA'], label=region)

plt.title('Kernel Density Estimate of MEDICINA Scores by Region', fontsize=15)
plt.xlabel('Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### 6. Corruption Indicators - Unusual Patterns

```python
# Calculate various statistical indicators for each region
region_indicators = []

for region in regions:
    region_data = medicina_df[medicina_df['REGION DE EVALUACION'] == region]['NOTA']
    
    # Skip if too few data points
    if len(region_data) < 5:
        continue
        
    # Calculate various statistics
    indicators = {
        'Region': region,
        'Count': len(region_data),
        'Mean': region_data.mean(),
        'Median': region_data.median(),
        'Std Dev': region_data.std(),
        'Skewness': region_data.skew(),
        'Kurtosis': region_data.kurtosis(),
        '% Above Overall Mean': (region_data > medicina_df['NOTA'].mean()).mean() * 100,
        'Perfect Score Count': (region_data == region_data.max()).sum(),
        'Score Range': region_data.max() - region_data.min(),
        'Q3-Q1': region_data.quantile(0.75) - region_data.quantile(0.25)
    }
    
    # Add clustering indicator
    # High kurtosis can indicate unusual clustering
    # Negative skew might indicate many high scores
    region_indicators.append(indicators)

# Convert to DataFrame and sort by suspicious indicators
indicators_df = pd.DataFrame(region_indicators)

# Calculate an overall suspicion score based on multiple factors
indicators_df['Suspicion Score'] = (
    # Z-score of the mean (from earlier)
    abs(indicators_df['Mean'] - medicina_df['NOTA'].mean()) / medicina_df['NOTA'].std() +
    # Weight for skewness - negative skew (high scores) is more suspicious
    abs(indicators_df['Skewness']) * (indicators_df['Skewness'] < 0) * 0.5 +
    # Weight for high kurtosis (unusual peaks)
    abs(indicators_df['Kurtosis']) * 0.3 +
    # Weight for % above mean
    (indicators_df['% Above Overall Mean'] > 75) * 2 +
    # Small score range could indicate answer sharing
    (indicators_df['Score Range'] < indicators_df['Score Range'].median()) * 0.5
)

# Sort by suspicion score
suspicious_regions = indicators_df.sort_values('Suspicion Score', ascending=False)
display(suspicious_regions)

# Visualize suspicion scores
plt.figure(figsize=(12, 6))
bars = plt.bar(suspicious_regions['Region'], suspicious_regions['Suspicion Score'])

# Color bars based on suspicion level
for i, bar in enumerate(bars):
    if suspicious_regions['Suspicion Score'].iloc[i] > 3:
        bar.set_color('red')
    elif suspicious_regions['Suspicion Score'].iloc[i] > 2:
        bar.set_color('orange')
    else:
        bar.set_color('blue')

plt.title('Suspicion Score by Region (Higher values warrant investigation)', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Suspicion Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

### 7. Score Pattern Analysis

```python
# Plot the percentage of exact same scores in each region
exact_score_counts = []

for region in regions:
    region_data = medicina_df[medicina_df['REGION DE EVALUACION'] == region]
    
    if len(region_data) < 5:  # Skip regions with too few candidates
        continue
        
    # Get value counts and calculate percentage
    value_counts = region_data['NOTA'].value_counts()
    duplicate_pct = sum(count for count in value_counts if count > 1) / len(region_data) * 100
    
    # Find most common score and its percentage
    most_common_score = value_counts.idxmax()
    most_common_pct = value_counts.max() / len(region_data) * 100
    
    exact_score_counts.append({
        'Region': region,
        'Candidates': len(region_data),
        'Duplicate Score %': duplicate_pct,
        'Most Common Score': most_common_score,
        'Most Common Score %': most_common_pct
    })

exact_score_df = pd.DataFrame(exact_score_counts)
exact_score_df_sorted = exact_score_df.sort_values('Duplicate Score %', ascending=False)
display(exact_score_df_sorted)

# Plot the percentage of duplicate scores
plt.figure(figsize=(12, 6))
bars = plt.bar(exact_score_df_sorted['Region'], exact_score_df_sorted['Duplicate Score %'])

# Color bars based on duplicate percentage
for i, bar in enumerate(bars):
    if exact_score_df_sorted['Duplicate Score %'].iloc[i] > 50:
        bar.set_color('red')
    elif exact_score_df_sorted['Duplicate Score %'].iloc[i] > 30:
        bar.set_color('orange')
    else:
        bar.set_color('blue')

plt.title('Percentage of Candidates with Duplicate Scores by Region', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=exact_score_df['Duplicate Score %'].mean(), color='black', linestyle='--', 
           label=f'Mean: {exact_score_df["Duplicate Score %"].mean():.1f}%')
plt.legend()
plt.tight_layout()
plt.show()

# Plot for most common score percentage
plt.figure(figsize=(12, 6))
bars = plt.bar(exact_score_df_sorted['Region'], exact_score_df_sorted['Most Common Score %'])

# Color bars based on percentage
for i, bar in enumerate(bars):
    if exact_score_df_sorted['Most Common Score %'].iloc[i] > 25:
        bar.set_color('red')
    elif exact_score_df_sorted['Most Common Score %'].iloc[i] > 15:
        bar.set_color('orange')
    else:
        bar.set_color('blue')

plt.title('Percentage of Candidates with the Most Common Score by Region', fontsize=15)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.axhline(y=exact_score_df['Most Common Score %'].mean(), color='black', linestyle='--', 
           label=f'Mean: {exact_score_df["Most Common Score %"].mean():.1f}%')
plt.legend()
plt.tight_layout()
plt.show()
```

## Summary of Findings

```python
# Display regions with high suspicion scores
high_suspicion = suspicious_regions[suspicious_regions['Suspicion Score'] > 2]
print("Regions with high suspicion scores that may warrant further investigation:")
display(high_suspicion)

# Display regions with unusual percentage of duplicate scores
high_duplicates = exact_score_df[exact_score_df['Duplicate Score %'] > 50]
print("\nRegions with unusually high percentage of duplicate scores:")
display(high_duplicates)

# Correlation analysis between suspicious indicators
correlation_cols = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis', 
                   '% Above Overall Mean', 'Suspicion Score', 'Score Range']
corr = indicators_df[correlation_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Between Suspicious Indicators', fontsize=15)
plt.tight_layout()
plt.show()

print("""
## Key Indicators of Potential Corruption/Test Leaks:

1. **Unusually High Mean Scores:** Regions with average scores significantly above the overall mean
2. **Score Clustering:** High percentage of identical scores or abnormal distribution patterns
3. **Negative Skewness:** Distribution heavily weighted toward high scores
4. **Low Standard Deviation:** Unusually consistent scores may indicate shared answers
5. **High Kurtosis:** Sharp peaks in score distribution could indicate orchestrated results
6. **High Percentage Above Mean:** Regions with disproportionate number of candidates scoring above average

## Recommendations for Further Investigation:

1. Compare with historical data to identify sudden shifts in performance
2. Examine answer patterns (if available) for suspicious similarities
3. Scrutinize test administration procedures in high-suspicion regions
4. Consider independent verification or retesting in suspicious regions
5. Interview proctors and administrators from flagged regions
""")
```

This analysis provides comprehensive statistical tools to identify potential test irregularities across different regions for the MEDICINA profession. The suspicion scores combine multiple statistical indicators to highlight regions that may warrant further investigation.