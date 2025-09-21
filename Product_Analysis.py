# --------------------------------------------------------------------------------
# Product Performance Analysis
# A Python-based data analytics project for evaluating product performance.
# --------------------------------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# --------------------------------------------------------------------------------
# Step 1: Dataset Loading
# --------------------------------------------------------------------------------

df_raw = pd.read_csv('data/product-sales.csv')
# --------------------------------------------------------------------------------
# Step 2: Data Cleaning & Preprocessing
# --------------------------------------------------------------------------------
print("\n" + "="*50)
print("Step 2: Data Cleaning & Preprocessing")
print("="*50)

# Make a copy to preserve the raw data
df = df_raw.copy()

# A. Handle Missing Values
# For numerical columns, fill with the median to avoid skew from outliers
for col in ['Units Sold', 'Customer Rating']:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    
# For categorical columns, fill with a placeholder 'Unknown'
df['Category'].fillna('Unknown', inplace=True)

# B. Correct Data Types (if necessary, though our synthetic data is mostly correct)
# This is a good practice for real datasets
df['Product ID'] = df['Product ID'].astype('string')

# C. Calculate Key Business Metrics
df['Total Revenue'] = df['Units Sold'] * df['Unit Price']
df['Total Profit'] = df['Units Sold'] * (df['Unit Price'] - df['Cost Per Unit'])
df['Profit Margin (%)'] = (df['Total Profit'] / df['Total Revenue']) * 100
df['Inventory Turnover'] = df['Units Sold'] / df['Inventory']

print("\nDataFrame Info (After Cleaning & Feature Engineering):")
df.info()
print("\nFirst 5 rows after cleaning and metric calculation:")
print(df.head())


# --------------------------------------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------------
print("\n" + "="*50)
print("Step 3: Exploratory Data Analysis (EDA)")
print("="*50)

# A. Statistical Summary of Key Metrics
print("\nStatistical Summary of Key Metrics:")
print(df[['Total Revenue', 'Total Profit', 'Customer Rating', 'Inventory Turnover']].describe().round(2))

# B. Top 10 Best and Worst Performing Products
print("\n--- Top 10 Products by Total Revenue ---")
top_revenue = df.sort_values('Total Revenue', ascending=False).head(10)
print(top_revenue[['Product Name', 'Category', 'Total Revenue', 'Total Profit']].round(2))

print("\n--- Bottom 10 Products by Total Profit ---")
bottom_profit = df.sort_values('Total Profit', ascending=True).head(10)
print(bottom_profit[['Product Name', 'Category', 'Total Profit', 'Units Sold']].round(2))

# C. Category-wise Performance
print("\n--- Category-wise Total Revenue and Profit ---")
category_performance = df.groupby('Category').agg(
    Total_Revenue=('Total Revenue', 'sum'),
    Total_Profit=('Total Profit', 'sum'),
    Avg_Rating=('Customer Rating', 'mean')
).sort_values('Total_Revenue', ascending=False).round(2)
print(category_performance)

# --------------------------------------------------------------------------------
# Step 4: Interactive Visualization with Plotly
# --------------------------------------------------------------------------------
print("\n" + "="*50)
print("Step 4: Interactive Visualization with Plotly")
print("="*50)

# A. Bar Chart: Total Revenue by Category
print("Generating interactive bar chart: Total Revenue by Category...")
fig_revenue = px.bar(
    category_performance,
    x=category_performance.index,
    y='Total_Revenue',
    title='Total Revenue by Product Category',
    labels={'x': 'Product Category', 'y': 'Total Revenue ($)'},
    hover_data={'Total_Profit': True}
)
fig_revenue.show()

# B. Scatter Plot: Profit vs. Units Sold
print("\nGenerating interactive scatter plot: Profit vs. Units Sold...")
fig_scatter = px.scatter(
    df,
    x='Units Sold',
    y='Total Profit',
    color='Category',
    hover_name='Product Name',
    size='Total Revenue',
    title='Profit vs. Units Sold with Revenue and Category Insights',
    labels={'Units Sold': 'Units Sold', 'Total Profit': 'Total Profit ($)'}
)
fig_scatter.show()

# C. Sunburst Chart: Hierarchical View of Revenue by Category and Subcategory
print("\nGenerating interactive sunburst chart: Revenue by Category/Subcategory...")
fig_sunburst = px.sunburst(
    df,
    path=['Category', 'Subcategory', 'Product Name'],
    values='Total Revenue',
    title='Hierarchical Revenue Breakdown'
)
fig_sunburst.show()

# --------------------------------------------------------------------------------
# Step 5: Summary Report & Actionable Insights
# --------------------------------------------------------------------------------
print("\n" + "="*50)
print("Step 5: Summary Report & Actionable Insights")
print("="*50)

print("\n--- BUSINESS INSIGHTS SUMMARY ---")
print("Based on the analysis of product performance metrics, here are key insights:")

# 1. High-Level Performance Overview
print(f"\n- Total products analyzed: {df.shape[0]}")
print(f"- Total Revenue across all products: ${df['Total Revenue'].sum():.2f}")
print(f"- Total Profit across all products: ${df['Total Profit'].sum():.2f}")

# 2. Key Category Drivers
top_category = category_performance.iloc[0]
print(f"\n- **Top Performing Category:** The **'{top_category.name}'** category is a significant revenue driver, accounting for ${top_category['Total_Revenue']:.2f} in sales.")

# 3. Product-Level Deep Dive
top_product = top_revenue.iloc[0]
bottom_product = bottom_profit.iloc[0]
print(f"- **Top Product:** **'{top_product['Product Name']}'** is the highest-grossing product, with a total revenue of ${top_product['Total Revenue']:.2f}.")
print(f"- **Underperforming Product:** **'{bottom_product['Product Name']}'** is in the bottom 10 for total profit, suggesting a potential issue with high costs or low sales volume. This product may require a review of its pricing strategy or inventory.")

# 4. Inventory Health Check
avg_turnover = df['Inventory Turnover'].mean()
print(f"\n- **Inventory Health:** The average inventory turnover is approximately {avg_turnover:.2f}, indicating that on average, stock is sold and replaced about this many times. Products with a very low turnover might be overstocked.")

# 5. Customer Feedback
avg_rating_overall = df['Customer Rating'].mean()
print(f"- **Customer Satisfaction:** The average customer rating across all products is **{avg_rating_overall:.2f}** out of 5. This is a solid baseline for customer sentiment.")

print("\n--- RECOMMENDATIONS ---")
print("1. **Focus on Top Performers:** Double down on marketing and supply chain efforts for products in the 'Electronics' and 'Apparel' categories.")
print("2. **Investigate Underperformers:** Conduct a root-cause analysis for the bottom-performing products to understand if it's a pricing, quality, or demand issue.")
print("3. **Optimize Inventory:** Analyze products with low inventory turnover to consider markdowns or a reduction in future stock orders to free up capital.")


```

