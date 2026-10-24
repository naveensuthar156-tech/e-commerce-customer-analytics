import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt

# ==========================================
# AUTOMATIC VISUAL & DASHBOARD SAVER
# ==========================================

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Counter for automatic file names
figure_counter = 0

# Original plt.show()
_original_show = plt.show


def automatic_save_show(*args, **kwargs):
    global figure_counter

    # Get all currently open figures
    figures = plt.get_fignums()

    for fig_num in figures:
        figure_counter += 1

        fig = plt.figure(fig_num)

        # Save automatically in the same project folder
        file_path = os.path.join(
            PROJECT_FOLDER,
            f"visual_{figure_counter}.png"
        )

        fig.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved: {file_path}")

    # Show figure normally
    _original_show(*args, **kwargs)


# Replace matplotlib's show()
plt.show = automatic_save_show

df = pd.read_csv("dataset.csv")

print(df.head())

print("\n")

print("step1:-----")

print("\n")
print("Dataset understanding:-----")
# Phase 1: Data Understanding

# 1. Dataset ka size
print("Rows and Columns:", df.shape)

# 2. First 5 records
print("\nFirst 5 Records:")
print(df.head())

# 3. Last 5 records
print("\nLast 5 Records:")
print(df.tail())

# 4. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# 5. Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary of numerical columns
print("Statistical Summary:")
print(df.describe())

# Check missing values
print("Missing Values:")
missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

# Missing value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Percentage": missing_percentage
})

print(missing_table[missing_table["Missing Values"] > 0])

# Check duplicate rows
duplicate_count = df.duplicated().sum()

print("Number of Duplicate Rows:", duplicate_count)

duplicates = df[df.duplicated(keep=False)]

print(duplicates)


# Unique values in categorical columns

categorical_columns = [
    "Gender",
    "State",
    "City",
    "Product_Category",
    "Product",
    "Payment_Method"
]

for column in categorical_columns:
    print(f"\n{column}")
    print("Unique Values:", df[column].nunique())
    print(df[column].unique()[:20])
    
    # Check data types
print("Data Types:")
print(df.dtypes)


print("\n")
print("step2:-----")

print("\n")
print("Data Cleaning:-----")


# Check missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Fill missing Payment_Method values with mode
df["Payment_Method"] = df["Payment_Method"].fillna(
    df["Payment_Method"].mode()[0]
)

print("Missing values after cleaning:")
print(df.isnull().sum())

# Check duplicate rows
print("Duplicate rows before cleaning:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("Duplicate rows after cleaning:", df.duplicated().sum())

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

print(df["Order_Date"].dtype)

numeric_columns = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Total_Amount"
]

print("Numerical columns:")
print(df[numeric_columns].describe())

# Check invalid negative values

print("Negative Quantity:", (df["Quantity"] < 0).sum())
print("Negative Unit Price:", (df["Unit_Price"] < 0).sum())
print("Negative Discount:", (df["Discount"] < 0).sum())
print("Negative Total Amount:", (df["Total_Amount"] < 0).sum())

# Check invalid age values
print("Age below 0:", (df["Age"] < 0).sum())
print("Age above 100:", (df["Age"] > 100).sum())


##invalid age records
invalid_age = df[(df["Age"] < 0) | (df["Age"] > 100)]

print("Invalid Age Records:")
print(invalid_age)

##discount Validation
print("Minimum Discount:", df["Discount"].min())
print("Maximum Discount:", df["Discount"].max())

##Data Consistency Check
print("Final Missing Values:")
print(df.isnull().sum())
## final duplicate check
print("Final Duplicate Rows:", df.duplicated().sum())

###clean data information
print("Clean Dataset Shape:", df.shape)

print("\nClean Dataset Information:")
df.info()

# Save cleaned dataset
df.to_csv("cleaned_ecommerce_customer_analytics.csv", index=False)

print("Cleaned dataset saved successfully!")

print("\n")
# Phase 3: Exploratory Data Analysis (EDA)

print("\n")
print("step3:-----")
print("Exploratory Data Analysis :------")
print("===== BUSINESS SUMMARY =====")

print("Total Orders:", df["Order_ID"].nunique())
print("Total Customers:", df["Customer_ID"].nunique())
print("Total Sales: ₹", round(df["Total_Amount"].sum(), 2))
print("Total Quantity Sold:", df["Quantity"].sum())
print("Average Order Value: ₹", round(df["Total_Amount"].mean(), 2))

####Gender Analysis
print("Gender Analysis:----")
gender_analysis = df.groupby("Gender").agg(
    Orders=("Order_ID", "nunique"),
    Customers=("Customer_ID", "nunique"),
    Quantity=("Quantity", "sum"),
    Sales=("Total_Amount", "sum")
).reset_index()

gender_analysis["Sales"] = gender_analysis["Sales"].round(2)

print(gender_analysis)


###visualization  of Gender Analysis

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.bar(
    gender_analysis["Gender"],
    gender_analysis["Sales"]
)

plt.title("Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Sales")

plt.show()


##Age Group Analysis
print("Age Group Analysis:----")
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 25, 35, 45, 55, 100],
    labels=[
        "Below 18",
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56+"
    ]
)

age_analysis = df.groupby("Age_Group", observed=True).agg(
    Customers=("Customer_ID", "nunique"),
    Orders=("Order_ID", "nunique"),
    Sales=("Total_Amount", "sum")
).reset_index()

print(age_analysis)

#visualization of Age Group Analysis

plt.figure(figsize=(9, 5))

plt.bar(
    age_analysis["Age_Group"].astype(str),
    age_analysis["Sales"]
)

plt.title("Sales by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Total Sales")

plt.xticks(rotation=30)

plt.show()

#####product category analysis
print("Product Category Analysis:----")
category_analysis = df.groupby("Product_Category").agg(
    Orders=("Order_ID", "nunique"),
    Quantity=("Quantity", "sum"),
    Sales=("Total_Amount", "sum")
).reset_index()

category_analysis = category_analysis.sort_values(
    "Sales",
    ascending=False
)

print(category_analysis)

#visualization of Product Category Analysis

plt.figure(figsize=(10, 5))

plt.bar(
    category_analysis["Product_Category"],
    category_analysis["Sales"]
)

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.show()


##top 10 products analysis
print("Top 10 Products Analysis:----")
product_analysis = df.groupby("Product").agg(
    Orders=("Order_ID", "nunique"),
    Quantity=("Quantity", "sum"),
    Sales=("Total_Amount", "sum")
).reset_index()

top_10_products = product_analysis.sort_values(
    "Sales",
    ascending=False
).head(10)

print(top_10_products)

#visualization of Top 10 Products Analysis

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_products["Product"],
    top_10_products["Sales"]
)

plt.title("Top 10 Products by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product")

plt.gca().invert_yaxis()

plt.show()


##state wise sales analysis
print
state_analysis = df.groupby("State").agg(
    Orders=("Order_ID", "nunique"),
    Customers=("Customer_ID", "nunique"),
    Quantity=("Quantity", "sum"),
    Sales=("Total_Amount", "sum")
).reset_index()

state_analysis = state_analysis.sort_values(
    "Sales",
    ascending=False
)

print(state_analysis)
####top 10 states by sales
print("Top 10 States by Sales:----")
top_states = state_analysis.head(10)

print(top_states)


####visualization of Top 10 States by Sales

plt.figure(figsize=(10, 6))

plt.bar(
    top_states["State"],
    top_states["Sales"]
)

plt.title("Top 10 States by Sales") 
plt.xlabel("State")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.show()

#### city wise sales analysis
print("City Wise Sales Analysis:----")
city_analysis = df.groupby("City").agg(
    Orders=("Order_ID", "nunique"),
    Customers=("Customer_ID", "nunique"),
    Sales=("Total_Amount", "sum")
).reset_index()

top_cities = city_analysis.sort_values(
    "Sales",
    ascending=False
).head(10)

print(top_cities)


####visualization of Top 10 Cities by Sales

plt.figure(figsize=(10, 6))

plt.bar(
    top_cities["City"],
    top_cities["Sales"]
)

plt.title("Top 10 Cities by Sales")
plt.xlabel("City")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.show()

####Payment Method Analysis
print("Payment Method Analysis:----")
payment_analysis = df.groupby("Payment_Method").agg(
    Orders=("Order_ID", "nunique"),
    Customers=("Customer_ID", "nunique"),
    Sales=("Total_Amount", "sum")
).reset_index()

payment_analysis = payment_analysis.sort_values(
    "Sales",
    ascending=False
)

print(payment_analysis)


####visualization of Payment Method Analysis

plt.figure(figsize=(9, 5))

plt.bar(
    payment_analysis["Payment_Method"],
    payment_analysis["Sales"]
)

plt.title("Sales by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Total Sales")

plt.xticks(rotation=30)

plt.show()

####Monthly Sales Analysis
print("Monthly Sales Analysis:----")
monthly_sales = df.groupby(
    df["Order_Date"].dt.to_period("M")
)["Total_Amount"].sum().reset_index()

monthly_sales["Order_Date"] = monthly_sales["Order_Date"].astype(str)


print(monthly_sales)

####visualization of Monthly Sales Analysis

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_sales["Order_Date"],
    monthly_sales["Total_Amount"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.show()

####Discount Analysis
print("Discount Analysis:----")
discount_analysis = df.groupby("Discount").agg(
    Orders=("Order_ID", "nunique"),
    Quantity=("Quantity", "sum"),
    Sales=("Total_Amount", "sum")
).reset_index()

print(discount_analysis)

####visualization of Discount Analysis discount vs total amount

plt.figure(figsize=(9, 5))

plt.scatter(
    df["Discount"],
    df["Total_Amount"]
)

plt.title("Discount vs Total Amount")
plt.xlabel("Discount")
plt.ylabel("Total Amount")

plt.show()

###quality check for discount vs total amount
plt.figure(figsize=(9, 5))

plt.scatter(
    df["Quantity"],
    df["Total_Amount"]
)

plt.title("Quantity vs Total Amount")
plt.xlabel("Quantity")
plt.ylabel("Total Amount")

plt.show()


###Correlation Analysis
print("Correlation Analysis:----")
numeric_columns = [
    "Age",
    "Quantity",
    "Unit_Price",
    "Discount",
    "Total_Amount"
]

correlation = df[numeric_columns].corr()

print(correlation)

####visualization of Correlation Heatmap

import seaborn as sns

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


print("\n")
print("step4:-----")
print("\n")
print("Customer Segmentation:-----")

customer_summary = df.groupby("Customer_ID").agg(
    Total_Orders=("Order_ID", "nunique"),
    Total_Quantity=("Quantity", "sum"),
    Total_Spending=("Total_Amount", "sum"),
    Average_Order_Value=("Total_Amount", "mean"),
    Last_Purchase=("Order_Date", "max")
).reset_index()

print(customer_summary.head())

print("reference data")
# Reference date
reference_date = df["Order_Date"].max()

print("Reference Date:", reference_date)

customer_summary["Recency"] = (
    reference_date - customer_summary["Last_Purchase"]
).dt.days

print(
    customer_summary[
        ["Customer_ID", "Last_Purchase", "Recency"]
    ].head(10)
)

####rfm segmentation
print("RFM  table---")
rfm = customer_summary[
    [
        "Customer_ID",
        "Recency",
        "Total_Orders",
        "Total_Spending"
    ]
].copy()

rfm = rfm.rename(columns={
    "Total_Orders": "Frequency",
    "Total_Spending": "Monetary"
})

print(rfm.head())


###rfm scoring
print("RFM Scoring:----")
# Recency score
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

# Frequency score
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

# Monetary score
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

print(rfm.head())


###rfm overall score
print("RFM Overall Score:----")
rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str) +
    rfm["F_Score"].astype(str) +
    rfm["M_Score"].astype(str)
)

print(
    rfm.sort_values(
        ["R_Score", "F_Score", "M_Score"],
        ascending=False
    ).head(10)
)


#####customer segmentation 
print("Customer Segmentation:-----")
def customer_segment(row):
    
    if row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
        return "Champions"

    elif row["R_Score"] >= 4 and row["F_Score"] >= 3:
        return "Loyal Customers"

    elif row["R_Score"] >= 4 and row["F_Score"] <= 2:
        return "New Customers"

    elif row["R_Score"] <= 2 and row["F_Score"] >= 3 and row["M_Score"] >= 3:
        return "At Risk"

    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "Lost Customers"

    else:
        return "Regular Customers"


rfm["Customer_Segment"] = rfm.apply(
    customer_segment,
    axis=1
)

print(rfm.head(10))


#####segment count
print("Customer Segment Count:----")
segment_count = (
    rfm["Customer_Segment"]
    .value_counts()
    .reset_index()
)

segment_count.columns = [
    "Customer_Segment",
    "Customer_Count"
]

print(segment_count)


### visualization of Customer Segment Count

plt.figure(figsize=(10, 5))

plt.bar(
    segment_count["Customer_Segment"],
    segment_count["Customer_Count"]
)

plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

###segmentation wise revenue analysis
print("Segmentation Wise Revenue Analysis:----")
segment_revenue = (
    rfm.groupby("Customer_Segment")["Monetary"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)
print(segment_revenue)

#### visualization of Segmentation Wise Revenue Analysis
plt.figure(figsize=(10, 5))

plt.bar(
    segment_revenue["Customer_Segment"],
    segment_revenue["Monetary"]
)

plt.title("Revenue by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Revenue")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


###top 10 valuable customers
print("Top 10 Valuable Customers:----")
top_customers = rfm.sort_values(
    "Monetary",
    ascending=False
).head(10)

print(top_customers)


####top customers by frequency
print("Top  Customers by Frequency:----")
frequent_customers = rfm.sort_values(
    "Frequency",
    ascending=False
).head(10)

print(frequent_customers)


#### final customer segmentation table
print("Final Customer Segmentation Table:----")
final_customer_analysis = rfm[
    [
        "Customer_ID",
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFM_Score",
        "Customer_Segment"
    ]
].copy()

print(final_customer_analysis.head(10))

###save customer analysis 
print("Saving Final Customer Segmentation Table to CSV...")
final_customer_analysis.to_csv(
    "customer_segmentation_analysis.csv",
    index=False
)

print("Customer segmentation file saved successfully!")


print("\n")
print("step5:-----")
print("\n")
print("Business Insights & Recommendations:----- ")
# Category-wise Sales
category_sales = df.groupby("Product_Category")["Total_Amount"].sum()


best_category = category_sales.idxmax()
best_category_sales = category_sales.max()

print("Best Product Category:", best_category)
print("Sales:", round(best_category_sales, 2))

###low Performing Product Category
lowest_category = category_sales.idxmin()
lowest_category_sales = category_sales.min()

print("Lowest Performing Category:", lowest_category)
print("Sales:", round(lowest_category_sales, 2))

###top products selling product
top_products = df.groupby("Product")["Total_Amount"].sum().sort_values(ascending=False)
best_product = top_products.index[0]
best_product_sales = top_products.iloc[0]

print("Top Product:", best_product)
print("Sales:", round(best_product_sales, 2))

###best performing state
state_sales = df.groupby("State")["Total_Amount"].sum().sort_values(ascending=False)
best_state = state_sales.idxmax()
best_state_sales = state_sales.max()

print("Best Performing State:", best_state)
print("Sales:", round(best_state_sales, 2))

##best performing city

best_city = city_analysis.loc[
    city_analysis["Sales"].idxmax(), "City"
]

best_city_sales = city_analysis["Sales"].max()

print("Best Performing City:", best_city)
print("Sales:", round(best_city_sales, 2))


####most popular payment method

payment_orders = (
    df.groupby("Payment_Method")["Order_ID"]
    .nunique()
    .sort_values(ascending=False)
)

popular_payment = payment_orders.idxmax()

print("Most Popular Payment Method:", popular_payment)
print("Number of Orders:", payment_orders.max())

##most valuable customer


best_segment = segment_revenue.loc[
    segment_revenue["Monetary"].idxmax(),
    "Customer_Segment"
]

best_segment_revenue = segment_revenue["Monetary"].max()

print("Most Valuable Customer Segment:", best_segment)
print("Revenue:", round(best_segment_revenue, 2))

### at risk customers

at_risk_customers = rfm[
    rfm["Customer_Segment"] == "At Risk"
]

print("Number of At-Risk Customers:", len(at_risk_customers))

print(
    at_risk_customers.sort_values(
        "Monetary",
        ascending=False
    ).head(10)
)

##champions customers
champions = rfm[
    rfm["Customer_Segment"] == "Champions"
]

print("Number of Champion Customers:", len(champions))

print(
    champions.sort_values(
        "Monetary",
        ascending=False
    ).head(10)
)

##monthly best sales month
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

# Monthly sales
monthly_sales = (
    df.groupby(df["Order_Date"].dt.to_period("M"))["Total_Amount"]
    .sum()
)
best_month = monthly_sales.idxmax()
best_month_sales = monthly_sales.max()

print("Best Sales Month:", best_month)
print("Sales:", round(best_month_sales, 2))

###monthly worst sales month
lowest_month = monthly_sales.idxmin()
lowest_month_sales = monthly_sales.min()

print("Lowest Sales Month:", lowest_month)
print("Sales:", round(lowest_month_sales, 2))

## discount analysis insights
average_discount = df["Discount"].mean()

print("Average Discount:", round(average_discount, 2))
print("Maximum Discount:", df["Discount"].max())


##final business insights and recommendations
# 5.12 Final Business Insights
print("final Business Insights:----")
# Basic Business Metrics

total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
total_sales = df["Total_Amount"].sum()
average_order_value = df["Total_Amount"].mean()
business_insights = pd.DataFrame({

    "Metric": [
        "Total Orders",
        "Total Customers",
        "Total Sales",
        "Average Order Value",
        "Best Product Category",
        "Best Product",
        "Best State",
        "Best City",
        "Most Popular Payment Method",
        "Most Valuable Customer Segment",
        "Best Sales Month",
        "At-Risk Customers"
    ],

    "Result": [
        total_orders,
        total_customers,
        round(total_sales, 2),
        round(average_order_value, 2),
        best_category,
        best_product,
        best_state,
        best_city,
        popular_payment,
        best_segment,
        str(best_month),
        len(at_risk_customers)
    ]

})

print(business_insights)


print("""
===== BUSINESS RECOMMENDATIONS =====

1. Focus on the highest-performing product category
   and maintain sufficient inventory for popular products.

2. Promote top-selling products through targeted offers
   and personalized recommendations.

3. Focus marketing campaigns on high-performing states
   and cities to maximize sales.

4. Encourage customers to use the most popular payment
   method while continuing to support other payment options.

5. Provide loyalty rewards and exclusive offers to
   Champions and Loyal Customers.

6. Target At-Risk Customers with personalized discounts,
   reminders, and re-engagement campaigns.

7. Analyze the best-performing months and prepare inventory
   and marketing campaigns before peak sales periods.

8. Monitor discounts carefully to ensure that higher sales
   do not reduce overall business profitability.
""")


####

print("\n")
print("step6:-----")
print("\n")
print("Dashboard Creation:-----")

# Make sure Order_Date is datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

# ==============================
# KPI CALCULATIONS
# ==============================

total_sales = df["Total_Amount"].sum()
total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
total_quantity = df["Quantity"].sum()
average_order_value = df["Total_Amount"].mean()

print("========== KEY PERFORMANCE INDICATORS ==========")
print("Total Sales:", round(total_sales, 2))
print("Total Orders:", total_orders)
print("Total Customers:", total_customers)
print("Total Quantity:", total_quantity)
print("Average Order Value:", round(average_order_value, 2))


# ==============================
# CATEGORY SALES
# ==============================

category_sales = (
    df.groupby("Product_Category")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
)


# ==============================
# TOP 10 PRODUCTS
# ==============================

top_products = (
    df.groupby("Product")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)


# ==============================
# GENDER SALES
# ==============================

gender_sales = (
    df.groupby("Gender")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
)


# ==============================
# TOP 10 STATES
# ==============================

state_sales = (
    df.groupby("State")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)


# ==============================
# PAYMENT METHOD
# ==============================

payment_sales = (
    df.groupby("Payment_Method")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
)


# ==============================
# MONTHLY SALES
# ==============================

monthly_sales = (
    df.groupby(
        df["Order_Date"].dt.to_period("M")
    )["Total_Amount"]
    .sum()
)

monthly_sales.index = monthly_sales.index.astype(str)

print("\nDashboard data prepared successfully!")


### create dashboard

print("creating dashboard visualization...")

fig = plt.figure(figsize=(18, 12))

fig.suptitle(
    "E-Commerce Customer Analytics Dashboard",
    fontsize=22,
    fontweight="bold"
)


# ==========================================
# KPI 1 - TOTAL SALES
# ==========================================

ax1 = plt.subplot(3, 3, 1)

ax1.text(
    0.5, 0.5,
    f"₹{total_sales:,.0f}",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

ax1.text(
    0.5, 0.20,
    "Total Sales",
    ha="center",
    fontsize=13
)

ax1.axis("off")


# ==========================================
# KPI 2 - TOTAL ORDERS
# ==========================================

ax2 = plt.subplot(3, 3, 2)

ax2.text(
    0.5, 0.5,
    f"{total_orders:,}",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

ax2.text(
    0.5, 0.20,
    "Total Orders",
    ha="center",
    fontsize=13
)

ax2.axis("off")


# ==========================================
# KPI 3 - CUSTOMERS
# ==========================================

ax3 = plt.subplot(3, 3, 3)

ax3.text(
    0.5, 0.5,
    f"{total_customers:,}",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

ax3.text(
    0.5, 0.20,
    "Total Customers",
    ha="center",
    fontsize=13
)

ax3.axis("off")


# ==========================================
# KPI 4 - QUANTITY
# ==========================================

ax4 = plt.subplot(3, 3, 4)

ax4.text(
    0.5, 0.5,
    f"{total_quantity:,}",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

ax4.text(
    0.5, 0.20,
    "Total Quantity Sold",
    ha="center",
    fontsize=13
)

ax4.axis("off")


# ==========================================
# KPI 5 - AVERAGE ORDER VALUE
# ==========================================

ax5 = plt.subplot(3, 3, 5)

ax5.text(
    0.5, 0.5,
    f"₹{average_order_value:,.0f}",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold"
)

ax5.text(
    0.5, 0.20,
    "Average Order Value",
    ha="center",
    fontsize=13
)

ax5.axis("off")


# ==========================================
# CATEGORY SALES
# ==========================================

ax6 = plt.subplot(3, 3, 6)

category_sales.plot(
    kind="bar",
    ax=ax6
)

ax6.set_title("Sales by Product Category")
ax6.set_xlabel("Category")
ax6.set_ylabel("Sales")

plt.xticks(rotation=45)


# ==========================================
# TOP PRODUCTS
# ==========================================

ax7 = plt.subplot(3, 3, 7)

top_products.sort_values().plot(
    kind="barh",
    ax=ax7
)

ax7.set_title("Top 10 Products")
ax7.set_xlabel("Sales")
ax7.set_ylabel("Product")


# ==========================================
# PAYMENT METHOD
# ==========================================

ax8 = plt.subplot(3, 3, 8)

payment_sales.plot(
    kind="bar",
    ax=ax8
)

ax8.set_title("Sales by Payment Method")
ax8.set_xlabel("Payment Method")
ax8.set_ylabel("Sales")

plt.xticks(rotation=45)


# ==========================================
# MONTHLY SALES
# ==========================================

ax9 = plt.subplot(3, 3, 9)

monthly_sales.plot(
    kind="line",
    marker="o",
    ax=ax9
)

ax9.set_title("Monthly Sales Trend")
ax9.set_xlabel("Month")
ax9.set_ylabel("Sales")

plt.xticks(rotation=45)


plt.tight_layout(rect=[0, 0, 1, 0.95])


# Save dashboard

plt.savefig(
    "ecommerce_customer_analytics_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
