import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import datetime

# Load the dataset
df = pd.read_csv('Superstore_Sales.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'])

# Preprocess data (handle missing values, format columns, etc.)
df.fillna(0, inplace=True)  # Filling missing values with zero, adjust as needed

# Dropdown for selecting the analysis type (Overview of All or a specific machine)
selection = st.selectbox("Select Analysis Type", ["Overview of All", "Sales Analysis by Category", "Sales Prediction", "Discount Impact Analysis"])

if selection == "Overview of All":
    # Overall Sales Analysis
    st.markdown("<h3 style='text-align: center;'>Overall Sales Performance</h3>", unsafe_allow_html=True)
    
    # Aggregating sales by date
    overall_sales = df.groupby('Order_Date')['Sales'].sum().reset_index()
    overall_sales['YearMonth'] = overall_sales['Order_Date'].dt.to_period('M')
    
    # Plotting sales trend over time (line chart)
    fig = px.line(overall_sales, x='YearMonth', y='Sales', labels={'YearMonth': 'Month', 'Sales': 'Total Sales'})
    st.plotly_chart(fig)

    # Aggregating sales by region
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    st.markdown("<h3 style='text-align: center;'>Sales by Region</h3>", unsafe_allow_html=True)
    
    # Plotting sales by region (bar chart)
    fig = px.bar(region_sales, x='Region', y='Sales', labels={'Region': 'Region', 'Sales': 'Total Sales'})
    st.plotly_chart(fig)

    # Sales by Category
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    st.markdown("<h3 style='text-align: center;'>Sales by Product Category</h3>", unsafe_allow_html=True)
    
    # Plotting sales by category (pie chart)
    fig = px.pie(category_sales, names='Category', values='Sales', title='Sales by Category')
    st.plotly_chart(fig)

elif selection == "Sales Analysis by Category":
    # Sales analysis for a specific category
    category = st.selectbox("Select Product Category", df['Category'].unique())
    category_data = df[df['Category'] == category]

    # Aggregating sales by sub-category
    subcategory_sales = category_data.groupby('Sub_Category')['Sales'].sum().reset_index()
    st.markdown(f"<h3 style='text-align: center;'>Sales by Sub-Category in {category}</h3>", unsafe_allow_html=True)
    
    # Plotting sales by sub-category (bar chart)
    fig = px.bar(subcategory_sales, x='Sub_Category', y='Sales', labels={'Sub_Category': 'Sub-Category', 'Sales': 'Total Sales'})
    st.plotly_chart(fig)

elif selection == "Sales Prediction":
    # Sales prediction (predict future sales)
    st.markdown("<h3 style='text-align: center;'>Predict Future Sales</h3>", unsafe_allow_html=True)

    # Create features for sales prediction
    df['YearMonth'] = df['Order_Date'].dt.to_period('M')
    df['Month'] = df['Order_Date'].dt.month
    df['Year'] = df['Order_Date'].dt.year

    # Prepare data for prediction
    X = df[['Year', 'Month', 'Discount']]  # Features (Year, Month, Discount)
    y = df['Sales']  # Target (Sales)

    # Split into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict sales on the test set
    y_pred = model.predict(X_test)

    # Show model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Make predictions for the future (e.g., next 6 months)
    future_dates = pd.date_range(datetime.datetime.now(), periods=6, freq='M')
    future_data = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Discount': [df['Discount'].mean()] * 6  # Assuming average discount for prediction
    })
    future_sales = model.predict(future_data)

    # Display future sales predictions
    st.write("Future Sales Predictions (Next 6 months):")
    for i, sales in enumerate(future_sales):
        st.write(f"Month: {future_dates[i].strftime('%B %Y')} - Predicted Sales: ${sales:.2f}")

elif selection == "Discount Impact Analysis":
    # Analyze the impact of discounts on sales/profit
    st.markdown("<h3 style='text-align: center;'>Impact of Discounts on Sales and Profit</h3>", unsafe_allow_html=True)

    # Create a scatter plot to visualize discount vs sales
    fig = px.scatter(df, x='Discount', y='Sales', color='Category', title="Discount vs Sales", labels={'Discount': 'Discount (%)', 'Sales': 'Sales ($)'})
    st.plotly_chart(fig)

    # Create a scatter plot to visualize discount vs profit
    fig = px.scatter(df, x='Discount', y='Profit', color='Category', title="Discount vs Profit", labels={'Discount': 'Discount (%)', 'Profit': 'Profit ($)'})
    st.plotly_chart(fig)

    # Correlation matrix between discount, sales, and profit
    corr = df[['Discount', 'Sales', 'Profit']].corr()
    st.write("Correlation Matrix (Discount, Sales, Profit):")
    st.table(corr)

else:
    st.write("Please select a valid analysis type.")


