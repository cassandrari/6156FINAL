import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
df = pd.read_csv('Superstore_Sales.csv',  encoding='ISO-8859-1') 
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Ship_Date'] = pd.to_datetime(df['Ship_Date'], errors='coerce')



df.fillna(0, inplace=True)

# Function to create sales overview plots
def overview_of_sales():
    overall_sales = df.groupby('Order_Date')['Sales'].sum().reset_index()
    overall_sales['YearMonth'] = overall_sales['Order_Date'].dt.to_period('M')

    # Plot overall sales trend
    fig1 = px.bar(overall_sales, x='YearMonth', y='Sales', labels={'YearMonth': 'Month', 'Sales': 'Total Sales'})

    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    fig2 = px.bar(region_sales, x='Region', y='Sales', labels={'Region': 'Region', 'Sales': 'Total Sales'})

    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig3 = px.pie(category_sales, names='Category', values='Sales', title='Sales by Category')

    return fig1, fig2, fig3

# Function to predict sales based on input features
def profit_prediction(year, month, discount, category, subcategory, region):
    # Prepare the data
    df['Month'] = df['Order_Date'].dt.month
    df['Year'] = df['Order_Date'].dt.year

    # Ensure category, subcategory, and region are encoded as numbers
    label_encoder = LabelEncoder()
    df['Category_encoded'] = label_encoder.fit_transform(df['Category'])
    df['Subcategory_encoded'] = label_encoder.fit_transform(df['Subcategory'])
    df['Region_encoded'] = label_encoder.fit_transform(df['Region'])

    # Prepare features (including new variables) and target (Profit)
    X = df[['Year', 'Month', 'Discount', 'Category_encoded', 'Subcategory_encoded', 'Region_encoded']]  # Features
    y = df['Profit']  # Target (Profit)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Encode the input data (year, month, discount, category, subcategory, region) for prediction
    category_encoded = label_encoder.transform([category])[0]
    subcategory_encoded = label_encoder.transform([subcategory])[0]
    region_encoded = label_encoder.transform([region])[0]
    
    # Prepare the input data for prediction
    input_data = np.array([[year, month, discount, category_encoded, subcategory_encoded, region_encoded]])

    # Make prediction
    predicted_profit = model.predict(input_data)[0]

    return f"Predicted Profit for {category} - {subcategory} in {region} for {month}/{year} with {discount}% discount: ${predicted_profit:.2f}"


# Function to analyze discount impact
def discount_impact_analysis():
    fig1 = px.line(df, x='Discount', y='Sales', color='Category', title="Discount vs Sales")
    fig2 = px.line(df, x='Discount', y='Profit', color='Category', title="Discount vs Profit")

    # Return the figures
    return fig1, fig2

# Sidebar navigation
st.sidebar.title('Sales Dashboard')
page = st.sidebar.radio("Select a page", ["Overview of Sales", "Sales Prediction", "Discount Impact"])

# Page for Overview of Sales
if page == "Overview of Sales":
    st.title("Overview of Sales")
    fig1, fig2, fig3 = overview_of_sales()
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)

# Page for Sales Prediction
elif page == "Sales Prediction":
    st.title("Sales Prediction")
    
    # Input features for prediction
    year_input = st.slider("Select Year", 2020, 2025, 2023)
    month_input = st.slider("Select Month", 1, 12, 1)
    discount_input = st.slider("Select Discount (%)", 0, 50, 10)
    
    prediction_output = sales_prediction(year_input, month_input, discount_input)
    st.write(prediction_output)

# Page for Discount Impact Analysis
elif page == "Discount Impact":
    st.title("Discount Impact Analysis")
    fig1_impact, fig2_impact = discount_impact_analysis()
    st.plotly_chart(fig1_impact)
    st.plotly_chart(fig2_impact)
