import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Bike Sharing Analysis", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    day_data = pd.read_csv("day.csv")
    hour_data = pd.read_csv("hour.csv")
    return day_data, hour_data

day_data, hour_data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Visualizations", "Correlation Heatmap"])

# Dataset Overview
if page == "Dataset Overview":
    st.title("Dataset Overview")
    st.subheader("Day Data")
    st.dataframe(day_data.head())
    st.subheader("Hour Data")
    st.dataframe(hour_data.head())

# Visualizations
elif page == "Visualizations":
    st.title("Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Daily Rentals Distribution")
        fig, ax = plt.subplots()
        sns.histplot(day_data['cnt'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Hourly Rentals Distribution")
        fig, ax = plt.subplots()
        sns.histplot(hour_data['cnt'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

# Correlation Heatmap
elif page == "Correlation Heatmap":
    st.title("Correlation Heatmap")
    st.write("Correlation of numerical features with rental counts")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(day_data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
