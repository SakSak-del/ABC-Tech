import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="ABC Tech Incident Management", layout="wide")

# Load the dataset from CSV
@st.cache_data
def load_data():
    return pd.read_csv("dataset_list.csv")

data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "EDA", "Correlation Heatmap"])

# Dataset Overview
if page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write("First 10 rows of the ABC Tech dataset")
    st.dataframe(data.head(10))
    st.write("Shape of dataset:", data.shape)
    st.write("Data types:", data.dtypes)

# EDA
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Priority Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Priority', palette="viridis", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Impact Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Impact', palette="magma", ax=ax)
        st.pyplot(fig)

    st.subheader("Urgency Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='Urgency', palette="coolwarm", ax=ax)
    st.pyplot(fig)

# Correlation Heatmap
elif page == "Correlation Heatmap":
    st.title("Correlation Heatmap")
    st.write("Correlation of numerical features in the dataset")

    # Convert categorical to numeric temporarily for correlation
    df_numeric = data.copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            df_numeric[col] = pd.factorize(df_numeric[col])[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
