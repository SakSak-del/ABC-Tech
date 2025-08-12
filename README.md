ABC Tech - Incident Management Optimization
📌 Project Overview
ABC Tech is a mid-size IT-enabled organization handling 22k–25k incidents/tickets annually.
While following ITIL best practices, recent surveys indicate poor incident management ratings.

This project uses Machine Learning & Time Series Forecasting to:

Predict high-priority (P1/P2) tickets

Forecast incident volumes for better resource allocation

Auto-tag tickets to correct departments

Predict RFC (Request for Change) failures

🚀 Features
EDA Dashboard – Interactive visualizations & data exploration

ML Models – RandomForest & XGBoost for priority prediction

Time Series Analysis – ARIMA & Prophet for forecasting incident volumes

Streamlit UI – Easy-to-use web interface

📂 Dataset
The dataset (dataset_list.csv) contains ITSM incident records with attributes like:

Incident ID

Open Time

Impact

Urgency

Priority

CI Category, Subcategory, WBS, KB Number

Closure Code

🛠️ Tech Stack
Frontend: Streamlit

Data Analysis: Pandas, NumPy, Matplotlib, Seaborn

Machine Learning: scikit-learn, XGBoost

Time Series: statsmodels, Prophet

Deployment: Streamlit Community Cloud

🌐 Deployment
This app is deployed on Streamlit Community Cloud.
Visit: Live Demo

📌 How to Use
Select Show EDA to explore the dataset

Enable Run Model to train and view ML predictions

View Forecasts for future incident volumes
