import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title='ABC Tech Incident Management', layout='wide')
st.title('ABC Tech - Incident Management Optimization')

# Sidebar for navigation and filters
st.sidebar.header('Controls')
show_eda = st.sidebar.checkbox('Show EDA Visualizations', value=True)
show_model = st.sidebar.checkbox('Run Model', value=False)
test_size = st.sidebar.slider('Test Size (for train/test split)', 0.1, 0.5, 0.2, 0.05)

st.markdown(''' Business Case:''')

st.markdown('''ABC Tech is an mid-size organisation operation in IT-enabled business segment over a decade. On an average ABC Tech receives 22-25k IT incidents/tickets , which were handled to best practice ITIL framework with incident management , problem management, change management and configuration management processes. These ITIL practices attained matured process level and a recent audit confirmed that further improvement initiatives may not yield return of investment.
ABC Tech management is looking for ways to improve the incident management process as recent customer survey results shows that incident management is rated as poor.''')

st.markdown(''' Machine Learning as way to improve ITSM processes''')

st.markdown('''1. Predicting High Priority Tickets: To predict priority 1 & 2 tickets, so that they can take preventive measures or fix the       problem before it surfaces.

2. Forecast the incident volume in different fields , quarterly and annual. So that they can be better prepared with resources     and technology planning.

3. Auto tag the tickets with right priorities and right departments so that reassigning and related delay can be reduced.

4. Predict RFC (Request for change) and possible failure / misconfiguration of ITSM assets''')

st.markdown(''' The kernal is based on my solution that i have come up with. ''')

st.markdown('''Step 1 - Importing necessary packages''')

st.markdown(''' Project Overview: ABC Tech - Incident Management Optimization using Machine Learning

**Client:** ABC Tech (PRCL-0012)  
**Industry:** IT-enabled Services  

**Challenge:**  
Although ABC Tech follows ITIL best practices for incident management, customer feedback indicates poor experience.

**Objective:**  
Use Machine Learning to improve ITSM (IT Service Management) efficiency, specifically in:
- Predicting high-priority (P1/P2) incidents
- Forecasting incident volume for planning
- Auto-tagging tickets to correct departments
- Predicting RFC failures/misconfigurations

This notebook focuses on the **first objective: predicting high-priority incidents**.
''')

if show_eda:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sqlalchemy import create_engine
    import pandas_profiling as pp

st.markdown('''Step 2 - Reading Data from SQL Table''')

da = pd.read_sql('show databases',engine)

da

engine1.table_names()

data = pd.read_sql_table('dataset_list',engine1)

st.markdown('''Step 3 - Saving the data as CSV file and loading it again into Pandas ''')

#data.to_csv('D:/final-project/dataset.csv')

#data=pd.read_csv('D:/final-project/itsm/dataset.csv')
data.head()

st.markdown('''Step 4 - Doing Basic Exploration to get info on the Data''')

data.columns

data.shape

data.info()

data.describe()

pp.ProfileReport(data)

st.markdown('''Step 5 - Starting the preprocessing and Cleaning of Data  ''')

st.markdown('''-Below Steps are intutive approach for finding out the solution.''')

# droping the columns which is not required

data.drop(labels='Alert_Status',axis=1,inplace=True)

# Finding out the counts of high and low Priority Tickets where 1 is high and 5 is lowest

from collections import Counter as c

print(c(data.Impact).most_common())
print(c(data.Urgency).most_common())
print(c(data.Priority).most_common())

# Treating the object type values separately

iup = data.loc[:,['Impact','Urgency','Priority']]

iup.head()

iup.Urgency.unique()

iup.Impact.replace(to_replace='NS',value=np.nan,inplace=True)
iup.Priority.replace(to_replace='NA',value=np.nan,inplace=True)
iup.Urgency.replace(to_replace='5 - Very Low',value='5',inplace=True)
#iup.drop(index='NS',inplace=True)
#iup.drop(index='5 - Very Low',inplace=True)

iup.isnull().sum()

iup.fillna(value='0',inplace=True)

iup.head()

iup=iup.astype('float')

# Checking The Correlation of of the Target Varible

iup.corr()

data2 = data[data['Priority']=='2']

data2.head()

col = list(data2.columns)
print(col)

# Finding the important features based on Columns

for i in col:
    print(i)
    print('----------------')
    print(c(data2[i]).most_common(5))
    print()

# Below Feature should be used as feature for modeling

pri_col=['CI_Cat','CI_Subcat','WBS','KB_number','Closure_Code']

data3 = data[data['Priority']=='3']

for i in col:
    print(i)
    print('----------------')
    print(c(data3[i]).most_common(5))
    print()

st.markdown('''Step 6 - Starting the modeling of data based on features''')

if show_model:
    # import basic packages for modeling
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

ttdata= data.loc[:,['CI_Cat','CI_Subcat','WBS','KB_number','Priority']]
ttdata.head()

enc = LabelEncoder()
ttdata.CI_Cat =enc.fit_transform(ttdata.CI_Cat)
ttdata.CI_Subcat =enc.fit_transform(ttdata.CI_Subcat)
ttdata.WBS =enc.fit_transform(ttdata.WBS)
ttdata.KB_number =enc.fit_transform(ttdata.KB_number)
ttdata.Priority = iup.Priority

#ttdata2 = ttdata[ttdata.Priority==2]
ttdata.head()

if show_model:
    X_train,X_test,y_train,y_test=train_test_split(ttdata.iloc[:,:-1],ttdata.iloc[:,-1],test_size=0.3,random_state=10)

st.markdown(''' Fiting the Data into the model and got an accuracy''')

if show_model:
    model = RandomForestClassifier(n_estimators=150).fit(X_train,y_train)
    ypred = model.predict(X_test)

from sklearn.metrics import accuracy_score

st.markdown(''' Model accuracy Score ''')

accuracy_score(y_test,ypred)

from xgboost import XGBClassifier

if show_model:
    model2 = XGBClassifier(max_depth=5,n_estimators=500).fit(X_train,y_train)
    y2pred = model2.predict(X_test)

accuracy_score(y_test,y2pred)

st.markdown(''' Checking the missclassification of the prediction data ''')

pd.crosstab(y_test,y2pred)

if show_eda:
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    # Confusion matrix and classification report for XGBoost
    print("Classification Report:\n")
    print(classification_report(y_test, y2pred))
    
    # Confusion matrix display
    cm = confusion_matrix(y_test, y2pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - XGBoost")
    st.pyplot(plt.gcf())
    

c(y_test)

st.markdown('''Original accuracy''')

151/210

st.markdown(''' At this point we have already predicted the the high priority tickets which complete objective 1 and 3
 Now going for predicting the resource allocation which is objective 2''')

st.markdown(''' Step 1-  Starting the preprocessing of Data''')

import re

date = list(data.Open_Time)

date[1]
newdate = []

for i in range(len(date)):
    lis=re.split('\s',date[i])
    newdate.append(lis[0])

dat = {'newdate':newdate}
new_date = pd.DataFrame(data=dat)

new_date.head()

new_date['incident']=data.Incident_ID


new_date['week']=pd.DatetimeIndex(new_date['newdate'],dayfirst=True).week
new_date['months'] = pd.DatetimeIndex(new_date['newdate'],dayfirst=True).month
new_date['year']=pd.DatetimeIndex(new_date['newdate'],dayfirst=True).year

new_date.index=pd.to_datetime(new_date.newdate,dayfirst=True)

cou=list(c(new_date.newdate).most_common())

cou1 = []
cou2 = []

for i in range(len(cou)):
    cou1.append(cou[i][1])
    cou2.append(cou[i][0])

new_date['date']=new_date.newdate

cou1[0]

for i in range(len(cou2)):
    new_date.loc[new_date['date']==cou2[i],'date'] = cou1[i]

st.markdown('''Here i have created a new feature which was not presented before will help us on prediction named as 'date' which is the frequency of days ''')

new_date.head()

new_date.date.isnull().sum()

#pd.groupby(new_date.index,new_date.incident)

r = {'dates':cou2,
    'freq':cou1}
tpd = pd.DataFrame(r)

tpd.index = pd.to_datetime(tpd.dates,dayfirst=True)

st.markdown(''' Step 2- Basic Explorarion of timeseries Data''')

if show_eda:
    plt.figure(figsize=(14,7))
    new_date.date.resample('m').mean().plot()

if show_eda:
    plt.figure(figsize=(14,7))
    new_date.date.resample('w').mean().plot()
    #new_date.head()

new_datw1=new_date['2013-6-1':'2014-4-1']

if show_eda:
    plt.figure(figsize=(14,7))
    new_datw1.date.resample('w').mean().plot()

new_datw2=new_date['2012-2-25':'2013-1-1']

new_datw3=new_date['2013-1-1':'2014-1-1']


new_datw2.date.resample('m').mean().plot()

new_datw3.date.resample('w').mean().plot()

ts = new_date.loc[:,['date']]
ts.tail()

st.markdown(''' Step 3 - Selecting specific time-period for analysis ''')

# data with frequency of days

ts2 = ts['2013-9-1':'2014-03-31']

ts2.head()

if show_eda:
    plt.figure(figsize=(14,7))
    sns.lineplot(data=ts2)

st.markdown(''' Checking the data if it is stationary or not with Results of Dickey-Fuller Test ''')

if show_eda:
    from statsmodels.tsa.stattools import adfuller
    def test_stationarity(df, ts):
        
        rolmean = df[ts].rolling(window = 12, center = False).mean()
        rolstd = df[ts].rolling(window = 12, center = False).std()
        
        
        plt.figure(figsize=(14,7))
        orig = plt.plot(df[ts], 
                        color = 'blue', 
                        label = 'Original')
        mean = plt.plot(rolmean, 
                        color = 'red', 
                        label = 'Rolling Mean')
        std = plt.plot(rolstd, 
                       color = 'black', 
                       label = 'Rolling Std')
        plt.legend(loc = 'best')
        plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
        plt.xticks(rotation = 45)
        plt.show(block = False)
        plt.close()
        
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(df[ts], 
                          autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], 
                             index = ['Test Statistic',
                                      'p-value',
                                      '# Lags Used',
                                      'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print (dfoutput)

test_stationarity(ts2,ts='date')

st.markdown(''' Step 4 -Makeing the signal Stationary''')

if show_eda:
    def plot_transformed_data(df, ts, ts_transform):
      plt.figure(figsize=(14,7))
      plt.plot(df[ts])
      
      plt.plot(df[ts_transform], color = 'green')
      plt.title('%s and %s time-series graph' %(ts, ts_transform))
      plt.tick_params(axis = 'x', rotation = 45)
      plt.legend([ts, ts_transform])
    st.pyplot(plt.gcf())
          plt.close()
      
      return

ts2['ts_trans'] = ts2.date.apply(lambda x: np.log(x)) - ts2.date.diff()

st.markdown(''' Ploting the stationary and non stationary data together''')

## transforming the data 

plot_transformed_data(ts2,ts='date',ts_transform='ts_trans')

ts2.isnull().sum()


ts2.fillna(0,inplace=True)

st.markdown(''' Decomposing the data to find  Seasonality, Residual and Trend with original''')

if show_eda:
    def plot_decomposition(df, ts, trend, seasonal, residual):
    
      f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (20, 10), sharex = True)
    
      ax1.plot(df[ts], label = 'Original')
      ax1.legend(loc = 'best')
      ax1.tick_params(axis = 'x', rotation = 45)
    
      ax2.plot(df[trend], label = 'Trend')
      ax2.legend(loc = 'best')
      ax2.tick_params(axis = 'x', rotation = 45)
    
      ax3.plot(df[seasonal],label = 'Seasonality')
      ax3.legend(loc = 'best')
      ax3.tick_params(axis = 'x', rotation = 45)
    
      ax4.plot(df[residual], label = 'Residuals')
      ax4.legend(loc = 'best')
      ax4.tick_params(axis = 'x', rotation = 45)
      plt.tight_layout()
    
      # Show graph
      plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' %(ts), 
                   x = 0.5, 
                   y = 1.05, 
                   fontsize = 18)
    st.pyplot(plt.gcf())
      plt.close()
      
      return

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts2['ts_trans'], freq = 365)

ts2['trend'] = decomposition.trend
ts2['seasonal'] = decomposition.seasonal
ts2['residual'] = decomposition.resid

plot_decomposition(df = ts2, 
                   ts = 'ts_trans', 
                   trend = 'trend',
                   seasonal = 'seasonal', 
                   residual = 'residual')

test_stationarity(df = ts2.dropna(), ts = 'residual')

if show_eda:
    def plot_acf_pacf(df, ts):
     
      f, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6)) 
    
      #Plot ACF: 
    
      ax1.plot(lag_acf)
      ax1.axhline(y=0,linestyle='--',color='gray')
      ax1.axhline(y=-1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
      ax1.axhline(y=1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
      ax1.set_title('Autocorrelation Function for %s' %(ts))
    
      #Plot PACF:
      ax2.plot(lag_pacf)
      ax2.axhline(y=0,linestyle='--',color='gray')
      ax2.axhline(y=-1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
      ax2.axhline(y=1.96/np.sqrt(len(df[ts])),linestyle='--',color='gray')
      ax2.set_title('Partial Autocorrelation Function for %s' %(ts))
      
      plt.tight_layout()
    st.pyplot(plt.gcf())
      plt.close()
      
      return

st.markdown(''' Finding the auto correlation and partial auto-correlation''')

from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(np.array(ts2['ts_trans']), nlags = 20)
lag_pacf = pacf(np.array(ts2['ts_trans']), nlags = 20)

plot_acf_pacf(ts2, ts = 'ts_trans')

if show_eda:
    def run_arima_model(df, ts, p, d, q):
      """
      Run ARIMA model
      """
      from statsmodels.tsa.arima_model import ARIMA
      model = ARIMA(df[ts], order=(p, d, q))  
      results_ = model.fit(disp=-1)  
      len_results = len(results_.fittedvalues)
      ts_modified = df[ts][-len_results:]
      rss = sum((results_.fittedvalues - ts_modified)**2)
      rmse = np.sqrt(rss / len(df[ts]))
      
      plt.plot(df[ts])
      plt.plot(results_.fittedvalues, color = 'red')
      plt.title('For ARIMA model (%i, %i, %i) for ts %s, RSS: %.4f, RMSE: %.4f' %(p, d, q, ts, rss, rmse))
      
    st.pyplot(plt.gcf())
      plt.close()
      
      return results_

model_AR = run_arima_model(ts2, 
                           ts = 'ts_trans', 
                           p = 1, 
                           d = 0, 
                           q = 0)
model_MA = run_arima_model(ts2, 
                           ts = 'ts_trans', 
                           p = 0, 
                           d = 0, 
                           q = 1)
model_MA = run_arima_model(ts2, 
                           ts = 'ts_trans', 
                           p = 1, 
                           d = 0, 
                           q = 1)

st.markdown(''' Using The Data to predict the timeseries on FB Prohet package''')

from fbprophet import Prophet
from datetime import datetime

def days_between(d1, d2):
    """Calculate the number of days between two dates.  D1 is start date (inclusive) and d2 is end date (inclusive)"""
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days + 1)

ts2.head()

ts2.shape

ts2['date'] = ts2.index

#ts2.reset_index(inplace = True)

#model = Prophet()

#ts2.rename(columns={'dat':'ds','ts_trans':'y'},inplace = True)

if show_model:
    #model.fit(ts2.loc[:,['ds','y']])

#future_days=model.make_future_dataframe(periods=365)

#future_days=future

if show_model:
    #pred = model.predict(future_days)

#pred.shape

#pred.head()

if show_eda:
    #plt.figure(figsize=(14,7))
    #pd.plotting.register_matplotlib_converters()
    #model.plot(pred)
    #sns.lineplot(x='ds',y='yhat',data=pred)

#plot_transformed_data(pred,ts='yhat')

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days + 1)

st.markdown(''' Step 5-Specifing the data For prediction''')

date_column = 'dt'
metric_column = 'ts'
table = ts2
start_training_date = '2013-09-02'
end_training_date = '2014-03-31'
start_forecasting_date = '2014-04-01'
end_forecasting_date = '2014-8-30'
year_to_estimate = '2014'
future_num_points = days_between(start_forecasting_date, end_forecasting_date)
cap = None # 2e6
growth = 'linear'
n_changepoints = 25 
changepoint_prior_scale = 0.05 
changepoints = None 
holidays_prior_scale = 10 
interval_width = 0.8 
mcmc_samples = 0
holidays = None
daily_seasonality = True

ts2.head()

ts2.tail()

#ts5 = ts2['date','ts_trans']
ts2.reset_index(inplace= True)
ts2.rename(columns = {'date': 'ds', 'ts_trans': 'y'},inplace=True) 

#df_prophet.head()

if show_eda:
    def create_daily_forecast(df,
    #                           cap,
                              holidays,
                              growth,
                              n_changepoints = 25,
                              changepoint_prior_scale = 0.05,
                              changepoints = None,
                              holidays_prior_scale = 10,
                              interval_width = 0.8,
                              mcmc_samples = 1,
                              future_num_points = 10, 
                              daily_seasonality = True):
      df1 = df
      m = Prophet(growth = growth,
                  n_changepoints = n_changepoints,
                  changepoint_prior_scale = changepoint_prior_scale,
                  changepoints = changepoints,
                  holidays = holidays,
                  holidays_prior_scale = holidays_prior_scale,
                  interval_width = interval_width,
                  mcmc_samples = mcmc_samples, 
                  daily_seasonality = daily_seasonality)
    
      m.fit(df1)
    
      # Create dataframe for predictions
      future = m.make_future_dataframe(periods = future_num_points)
      plt.figure(figsize=(14,7))
      pd.plotting.register_matplotlib_converters()
      
      fcst = m.predict(future)
      m.plot(fcst);
      m.plot_components(fcst)
    
      return fcst

st.markdown(''' Step 6- Creating the Daily Forcast''')


fcst = create_daily_forecast(ts2,
#                             cap,
                             holidays,
                             growth,
                             n_changepoints,
                             changepoint_prior_scale,
                             changepoints, 
                             holidays_prior_scale,
                             interval_width,
                             mcmc_samples,
                             future_num_points, 
                             daily_seasonality)

st.markdown('''  Business Impact & Conclusion

This model can help **ABC Tech** identify critical (Priority 1 or 2) incidents **before they escalate**, allowing preventive actions by IT teams.

**Benefits:**
- Reduced SLA breaches due to early warning
- Lower customer dissatisfaction with timely resolution
- Better resource allocation for high-risk incidents

Further enhancements:
- Real-time incident tagging automation
- Time series forecasting of incident volumes
- Integration with ABC Tech's ITSM tool for full automation

''')

st.markdown('''''')
