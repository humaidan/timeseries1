# timeseries1
Course: Time Series Analysis, Forecasting, and Machine Learning

## Lessons Learnt:

### Basics
- There are classification and regression (forecasting = regression)
- Data Transformation: Necessary before inputing it to ML model:
    - Power Transform: y'(t) = y(t)<sup>1/2</sup>==> smoother ==> more linear which is easier for ML
    - Log Transform: y'(t) = logy(y(t) + 1) ==> even smoother + log-normal distrib + log returns instead of %
    - Box Cox Transform: for non-time series data ==> make data normally distributed => stationary data that doesn't change over time

- Forecasting/Regression Metrics:
    - SSE (Sum Squared Error): Pros - Famous for being non-negative + liklihood errors are Gaussian distributed  [Correct Metric to minize when errors are normally distributed]  ==>  Cons: More errors the more predictions you make (dont cancel out)
    - MSE (Mean Square Error): SSE/N - Pros - indepdent of data points you have so you can compare two samples, Cons - No units & depend on scale of data (i.e. house prediction error 10_000)
    - RMSE (Root Mean Square Error): Pros - now it's on the same scale as the original data (i.e. house prediction error $100)
    - MAE (Mean Absolute Error): sum(absolute difference between prediction vs target)/N: Pros - model is less influenced by outliers, Cons - depend on scale of data
    - R<sup>2</sup>: 1 - SSE/SST - Not an error as we want it to be bigger | not smaller.  If R^<sup>2</sup> < 0: predictions are worse than predicting mean of target !
        - If model is perfect: MSE=0 & R<sup>2</sup>=1
    - MAPE (Mean Absolution Percentage Error): Pro - scale indepedent (house prices off by 0.1% instead of $1000 for $1M prediction)
    - sMAPE (Symmetric MAPE)
<br><br>
- Both SMA, EMA, and EWMA are used to measure data over time
    - SMA:  rolling_window = df['symbol'].rolling(window) ==> rolling_window.mean()  |  covar = df[['sym1', 'sym2']].rolling(window).cov()
    - EWMA: xhat = df['symbol].ewm(alpha, adjust=False).mean()  #also var/cov
    - Exponential Smoothing (it is a forecast opposing to SMA/EWMA):
      y<sup>^</sup> <sub>t+h|t</sub> = l<sub>t</sub>  (depends on level)
        ```python
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing

            ses = SimpleExpSmoothing(data)  # data is univariate
            res = ses.fit(smoothing_level=alpha, optimized=False) # returns HoltWintersResults object
            res.fittedvalues # get all in-sample predictions
            res.predict(start=dt1, end=dt2)  # in-sample or out-of-sample predictions 
            res.predict(n) #forecast n steps  ==> line
<br><br>
    - Holt's __Linear__ Trend Model:  y<sup>^</sup> <sub>t+h|t</sub> = l<sub>t</sub> + hb<sub>t</sub> (depends on level + trend)
        ```python
            from statsmodels.tsa.holtwinters import Holt

            model = Holt(data)      #  Holt(train['symbol'], initilization_method='legacy_heuristic')
            res = model.fit() # returns HoltWintersResults object
            res.fittedvalues # get all in-sample predictions
            res.predict(n) #forecast n steps  ==> line

            # To graph values:
            df.loc[train_index, 'Holt'] = res.fittedvalues
            df.loc[test_index, 'Holt'] = res.forecast(N)
            df['symbol', 'Holt'].plot()
<br><br>
    - Hol-__Winters__ Model:   (depends on trend + seasonal (cycles) + level (average))
      - Additive method: y<sup>^</sup> <sub>t+h|t</sub> = l<sub>t</sub> + hb<sub>t</sub> + s<sub>t+h-mk</sub>
      - Multiplicative method: y<sup>^</sup> <sub>t+h|t</sub> = (l<sub>t</sub> + hb<sub>t</sub>)s<sub>t+h-mk</sub>
            ```python
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=12)      #args are 'add' or 'mul'
            res = model.fit() # returns HoltWintersResults object
            res.fittedvalues # get all in-sample predictions
            res.predict(n) #forecast n steps  
<br><br>

### ARIMA Model
- ARIM(p, d, q):
  - AR(p): The autoregressive component is just linear (lines or plane) regression of order (p) for past p datapoints to predict the future.
  - I(I): The differencing component is used to make the time series stationary. "d" parameter represents the number of times the time series is differenced.
  - MA(q): The model uses the past (q) forecast errors to predict future values
  - Example: 
    ```
    arima = ARIMA(train['PAX'], order=(8,1,1)) 
    ```

-ADfuller:
    - Test for stationary.  adfuller() returns test value, p-value, and others:
        ```
        res = adfuller(x)
        print(f"Test-Statistic: {res[0]}")
        print(f"P-Value: {res[1]}")
        ```
-ACF & PACF:
    - Auto-Correlation Function (ACF): understand the correlation between a time series data point and its lagged versions. The ACF plot shows the correlation coefficients of the time series data with different lags. The x-axis represents the lag, and the y-axis shows the correlation coefficient.
    - Partial Auto-Correlation Function (PACF): Measure the correlation between a data point and its lagged versions, while controlling for the influence of other lags. The PACF plot shows the partial correlation coefficients of the time series data with different lags. The x-axis represents the lag, and the y-axis shows the correlation coefficient.
    - Both ACF and PACF plots are commonly used in time series analysis to identify the order of autoregressive (AR) and moving average (MA) models. Specifically, ACF is used to identify the order of MA models, while PACF is used to identify the order of AR models. 
  
- Auto ARIMA / Seasonal ARIMA (SARIMA):





<br><br><br><br>
## Extra Readings:

### Lazy Programmer Github
- https://github.com/lazyprogrammer/machine_learning_examples/

### Estimating Box-Cox power transformation parameter via goodness of fit tests
- https://arxiv.org/pdf/1401.3812.pdf

### Linear Regression
- https://deeplearningcourses.com/c/data-science-linear-regression-in-python/

### Logistic Regression
- https://deeplearningcourses.com/c/data-science-logistic-regression-in-python/

### Support Vector Machines
- https://deeplearningcourses.com/c/support-vector-machines-in-python

### Random Forests
- https://deeplearningcourses.com/c/machine-learning-in-python-random-forest-adaboost

### Deep Learning and Tensorflow 2
- https://deeplearningcourses.com/c/deep-learning-tensorflow-2

### Gaussian Processes for Regression and Classification
- https://www.cs.toronto.edu/~radford/ftp/val6gp.pdf

### How Does Backpropagation Work? (In-Depth)
- https://deeplearningcourses.com/c/data-science-deep-learning-in-python/
- https://deeplearningcourses.com/c/data-science-deep-learning-in-theano-tensorflow/

### Forecasting at Scale (Facebook Prophet)
- https://peerj.com/preprints/3190.pdf

### Statistical and Machine Learning forecasting methods: Concerns and ways forward
- https://journals.plos.org/plosone/article%3Fid%3D10.1371/journal.pone.0194889

## Datasets:
### Time Series Basics
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/SPY.csv

### ETS and Exponential Smoothing
- https://lazyprogrammer.me/course_files/sp500_close.csv
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/timeseries/perrin-freres-monthly-champagne.csv
- https://lazyprogrammer.me/course_files/sp500sub.csv

### ARIMA
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/sp500sub.csv
- https://lazyprogrammer.me/course_files/timeseries/perrin-freres-monthly-champagne.csv

### VARMA
- https://lazyprogrammer.me/course_files/timeseries/temperature.csv
- https://lazyprogrammer.me/course_files/timeseries/us_macro_quarterly.xlsx

### Machine Learning Methods
- https://lazyprogrammer.me/course_files/SPY.csv
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/timeseries/perrin-freres-monthly-champagne.csv
- https://lazyprogrammer.me/course_files/sp500sub.csv

### Artificial Neural Networks
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/sp500sub.csv
- https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones 
- https://www.kaggle.com/erenaktas/human-activity-recognition 
- https://lazyprogrammer.me/course_files/timeseries/UCI-HAR.zip

### Convolutional Neural Networks
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/timeseries/UCI-HAR.zip

### Recurrent Neural Networks
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/timeseries/UCI-HAR.zip

### GARCH
- https://lazyprogrammer.me/course_files/SPY.csv

### Facebook Prophet
- https://www.kaggle.com/c/rossmann-store-sales 
- https://lazyprogrammer.me/course_files/timeseries/rossmann_train.csv 
- https://lazyprogrammer.me/course_files/airline_passengers.csv
- https://lazyprogrammer.me/course_files/timeseries/UCI-HAR.zip
