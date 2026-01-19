from statsmodels.tsa.arima.model import ARIMA

def train_arima(train_series, order=(2,1,2)):
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    return fitted