import matplotlib.pyplot as plt

def plot_forecast(train, test, predictions, title):
    plt.figure(figsize=(12,6))
    plt.plot(train["date"], train["patient_visits"], label="Train")
    plt.plot(test["date"], test["patient_visits"], label="Actual")
    plt.plot(test["date"], predictions, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.show()