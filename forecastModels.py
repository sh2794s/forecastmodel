# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.stats import linregress

# --- Single Exponential Smoothing ---
def single_ES(data, alpha):
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1!")
    smoothed_forecasting = [data[0]]  # Initialize with the first actual demand value
    for i in range(1, len(data) + 1):
        # Forecast = alpha * previous actual + (1 - alpha) * previous forecast
        new_forecasting = alpha * data[i - 1] + (1 - alpha) * smoothed_forecasting[i - 1]
        smoothed_forecasting.append(new_forecasting)
    return smoothed_forecasting

# --- Double Exponential Smoothing ---
def double_ES(data, alpha, beta):
    n = len(data)
    if n < 2:
        raise ValueError("Data must contain at least two periods' data")
    I = [data[0]]  # Initial level
    S = [data[1] - data[0]]  # Initial trend
    y = ["N/A"] * 2  # First two forecasts are not available
    for t in range(1, n):
        # Level equation: I[t] = alpha * actual + (1 - alpha) * (previous level + previous trend)
        new_I = alpha * data[t] + (1 - alpha) * (I[t - 1] + S[t - 1])
        I.append(new_I)
        # Trend equation: S[t] = beta * (current level - previous level) + (1 - beta) * previous trend
        new_S = beta * (I[t] - I[t - 1]) + (1 - beta) * S[t - 1]
        S.append(new_S)
        # Forecast equation: y[t] = level + trend
        y.append(new_I + new_S)
    return I, S, y

# --- Triple Exponential Smoothing (Holt-Winters) ---
def triple_ES(D, alpha, beta, gamma, N):
    # Initialize trend (S), level (I), and seasonality (C)
    S = ["N/A"] * (2 * N - 1)
    S.append((sum(D[N:2 * N]) - sum(D[0:N])) / N ** 2)  # Trend initialization

    # Initialize seasonality: average of ratios from two seasons
    C = ["N/A"] * N
    for t in range(N):
        new_c = D[t] / (sum(D[0:N]) / N) + D[N + t] / (sum(D[N:2 * N]) / N)
        new_c /= 2
        C.append(new_c)

    # Initialize level (I)
    I = ["N/A"] * (2 * N - 1)
    I.append(D[2 * N - 1] / C[2 * N - 1])

    # Initialize forecast
    y = ["N/A"] * (2 * N)
    y.append((I[2 * N - 1] + S[2 * N - 1]) * C[N])

    # Recursively compute components
    for t in range(2 * N, len(D)):
        # Level: I[t] = alpha * (demand / seasonal) + (1 - alpha) * (previous level + trend)
        I.append(alpha * D[t] / C[t - N] + (1 - alpha) * (I[t - 1] + S[t - 1]))
        # Trend: S[t] = beta * (current level - previous level) + (1 - beta) * previous trend
        S.append(beta * (I[t] - I[t - 1]) + (1 - beta) * S[t - 1])
        # Seasonality: C[t] = gamma * (demand / level) + (1 - gamma) * previous seasonal
        C.append(gamma * D[t] / I[t] + (1 - gamma) * C[t - N])
        # Forecast: y[t] = (level + trend) * seasonal
        y.append((I[t] + S[t]) * C[t + 1 - N])

    return I, S, C, y

# --- Linear Regression ---
def LR(independent_variable, dependent_variable):
    x = np.array(independent_variable)
    y = np.array(dependent_variable)
    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    fitting_line = slope * x + intercept  # y = mx + b line fit
    return fitting_line, slope, intercept

# --- Performance Metrics ---
def performance_measures(forecasting, observation):
    n = len(observation)
    sum_AD = sum_SE = sum_APE = 0.0
    for i in range(n):
        if forecasting[i] == "N/A":
            n -= 1  # Exclude "N/A" entries from calculations
        else:
            e_t = forecasting[i] - observation[i]  # Error at time t
            sum_AD += abs(e_t)  # Absolute Deviation
            sum_SE += e_t ** 2  # Squared Error
            sum_APE += abs(e_t) / observation[i]  # Absolute Percentage Error
    MAD = sum_AD / n
    MSE = sum_SE / n
    MAPE = sum_APE / n
    return MAD, MSE, MAPE

# --- Main Execution Block ---
df = pd.read_excel("C:\\Users\\Ravindra\\Downloads\\MSU-Assignment-Summaya\\2\\HW 5 - ES and LR.xlsx")  # Load data from Excel file
month = df['Month'].tolist()  # Extract month as x-axis (independent variable)
demand = df['Demand'].tolist()  # Extract demand as y-axis (dependent variable)

# Prepare a list of periods and a DataFrame to store results
period = [i + 1 for i in range(len(month) + 1)] + ["MAD", "MSE", "MAPE"]
new_df = pd.DataFrame()
new_df['Month'] = period

# --- Linear Regression ---
# Fit a straight line to the demand data
forecasting_LR, slope, intercept = LR(month, demand)
forecasting_LR = np.append(forecasting_LR, intercept + slope * (len(month) + 1))  # Forecast for next period
MAD, MSE, MAPE = performance_measures(forecasting_LR, demand)  # Evaluate model
forecasting_LR = np.append(forecasting_LR, [MAD, MSE, MAPE])
new_df['LR'] = forecasting_LR  # Store in DataFrame

# --- Single Exponential Smoothing ---
alpha = 0.5  # Smoothing parameter
forecasting_SES = single_ES(demand, alpha)
forecasting_SES += list(performance_measures(forecasting_SES, demand))
new_df['Single ES'] = forecasting_SES

# --- Double Exponential Smoothing ---
beta = 0.35  # Trend smoothing parameter
I_DES, S_DES, forecasting_DES = double_ES(demand, alpha, beta)  # Compute level, trend, and forecasts
forecasting_DES += list(performance_measures(forecasting_DES, demand))
new_df['Double ES'] = forecasting_DES

# --- Triple Exponential Smoothing ---
gamma = 0.6  # Seasonality smoothing parameter
N = 4  # Number of seasons
I_TES, S_TES, C_TES, forecasting_TES = triple_ES(demand, alpha, beta, gamma, N)  # Compute level, trend, seasonality, forecast
forecasting_TES += list(performance_measures(forecasting_TES, demand))
new_df['Triple ES'] = forecasting_TES

# --- Save results and print ---
new_df.to_excel("Results.xlsx", index=False)  # Save output to Excel
