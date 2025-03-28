# 📊 Forecasting Models in Python

This project implements basic forecasting techniques on time series data using:

- 🔹 Single Exponential Smoothing (SES)
- 🔹 Double Exponential Smoothing (DES)
- 🔹 Triple Exponential Smoothing (Holt-Winters Method)
- 🔹 Linear Regression (LR)

---

## 📁 Files Included

| File Name            | Description                                      |
|----------------------|--------------------------------------------------|
| `forecastModels.py`  | Main Python script with all forecasting models  |
| `Results.xlsx`       | Forecasted results and error metrics            |

---

## 🧠 Methodology

- Custom implementations for SES, DES, and TES (no external forecasting libraries)
- Performance metrics used:
  - **MAD** – Mean Absolute Deviation
  - **MSE** – Mean Squared Error
  - **MAPE** – Mean Absolute Percentage Error
- Final results are programmatically saved to `Results.xlsx`

---

## 🚀 How to Run

```bash
pip install pandas numpy scipy openpyxl
python forecastModels.py
