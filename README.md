# U.S. Retail Sales Forecasting using LSTM

This project implements a deep learning-based time series forecasting model using LSTM (Long Short-Term Memory) networks to predict future U.S. retail and food services sales figures. The dataset contains monthly sales data from January 1992 to January 2021.

---

## 📊 Dataset

- **Source:** RSCCASN.csv
- **Entries:** 349 monthly records
- **Date Range:** 1992-01 to 2021-01
- **Column:**
  - `RSCCASN`: Seasonally adjusted retail sales (in millions of USD)

---

## 🎯 Objectives

- Visualize historical retail sales trends.
- Preprocess and scale data using MinMaxScaler.
- Build an LSTM model using Keras/TensorFlow.
- Forecast future sales using trained LSTM model.
- Evaluate and visualize predictions on the test set.

---

## 🧰 Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## 🛠️ Preprocessing Steps

- Parsed dates and set `DATE` column as index
- Visualized data trends using `matplotlib`
- Scaled data between 0 and 1 using `MinMaxScaler`
- Used `TimeseriesGenerator` to structure input for LSTM

---

## 🧠 LSTM Model Details

- Model Architecture:
  - 1 LSTM Layer (100 units, ReLU activation)
  - 1 Dense Layer
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- EarlyStopping used to prevent overfitting
- Trained using a sliding window approach (12 time steps)

```python
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(12, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
````

---

## 📈 Model Training & Evaluation

* **Training Samples:** First 331 months
* **Test Samples:** Last 18 months
* The model predicted 18 months ahead using a recursive strategy.

| Metric           | Value    |
| ---------------- | -------- |
| Final Train Loss | \~0.0028 |
| Best Val Loss    | \~0.0055 |

---

## 📉 Visualization

* Plot of actual vs predicted sales for the test set
* Residual loss plot across epochs
* Forecasted next 12 months after January 2021

---

## 🔮 Future Forecast (12 months)

Using the entire dataset, the model was retrained and then used to forecast the next 12 months of retail sales beyond the final date in the dataset (Jan 2021).

```python
forecast = []
for i in range(12):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred)
```

---

## 📂 Suggested Folder Structure

```
retail-sales-lstm/
├── RSCCASN.csv
├── retail_sales_forecast.ipynb
├── README.md
└── images/
    ├── forecast_plot.png
    └── loss_plot.png
```

---

## 📌 Key Takeaways

* LSTM is effective for time series forecasting of economic indicators.
* Recursive prediction strategy helps extend forecast horizons.
* Model performance is sensitive to scaling, window size, and LSTM depth.

---

## 👨‍💻 Author

**D. Lalith Kumar**
B.Tech CSE (AI & ML)

---

## 📜 License

This project is open for academic and research use. Data courtesy of U.S. Census Bureau.
