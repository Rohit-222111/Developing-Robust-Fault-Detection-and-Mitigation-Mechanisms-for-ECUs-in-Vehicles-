# Developing-Robust-Fault-Detection-and-Mitigation-Mechanisms-for-ECUs-in-Vehicles-

This project implements a real-time machine learning-based fault detection and performance prediction system for Electronic Control Units (ECUs) and critical automotive components such as:

- Oxygen Sensor
- Radiator
- Spark Plug
- Turbocharger

It leverages classification and regression models (Random Forest and XGBoost) and is aligned with **SDG 9 (Industry, Innovation & Infrastructure)** and **SDG 11 (Sustainable Cities & Communities)**.

<details>
<summary>💡 Objectives</summary>

- Develop a classification model to categorize component health status (Critical/Degrading/Healthy).
- Build regression models to predict component performance over time.
- Integrate ML models into a digital twin environment for real-time visualization.
- Improve predictive maintenance for vehicle safety and efficiency.

</details>

<details>
<summary>🧠 Methodology</summary>

- Data Cleaning and Preprocessing (Normalization, Outlier Removal)
- Classification using `XGBoostClassifier` (Health status)
- Regression using `RandomForestRegressor` (Performance prediction)
- Model evaluation using R², MAE, MSE, Accuracy, F1-score
- Subset-wise performance validation
- Digital Twin-based performance visualization

</details>

<details>
<summary>📂 Dataset Description</summary>

- **Oxygen Sensor:** Air-Fuel Ratio, Sensor Voltage, Fuel Trim, etc.
- **Radiator:** Coolant Temp, Flow Rate, Ambient Temp, etc.
- **Spark Plug:** Ignition Timing, Plug Gap, Voltage Supply
- **Turbocharger:** Boost Pressure, Exhaust Temp, Turbo RPM

Synthetic yet realistic datasets with 220,000 samples per component were used.

</details>

<details>
<summary>📈 Model Results</summary>

| Component       | MAE     | MSE     | R² Score |
|----------------|---------|---------|----------|
| Oxygen Sensor  | 1.73    | 4.83    | 0.94     |
| Radiator       | 4.05    | 25.49   | 0.85     |
| Spark Plug     | 4.14    | 26.97   | 0.96     |
| Turbocharger   | 5.71    | 51.40   | 0.88     |

- Classification Accuracy up to **91%**
- Subset R² scores consistently above **0.85**

</details>

<details>
<summary>🛠️ Technologies Used</summary>

- Python 3.12
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Pandas / NumPy
- Jupyter Notebook
- Tkinter (for GUI Digital Twin)
  
</details>

<details>
<summary>🧪 How to Run</summary>

```bash
pip install -r requirements.txt
python component_pipeline.py  # or individual .ipynb files
