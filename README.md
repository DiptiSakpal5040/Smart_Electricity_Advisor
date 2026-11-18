# ⚡ Smart Electricity Consumption and Saving Advisor

A Machine Learning-based web application that classifies household electricity consumption into **Low, Medium, and High** categories using three machine learning algorithms: Random Forest, Support Vector Machine (SVM), and XGBoost.

The project also provides a user-friendly website to manage household electricity details, view consumption predictions, compare machine learning models, explore electricity-saving suggestions, estimate electricity bills, and track consumption history.

---

## 📌 1. Project Overview

Electricity consumption is an important part of everyday life. Understanding electricity usage can help households become more aware of their consumption patterns.

This project uses historical electricity consumption data and machine learning techniques to classify electricity consumption levels.

The system provides three different model predictions and compares their performance to identify the best-performing algorithm in our experiment.

### Main Objectives

- Analyze historical household electricity consumption data.
- Clean and preprocess the dataset.
- Apply machine learning classification algorithms.
- Classify electricity consumption into Low, Medium, and High categories.
- Compare the performance of different ML algorithms.
- Develop an interactive website using Streamlit.
- Provide electricity-saving suggestions and consumption-related information.

---

## 🧠 2. Machine Learning Algorithms

The project implements three machine learning algorithms:

### 1. Random Forest
Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate predictions. It uses majority voting for classification.

### 2. Support Vector Machine (SVM)
SVM is a supervised machine learning algorithm that finds a decision boundary to separate different classes. We used LinearSVC with feature scaling.

### 3. XGBoost
XGBoost is an advanced gradient boosting algorithm that builds decision trees sequentially to improve prediction performance.

---

## 📊 3. Dataset

**Dataset Name:** Individual Household Electric Power Consumption

**Source:** UCI Machine Learning Repository

**File Name:** `household_power_consumption.txt`

Dataset characteristics:

- Original records: 2,075,259
- Number of columns: 9
- Cleaned records: 2,049,280
- Data collection period: December 2006 to November 2010
- Data frequency: Approximately one-minute intervals
- Data represents one household in France.

### Important Dataset Features

| Feature | Description |
|---|---|
| Date | Date of electricity reading |
| Time | Time of electricity reading |
| Global_active_power | Overall active power consumption |
| Global_reactive_power | Reactive power |
| Voltage | Voltage measurement |
| Global_intensity | Current intensity |
| Sub_metering_1 | Kitchen-related energy measurement |
| Sub_metering_2 | Laundry-related energy measurement |
| Sub_metering_3 | Water heater and air conditioner-related energy measurement |

### Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Handled missing values by removing incomplete rows.
3. Combined Date and Time into a DateTime column.
4. Extracted time-based features such as Hour, Day, Month, Year, and DayOfWeek.
5. Created three target categories: Low, Medium, and High.
6. Selected relevant input features.
7. Divided the dataset chronologically into training and testing sets.

### Target Categories

The target variable is `Consumption_Level`.

The categories were created using quantile thresholds calculated from the training data:

- Low: Global_active_power ≤ 0.342
- Medium: Global_active_power > 0.342 and ≤ 1.374
- High: Global_active_power > 1.374

These thresholds represent relative consumption categories based on the dataset, not universal electricity consumption limits.

---

## 📈 4. Model Performance

All three algorithms were evaluated using the same 50,000-record test sample.

| Algorithm | Accuracy |
|---|---:|
| Random Forest | 77.726% |
| SVM (LinearSVC) | 69.202% |
| XGBoost | 77.600% |

### Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

Random Forest achieved the highest accuracy in our experiment, closely followed by XGBoost.

---

## 🌐 5. Website Features

The website is developed using Streamlit and includes the following pages:

| Page | Description |
|---|---|
| Dashboard | Displays the user's electricity overview. |
| My Electricity Details | Allows users to enter household electricity information. |
| Consumption Prediction | Generates predictions using all three ML algorithms. |
| Model Comparison | Compares model performance using metrics and charts. |
| Electricity Saving Advisor | Provides electricity-saving tips and a savings calculator. |
| Electricity Bill Estimator | Estimates electricity bills using simplified tariff assumptions. |
| Consumption Trends | Displays monthly electricity consumption trends. |
| Reports & History | Displays prediction history and monthly records. |
| My Profile | Displays and updates user profile information. |
| About Project | Provides information about the project and technologies. |

### Additional Features

- User registration and login.
- Separate user electricity records.
- Prediction history.
- Interactive charts.
- Monthly consumption tracking.
- CSV report downloads.
- Electricity-saving suggestions.

---

## 🛠️ 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning algorithms |
| XGBoost | Gradient boosting algorithm |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive charts |
| Streamlit | Website development |
| Joblib | Saving and loading trained models |
| JSON | Local application data storage |
| Jupyter Notebook | Model development and experimentation |
| Git & GitHub | Version control and project hosting |

---

## 📁 7. Project Structure

```text
Smart_Electricity_Advisor/
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── notebook/
│
├── models/
│   ├── random_forest.pkl
│   ├── svm_model.pkl
│   ├── xgboost_model.pkl
│   ├── svm_scaler.pkl
│   └── xgboost_label_encoder.pkl
│
├── websites/
│   ├── app.py
│   ├── users.json
│   ├── electricity_data.json
│   └── prediction_history.json
│
├── screenshots/
│
├── report/
│
├── README.md
│
└── venv/
```

---

## ⚙️ 8. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the Project Folder

```bash
cd Smart_Electricity_Advisor
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly streamlit joblib xgboost threadpoolctl
```

### Step 6: Prepare Project Files

Make sure the trained model files are available inside the `models` folder.

Place the dataset in the appropriate location under `dataset/raw/` if you want to reproduce the model training workflow.

### Step 7: Run the Website

```bash
streamlit run websites/app.py
```

The application will open in your browser.

Default local address:

```text
http://localhost:8501
```

---

## 🔄 9. Working Process

The complete project follows this workflow:

1. Collect historical electricity consumption data.
2. Preprocess and clean the dataset.
3. Extract useful features.
4. Create Low, Medium, and High consumption categories.
5. Split the data into training and testing sets.
6. Train Random Forest, SVM, and XGBoost models.
7. Evaluate and compare model performance.
8. Save trained models using Joblib.
9. Load the trained models into the Streamlit website.
10. Accept user input and generate consumption predictions.
11. Display model results and save prediction history.

---

## ⚠️ 10. Limitations

- The dataset represents one household in France and may not represent all households.
- Consumption categories are based on dataset-derived thresholds.
- The ML system predicts consumption categories, not exact future electricity units or bills.
- The website currently accepts manually entered electrical measurements and is not connected to a live smart meter.
- The electricity bill estimator uses simplified tariff assumptions.
- The application uses local JSON files for prototype-level data storage.

---

## 🚀 11. Future Scope

Possible future improvements include:

- Integration with IoT-based smart electricity meters.
- Using larger datasets from different households.
- Improving model performance through hyperparameter tuning.
- Adding electricity consumption forecasting.
- Providing more personalized electricity-saving recommendations.
- Using a secure database for user information.
- Adding explainable AI to understand model predictions.

---

## 🎯 12. Conclusion

The Smart Electricity Consumption and Saving Advisor demonstrates how machine learning can be applied to household electricity consumption analysis.

By implementing Random Forest, SVM, and XGBoost, the project compares different classification algorithms and identifies Random Forest as the best-performing model in our experiment.

The Streamlit web application provides an interactive platform for users to view predictions, compare models, understand consumption categories, and explore electricity-saving information.

The project combines machine learning, data preprocessing, visualization, and web development into one practical application.

---

**Project Name:** Smart Electricity Consumption and Saving Advisor

**Domain:** Machine Learning and Web Development

**Programming Language:** Python

**Framework:** Streamlit

**Algorithms:** Random Forest, SVM, XGBoost
