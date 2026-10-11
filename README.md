# Car Price Prediction

## 1. Project Overview

This project uses machine learning to predict the prices of used cars based on their characteristics.

The target variable is `price_usd`, which represents the price of a car in US dollars.

The project follows a complete machine learning workflow, from exploratory data analysis to model training, evaluation, comparison, and price prediction.

## 2. Dataset

The project uses the `cars.csv` dataset.

The dataset contains information about used cars, including:

- Make and model
- Year of manufacture
- Vehicle condition
- Mileage
- Fuel type
- Engine volume
- Color
- Transmission
- Drive unit
- Car segment

The original target column is `priceUSD`.

## 3. Project Structure

```text
car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_eda_cars_dataset.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   ├── linear_regression_model.joblib
│   └── car_price_model.joblib
├── car_price_prediction.py
├── predict_car.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. Technologies

The project uses:

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## 5. Machine Learning Workflow

The project includes the following steps:

1. Exploratory Data Analysis (EDA)
2. Data cleaning
3. Feature engineering
4. Data preprocessing
5. Linear Regression baseline training
6. Model evaluation
7. Comparison of regression models
8. Selection and saving of the best model
9. Prediction of used car prices

## 6. Regression Models

The following regression algorithms are compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² score

The best model is selected based on the evaluation results, with MAE used as the main comparison metric.

## 7. Installation

Clone the repository and open the project folder.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 8. Run the Project

To run the main machine learning workflow:

```bash
python car_price_prediction.py
```

To predict the price of an individual car:

```bash
python predict_car.py
```

The prediction script asks the user to enter the car characteristics and returns an estimated price in US dollars.

The trained model must exist in the `models/` directory before running the prediction script.

## 9. Models and Results

The trained models are saved in the `models/` directory.

The comparison results are used to identify the best-performing regression algorithm.

The final model is stored as `car_price_model