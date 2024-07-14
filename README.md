# House-Price-Prediction.py

A Python-based project for predicting house prices using machine learning techniques.

## Overview

House-Price-Prediction.py utilizes machine learning algorithms to predict house prices based on various features such as location, size, and amenities. The project includes data preprocessing, model training, and evaluation.

## Features

- **Data Preprocessing:** Cleans and prepares the dataset for modeling.
- **Feature Engineering:** Creates new features from existing data.
- **Model Training:** Trains various machine learning models.
- **Model Evaluation:** Evaluates model performance using metrics like RMSE.

## Requirements

- Python 3.6+
- Jupyter Notebook
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/noothispoorthi/House-Price-Prediction.py.git
    cd House-Price-Prediction.py
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook house_price__prediction_Spoorthi.ipynb
    ```

2. Follow the steps in the notebook to load data, preprocess it, train models, and make predictions.

## Example

```python
from house_price_prediction import HousePricePredictor

predictor = HousePricePredictor()
data = predictor.load_data('data/house_prices.csv')
cleaned_data = predictor.preprocess_data(data)
model = predictor.train_model(cleaned_data)
prediction = model.predict([[3, 2000, 2, 'suburban']])
print(f"Predicted House Price: {prediction}")
