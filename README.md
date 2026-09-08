## Overview

This project implements a machine learning workflow for predicting flight prices using a Random Forest regression model.

## Folder Structure

```text
mlops-project-44880/
├── data/
├── src/
│   └── train_44880.py
├── model/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

From the project root directory:

```powershell
python -m pip install -r requirements.txt
```

## Run the Training Script

From the project root directory:

```powershell
python src/train_44880.py
```

The trained model will be saved to:

```text
model/flight_price_model_44880.joblib
```

## Dataset

The dataset contains flight information such as airline, source, destination, number of stops, departure and arrival times, duration, and flight price.


## Model

The project uses a Random Forest Regressor to predict flight prices.


