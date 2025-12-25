Sales Forecaster

A machine learning–based project to predict future sales using historical data.
This project was developed as a 3rd-year Computer Science group project.

📌 Project Overview

Sales Forecaster trains a regression model to predict sales (e.g., retail/store sales).
It includes data preprocessing, model training, and a simple Flask web application to interact with predictions.

📦 Project Structure
├── models/              # Saved trained models
├── static/              # Frontend assets
├── templates/           # HTML templates
├── app.py               # Flask application
├── model.ipynb          # Model training notebook
├── requirements.txt     # Python dependencies
├── Train.csv            # Training dataset
├── Walmart_customer_purchases.csv
└── README.md

🚀 Getting Started
Prerequisites

Python 3.8 or higher

Setup

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py


Open your browser and visit:

http://127.0.0.1:5000

🛠 Model Training

To retrain the model:

Open model.ipynb

Run all cells (data loading → preprocessing → training → evaluation)

Save the trained model inside the models/ directory

👥 Authors

This project was developed by:

Purvesh Shinde

Amey Gawade

Pratik Yadav

Prathamesh Ambekar
