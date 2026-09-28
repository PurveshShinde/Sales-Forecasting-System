# 📈 Sales Forecaster

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-black?logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/PurveshShinde/Sales-Forecasting-System.svg)](https://github.com/PurveshShinde/Sales-Forecasting-System/graphs/contributors)

A machine learning–based project to predict future sales using historical data. This project was developed as a 3rd-year Computer Science group project to help businesses and retail stores optimize their inventory and estimate their revenue.

## 🚀 Features
- **Predictive Analytics**: Estimates future sales for specific items at various store outlets based on historical data.
- **Interactive Web Interface**: A clean, easy-to-use Flask UI for inputting parameters and viewing predictions.
- **Pre-trained Models**: Includes pre-trained scikit-learn regression models and categorical encoders for immediate usage.
- **Comprehensive Data Features**: Takes into account item type, fat content, store location, size, and establishment year.

## 📦 Project Structure

```text
├── models/                     # Saved/trained ML models and encoders
├── static/                     # Frontend assets (CSS/JS)
├── templates/                  # HTML templates for web UI (index, result)
├── app.py                      # Main Flask application server
├── model.ipynb                 # Jupyter Notebook for data processing & training
├── requirements.txt            # Python package dependencies
├── Train.csv                   # Training dataset
├── Walmart_customer_purchases.csv # Supplementary dataset
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python 3.8 or higher installed on your machine.

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PurveshShinde/Sales-Forecasting-System.git
   cd Sales-Forecasting-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser and visit:**
   `http://127.0.0.1:5000`

## 🧠 Model Training

To experiment with or retrain the machine learning model:
1. Open the `model.ipynb` notebook using Jupyter.
2. Run all cells sequentially (Data Loading → Preprocessing → Training → Evaluation).
3. The newly trained model and encoders will be saved automatically inside the `models/` directory, ready to be used by the Flask app.

## 👥 Contributors

This project was developed by:
- [Purvesh Shinde](https://github.com/PurveshShinde)
- Amey Gawade
- Pratik Yadav
- Prathamesh Ambekar

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
