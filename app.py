# app.py
from flask import Flask, render_template_string
import pandas as pd
import os # To help with file paths
from prophet import Prophet # Import Prophet for forecasting
import logging # For better logging of Prophet messages
import warnings # To suppress specific warnings

# Suppress cmdstanpy warnings (often verbose from Prophet)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# Suppress specific pandas FutureWarnings if they arise
warnings.simplefilter(action='ignore', category=FutureWarning)


# Initialize the Flask application
app = Flask(__name__)

# Define the path to your CSV file
DATA_FILE = os.path.join(os.path.dirname(__file__), 'Walmart_customer_purchases.csv')

@app.route('/')
def home():
    """
    Loads, preprocesses, trains a Prophet model, makes sales predictions for 2025 (season-wise),
    and displays the results in the web application.
    """
    df_summary_html = ""
    forecast_summary_html = ""
    error_message = None

    # --- Debugging: Print the full path the app is looking for ---
    print(f"Attempting to load data from: {DATA_FILE}")

    try:
        # Check if the file actually exists before trying to read it
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"The file '{os.path.basename(DATA_FILE)}' was not found at the expected path.")

        # Load the dataset
        df = pd.read_csv(DATA_FILE)

        # --- Initial Data Inspection and Preprocessing ---
        # 1. Convert 'Purchase_Date' to datetime objects
        df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

        # 2. Aggregate sales data by week
        # Prophet requires a 'ds' (datetime) and 'y' (numeric) column.
        # We'll aggregate to the end of the week (Sunday by default for 'W')
        # to ensure consistent weekly points.
        # First, set Purchase_Date as index for resampling
        df_resampled = df.set_index('Purchase_Date').resample('W')['Purchase_Amount'].sum().reset_index()

        # Rename columns to Prophet's required format
        df_resampled = df_resampled.rename(columns={'Purchase_Date': 'ds', 'Purchase_Amount': 'y'})

        # Ensure 'ds' is datetime and 'y' is numeric
        df_resampled['ds'] = pd.to_datetime(df_resampled['ds'])
        df_resampled['y'] = pd.to_numeric(df_resampled['y'])

        # Filter out any potential zero or negative sales if they exist and are not meaningful
        df_resampled = df_resampled[df_resampled['y'] >= 0]

        # Get a summary for display (using df_resampled for consistency)
        df_summary_html = f"""
        <h3 class="text-2xl font-semibold text-gray-700 mt-6 mb-4">Weekly Sales Data Summary</h3>
        <p class="text-md text-gray-600 mb-2">Total raw records loaded: **{len(df):,}**</p>
        <p class="text-md text-gray-600 mb-2">Date Range of Raw Data: **{df['Purchase_Date'].min().strftime('%Y-%m-%d')}** to **{df['Purchase_Date'].max().strftime('%Y-%m-%d')}**</p>
        <p class="text-md text-gray-600 mb-4">Aggregated Weekly Sales Records: **{len(df_resampled):,}**</p>
        <h4 class="text-xl font-medium text-gray-700 mb-3">First 5 Weekly Sales Records (for Prophet):</h4>
        <div class="overflow-x-auto">
            <table class="min-w-full bg-white border border-gray-300 rounded-md">
                <thead>
                    <tr class="bg-gray-100 text-gray-700 uppercase text-sm leading-normal">
                        <th class="py-3 px-6 text-left">Date (ds)</th>
                        <th class="py-3 px-6 text-right">Weekly Sales (y)</th>
                    </tr>
                </thead>
                <tbody class="text-gray-600 text-sm font-light">
        """
        for _, row in df_resampled.head(5).iterrows():
            df_summary_html += f"""
                    <tr class="border-b border-gray-200 hover:bg-gray-50">
                        <td class="py-3 px-6 text-left whitespace-nowrap">{row['ds'].strftime('%Y-%m-%d')}</td>
                        <td class="py-3 px-6 text-right">${row['y']:,.2f}</td>
                    </tr>
            """
        df_summary_html += """
                </tbody>
            </table>
        </div>
        """

        # --- Sales Forecasting with Prophet ---
        # Initialize and fit the Prophet model
        # We add weekly and yearly seasonality. Daily is not relevant for weekly data.
        # Adjusting seasonality_mode to 'multiplicative' can sometimes capture
        # increasing/decreasing seasonal impact better for sales data.
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='multiplicative')
        model.fit(df_resampled)

        # Create a DataFrame with future dates for 2025
        # We need to forecast for all weeks in 2025. There are 52 or 53 weeks.
        # We'll generate 53 weeks to cover the entire year, as Prophet handles this.
        future = model.make_future_dataframe(periods=53, freq='W') # 'W' for weekly frequency

        # Filter future dates to only include 2025
        future_2025 = future[future['ds'].dt.year == 2025]

        # Make predictions
        forecast = model.predict(future_2025)

        # --- Season-wise Prediction Aggregation for 2025 ---
        # Define seasons (quarters)
        # Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
        forecast['quarter'] = forecast['ds'].dt.quarter
        forecast['year'] = forecast['ds'].dt.year

        # Aggregate predictions by quarter for 2025
        seasonal_forecast_2025 = forecast.groupby(['year', 'quarter'])['yhat'].sum().reset_index()
        seasonal_forecast_2025['quarter_name'] = seasonal_forecast_2025['quarter'].map({
            1: 'Q1 (Jan-Mar)',
            2: 'Q2 (Apr-Jun)',
            3: 'Q3 (Jul-Sep)',
            4: 'Q4 (Oct-Dec)'
        })

        # Generate HTML for 2025 seasonal forecast
        forecast_summary_html = f"""
        <h3 class="text-2xl font-semibold text-gray-700 mt-8 mb-4">2025 Seasonal Sales Forecast</h3>
        <p class="text-md text-gray-600 mb-4">Predicted total sales for each season in 2025:</p>
        <div class="overflow-x-auto">
            <table class="min-w-full bg-white border border-gray-300 rounded-md">
                <thead>
                    <tr class="bg-gray-100 text-gray-700 uppercase text-sm leading-normal">
                        <th class="py-3 px-6 text-left">Year</th>
                        <th class="py-3 px-6 text-left">Season</th>
                        <th class="py-3 px-6 text-right">Predicted Sales</th>
                    </tr>
                </thead>
                <tbody class="text-gray-600 text-sm font-light">
        """
        for _, row in seasonal_forecast_2025.iterrows():
            forecast_summary_html += f"""
                    <tr class="border-b border-gray-200 hover:bg-gray-50">
                        <td class="py-3 px-6 text-left whitespace-nowrap">{row['year']}</td>
                        <td class="py-3 px-6 text-left">{row['quarter_name']}</td>
                        <td class="py-3 px-6 text-right">${row['yhat']:,.2f}</td>
                    </tr>
            """
        forecast_summary_html += """
                </tbody>
            </table>
        </div>
        """

    except FileNotFoundError as e:
        error_message = str(e) + f" Please ensure '{os.path.basename(DATA_FILE)}' is in the same directory as 'app.py'."
        print(f"File Error: {error_message}")
    except pd.errors.EmptyDataError:
        error_message = "Error: The CSV file is empty. Please check the content of 'Walmart_customer_purchases.csv'."
        print(f"Data Error: {error_message}")
    except KeyError as e:
        error_message = f"Error: Missing expected column in CSV: {e}. Please check the column names in 'Walmart_customer_purchases.csv' (expected 'Purchase_Date' and 'Purchase_Amount')."
        print(f"Column Error: {error_message}")
    except Exception as e:
        error_message = f"An unexpected error occurred during data loading, processing, or forecasting: {e}"
        print(f"General Error: {error_message}")

    # Construct the full HTML response
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Walmart Sales Forecaster - Forecast Ready!</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f0f4f8; /* Light blue-gray background */
            }}
        </style>
    </head>
    <body class="flex items-center justify-center min-h-screen p-4">
        <div class="bg-white p-8 rounded-xl shadow-lg max-w-2xl w-full text-center">
            <h1 class="text-4xl font-bold text-gray-800 mb-4">
                Walmart Sales Forecaster - Forecast Ready!
            </h1>
            {"<p class='text-red-500 text-md mt-4 font-semibold'>" + error_message + "</p>" if error_message else "<p class='text-lg text-gray-600'>Data loaded, preprocessed, and sales forecast generated.</p>"}
            {df_summary_html if not error_message else ""}
            {forecast_summary_html if not error_message else ""}
            <p class="text-sm text-gray-500 mt-6">
                We now have seasonal sales predictions for 2025!
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
