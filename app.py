# app.py
from flask import Flask, render_template_string
import pandas as pd
import os # To help with file paths
from prophet import Prophet # Import Prophet for forecasting
import logging # For better logging of Prophet messages
import warnings # To suppress specific warnings
import json # To pass data to JavaScript

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
    and displays the results in the web application along with a visualization.
    """
    df_summary_html = ""
    forecast_summary_html = ""
    chart_data_json = "{}" # Initialize as empty JSON string
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
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='multiplicative')
        model.fit(df_resampled)

        # Create a DataFrame with future dates for 2025
        future = model.make_future_dataframe(periods=53, freq='W') # 'W' for weekly frequency

        # Make predictions for the entire future dataframe (including historical dates)
        forecast_full = model.predict(future)

        # Filter forecast to only include 2025 for seasonal summary
        forecast_2025_only = forecast_full[forecast_full['ds'].dt.year == 2025]

        # --- Season-wise Prediction Aggregation for 2025 ---
        forecast_2025_only['quarter'] = forecast_2025_only['ds'].dt.quarter
        forecast_2025_only['year'] = forecast_2025_only['ds'].dt.year

        seasonal_forecast_2025 = forecast_2025_only.groupby(['year', 'quarter'])['yhat'].sum().reset_index()
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

        # --- Prepare data for Chart.js visualization ---
        # Combine historical and forecasted data for plotting
        # We need 'ds' (date) and 'y' (actual sales) from df_resampled
        # and 'ds' (date) and 'yhat' (predicted sales) from forecast_full
        plot_data = pd.DataFrame({
            'ds': pd.concat([df_resampled['ds'], forecast_full['ds']]),
            'y_actual': pd.concat([df_resampled['y'], pd.Series([None]*len(forecast_full))]), # Actuals only for historical
            'y_forecast': pd.concat([pd.Series([None]*len(df_resampled)), forecast_full['yhat']]), # Forecasts for all
            'yhat_lower': pd.concat([pd.Series([None]*len(df_resampled)), forecast_full['yhat_lower']]),
            'yhat_upper': pd.concat([pd.Series([None]*len(df_resampled)), forecast_full['yhat_upper']])
        }).sort_values('ds').reset_index(drop=True)

        # Convert datetime objects to string for JSON serialization
        plot_data['ds'] = plot_data['ds'].dt.strftime('%Y-%m-%d')

        # Convert DataFrame to dictionary and then to JSON
        chart_data_json = json.dumps(plot_data.to_dict(orient='list'))


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
        <title>Walmart Sales Forecaster - Forecast Visualization</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f0f4f8; /* Light blue-gray background */
            }}
            .chart-container {{
                position: relative;
                height: 400px; /* Fixed height for the chart */
                width: 100%;
                margin-top: 2rem;
                margin-bottom: 2rem;
            }}
        </style>
    </head>
    <body class="flex items-center justify-center min-h-screen p-4">
        <div class="bg-white p-8 rounded-xl shadow-lg max-w-4xl w-full text-center">
            <h1 class="text-4xl font-bold text-gray-800 mb-4">
                Walmart Sales Forecaster - Forecast Visualization
            </h1>
            {"<p class='text-red-500 text-md mt-4 font-semibold'>" + error_message + "</p>" if error_message else "<p class='text-lg text-gray-600'>Data loaded, preprocessed, and sales forecast generated.</p>"}
            {df_summary_html if not error_message else ""}
            {forecast_summary_html if not error_message else ""}

            {"" if not error_message else ""}
            {"<h3 class='text-2xl font-semibold text-gray-700 mt-8 mb-4'>Weekly Sales: Historical & Forecasted</h3>" if not error_message else ""}
            {"<div class='chart-container mx-auto'>" if not error_message else ""}
            {"<canvas id='salesChart'></canvas>" if not error_message else ""}
            {"</div>" if not error_message else ""}

            <p class="text-sm text-gray-500 mt-6">
                We now have seasonal sales predictions for 2025 and a visual representation!
            </p>
        </div>

        {"" if not error_message else ""}
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const chartData = {chart_data_json}; // Data passed from Flask

                if (Object.keys(chartData).length > 0 && chartData.ds && chartData.y_actual && chartData.y_forecast) {{
                    const ctx = document.getElementById('salesChart').getContext('2d');
                    new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: chartData.ds,
                            datasets: [
                                {{
                                    label: 'Actual Weekly Sales',
                                    data: chartData.y_actual,
                                    borderColor: 'rgb(75, 192, 192)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                    fill: false,
                                    tension: 0.1,
                                    spanGaps: true // Connects nulls if there are gaps in actuals
                                }},
                                {{
                                    label: 'Predicted Weekly Sales',
                                    data: chartData.y_forecast,
                                    borderColor: 'rgb(255, 99, 132)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                    fill: false,
                                    tension: 0.1,
                                    spanGaps: true // Connects nulls if there are gaps in forecasts
                                }},
                                {{
                                    label: 'Forecast Confidence Interval (Lower)',
                                    data: chartData.yhat_lower,
                                    borderColor: 'rgba(255, 99, 132, 0.3)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                    fill: '-1', // Fill below this line, connecting to the previous dataset
                                    tension: 0.1,
                                    pointRadius: 0,
                                    borderWidth: 0,
                                    hidden: true // Hide by default, can be toggled
                                }},
                                {{
                                    label: 'Forecast Confidence Interval (Upper)',
                                    data: chartData.yhat_upper,
                                    borderColor: 'rgba(255, 99, 132, 0.3)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                    fill: '1', // Fill above this line, connecting to the previous dataset
                                    tension: 0.1,
                                    pointRadius: 0,
                                    borderWidth: 0,
                                    hidden: true // Hide by default, can be toggled
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false, // Allows chart to fill container height
                            scales: {{
                                x: {{
                                    type: 'time',
                                    time: {{
                                        unit: 'month',
                                        tooltipFormat: 'MMM DD, YYYY',
                                        displayFormats: {{
                                            month: 'MMM YYYY'
                                        }}
                                    }},
                                    title: {{
                                        display: true,
                                        text: 'Date'
                                    }}
                                }},
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: 'Weekly Sales ($)'
                                    }},
                                    ticks: {{
                                        callback: function(value, index, values) {{
                                            return '$' + value.toLocaleString();
                                        }}
                                    }}
                                }}
                            }},
                            plugins: {{
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            let label = context.dataset.label || '';
                                            if (label) {{
                                                label += ': ';
                                            }}
                                            if (context.parsed.y !== null) {{
                                                label += '$' + context.parsed.y.toLocaleString();
                                            }}
                                            return label;
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }});
                }} else {{
                    console.error("Chart data is missing or malformed:", chartData);
                }}
            }});
        </script>
    </body>
    </html>
    """
    return render_template_string(html_content)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
