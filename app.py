# app.py
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import os
from prophet import Prophet
import logging
import warnings
import json
import numpy as np # For numerical operations

# Suppress cmdstanpy warnings (often verbose from Prophet)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# Suppress specific pandas FutureWarnings if they arise
warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize the Flask application
app = Flask(__name__)

# Define the path to your CSV file
DATA_FILE = os.path.join(os.path.dirname(__file__), 'Walmart_customer_purchases.csv')

# Global variable to store the preprocessed data (loaded once at startup)
df_resampled_global = None

# --- Data Loading and Preprocessing Function (runs once at startup) ---
def load_and_preprocess_data():
    global df_resampled_global
    if df_resampled_global is not None:
        return df_resampled_global # Data already loaded

    print(f"Loading and preprocessing data from: {DATA_FILE}")
    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"The file '{os.path.basename(DATA_FILE)}' was not found at the expected path.")

        df = pd.read_csv(DATA_FILE)
        df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])
        df_resampled = df.set_index('Purchase_Date').resample('W')['Purchase_Amount'].sum().reset_index()
        df_resampled = df_resampled.rename(columns={'Purchase_Date': 'ds', 'Purchase_Amount': 'y'})
        df_resampled['ds'] = pd.to_datetime(df_resampled['ds'])
        df_resampled['y'] = pd.to_numeric(df_resampled['y'])
        df_resampled = df_resampled[df_resampled['y'] >= 0]

        df_resampled_global = df_resampled # Store in global variable
        print("Data loaded and preprocessed successfully.")
        return df_resampled_global

    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None

# --- Route for the Home Page ---
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Handles displaying the forecast page, processing user input for forecast horizon.
    """
    df_summary_html = ""
    forecast_summary_html = ""
    chart_data_json = "{}"
    error_message = None
    forecast_periods_input = 52 # Default to 52 weeks (approx. 1 year)

    df_resampled = load_and_preprocess_data()

    if df_resampled is None:
        error_message = "Failed to load historical sales data. Please check the CSV file."
        print(f"Home route error: {error_message}")
        return render_template_string(generate_html(df_summary_html, forecast_summary_html, chart_data_json, error_message, forecast_periods_input))

    if request.method == 'POST':
        try:
            forecast_periods_input = int(request.form.get('forecast_periods', 52))
            if forecast_periods_input <= 0:
                raise ValueError("Forecast periods must be a positive integer.")
        except ValueError as e:
            error_message = f"Invalid input for forecast periods: {e}. Please enter a positive whole number."
            forecast_periods_input = 52 # Reset to default on error
            print(f"Input error: {error_message}")

    try:
        # --- Data Summary for Display ---
        df_summary_html = f"""
        <h3 class="text-2xl font-semibold text-gray-700 mt-6 mb-4">Weekly Sales Data Summary</h3>
        <p class="text-md text-gray-600 mb-2">Total raw records loaded: **{len(df_resampled_global)}**</p>
        <p class="text-md text-gray-600 mb-2">Date Range of Raw Data: **{df_resampled_global['ds'].min().strftime('%Y-%m-%d')}** to **{df_resampled_global['ds'].max().strftime('%Y-%m-%d')}**</p>
        <p class="text-md text-gray-600 mb-4">Aggregated Weekly Sales Records: **{len(df_resampled_global)}**</p>
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
        for _, row in df_resampled_global.head(5).iterrows():
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
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode='multiplicative')
        model.fit(df_resampled_global)

        # Create a DataFrame with future dates based on user input
        future = model.make_future_dataframe(periods=forecast_periods_input, freq='W')

        # Make predictions for the entire future dataframe (including historical dates)
        forecast_full = model.predict(future)

        # --- Dynamic Season-wise Prediction Aggregation ---
        # Get the max date from the historical data
        last_historical_date = df_resampled_global['ds'].max()
        # Filter forecasts to only include dates *after* the last historical date
        forecast_future_only = forecast_full[forecast_full['ds'] > last_historical_date].copy()

        if not forecast_future_only.empty:
            forecast_future_only['quarter'] = forecast_future_only['ds'].dt.quarter
            forecast_future_only['year'] = forecast_future_only['ds'].dt.year

            # Ensure all quarters of forecasted years are present, even if no prediction falls into them
            # This handles cases where forecast_periods_input is small and doesn't cover full quarters
            all_years_in_forecast = forecast_future_only['year'].unique()
            seasonal_forecast_data = []

            for year in all_years_in_forecast:
                for quarter_num in range(1, 5):
                    quarter_data = forecast_future_only[(forecast_future_only['year'] == year) & (forecast_future_only['quarter'] == quarter_num)]
                    predicted_sales = quarter_data['yhat'].sum()
                    if predicted_sales > 0: # Only include if there's actual prediction for the quarter
                        seasonal_forecast_data.append({
                            'year': year,
                            'quarter': quarter_num,
                            'yhat': predicted_sales
                        })
            seasonal_forecast = pd.DataFrame(seasonal_forecast_data)
            
            if not seasonal_forecast.empty:
                seasonal_forecast['quarter_name'] = seasonal_forecast['quarter'].map({
                    1: 'Q1 (Jan-Mar)',
                    2: 'Q2 (Apr-Jun)',
                    3: 'Q3 (Jul-Sep)',
                    4: 'Q4 (Oct-Dec)'
                })
                # Sort for display
                seasonal_forecast = seasonal_forecast.sort_values(by=['year', 'quarter'])


                # Generate HTML for seasonal forecast
                forecast_summary_html = f"""
                <h3 class="text-2xl font-semibold text-gray-700 mt-8 mb-4">Seasonal Sales Forecast ({forecast_future_only['ds'].min().year}-{forecast_future_only['ds'].max().year})</h3>
                <p class="text-md text-gray-600 mb-4">Predicted total sales for each season in the forecasted period:</p>
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
                for _, row in seasonal_forecast.iterrows():
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
            else:
                forecast_summary_html = "<p class='text-lg text-gray-600 mt-8'>No seasonal forecast generated for the selected period.</p>"
        else:
            forecast_summary_html = "<p class='text-lg text-gray-600 mt-8'>No future forecast generated for the selected number of periods. Please increase the forecast periods.</p>"


        # --- Prepare data for Chart.js visualization ---
        # For plotting, ensure actuals are aligned correctly with forecast
        full_plot_data = pd.merge(df_resampled_global, forecast_full[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='outer')
        full_plot_data = full_plot_data.sort_values('ds').reset_index(drop=True)

        # Fill y_actual with actuals and leave yhat for forecast.
        # This will create a single line chart with historical and forecasted values.
        # We'll use y_actual for historical and yhat for predictions.
        # For the plot, let's create two distinct datasets for actual and predicted.
        # Predicted line should start from the last known actual data point.

        # Find the last historical date
        last_historical_date_plot = df_resampled_global['ds'].max()

        # Data for Actual Sales (historical only)
        actual_sales_plot = full_plot_data[full_plot_data['ds'] <= last_historical_date_plot].copy()
        actual_sales_plot['y_actual_display'] = actual_sales_plot['y']

        # Data for Predicted Sales (starting from last historical point + future)
        predicted_sales_plot = full_plot_data[full_plot_data['ds'] >= last_historical_date_plot].copy()
        # For the predicted line, the first point should be the last actual point
        if not predicted_sales_plot.empty and not actual_sales_plot.empty:
            predicted_sales_plot.loc[predicted_sales_plot['ds'] == last_historical_date_plot, 'yhat'] = \
                actual_sales_plot[actual_sales_plot['ds'] == last_historical_date_plot]['y'].values[0]

        # Combine for chart JSON
        chart_labels = full_plot_data['ds'].dt.strftime('%Y-%m-%d').tolist()
        chart_actuals = actual_sales_plot['y_actual_display'].fillna(np.nan).tolist() # Use NaN for gaps
        chart_forecasts = predicted_sales_plot['yhat'].fillna(np.nan).tolist() # Use NaN for gaps
        chart_forecast_lower = predicted_sales_plot['yhat_lower'].fillna(np.nan).tolist()
        chart_forecast_upper = predicted_sales_plot['yhat_upper'].fillna(np.nan).tolist()

        # To align the data for Chart.js correctly when combining,
        # we need to map the forecast values to the full_plot_data's 'ds' index.
        # Create a unified list of 'ds' labels
        all_ds_labels = full_plot_data['ds'].dt.strftime('%Y-%m-%d').tolist()

        # Create actuals and forecasts lists that match the full_plot_data's length
        actual_data_aligned = [None] * len(all_ds_labels)
        forecast_data_aligned = [None] * len(all_ds_labels)
        forecast_lower_aligned = [None] * len(all_ds_labels)
        forecast_upper_aligned = [None] * len(all_ds_labels)

        # Populate actuals
        for i, ds_date in enumerate(full_plot_data['ds']):
            if ds_date in df_resampled_global['ds'].values:
                actual_data_aligned[i] = df_resampled_global[df_resampled_global['ds'] == ds_date]['y'].iloc[0]

        # Populate forecasts (yhat and intervals)
        for i, ds_date in enumerate(full_plot_data['ds']):
            if ds_date in forecast_full['ds'].values:
                forecast_row = forecast_full[forecast_full['ds'] == ds_date].iloc[0]
                forecast_data_aligned[i] = forecast_row['yhat']
                forecast_lower_aligned[i] = forecast_row['yhat_lower']
                forecast_upper_aligned[i] = forecast_row['yhat_upper']

        # Ensure forecast line starts exactly from the last actual point, not just the forecast start
        # This requires careful handling for the transition point
        if actual_data_aligned and forecast_data_aligned:
             last_actual_index = next((i for i, x in reversed(list(enumerate(actual_data_aligned))) if x is not None), -1)
             if last_actual_index != -1 and last_actual_index + 1 < len(forecast_data_aligned):
                 # Set the first forecasted point to be the last actual point to ensure line continuity
                 forecast_data_aligned[last_actual_index] = actual_data_aligned[last_actual_index]


        chart_data_json = json.dumps({
            'labels': all_ds_labels,
            'actuals': actual_data_aligned,
            'forecasts': forecast_data_aligned,
            'lower_bound': forecast_lower_aligned,
            'upper_bound': forecast_upper_aligned
        })


    except Exception as e:
        error_message = f"An unexpected error occurred during data loading, processing, or forecasting: {e}"
        print(f"General Error: {error_message}")

    return render_template_string(generate_html(df_summary_html, forecast_summary_html, chart_data_json, error_message, forecast_periods_input))

# --- HTML Generation Function ---
def generate_html(df_summary_html, forecast_summary_html, chart_data_json, error_message, forecast_periods_input):
    """Generates the full HTML content for the Flask application."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Walmart Sales Forecaster</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                background-color: #f0f4f8;
            }}
            .chart-container {{
                position: relative;
                height: 450px; /* Increased height for better visualization */
                width: 100%;
                margin-top: 2rem;
                margin-bottom: 2rem;
            }}
        </style>
    </head>
    <body class="flex items-center justify-center min-h-screen p-4">
        <div class="bg-white p-8 rounded-xl shadow-lg max-w-4xl w-full">
            <h1 class="text-4xl font-bold text-gray-800 mb-4 text-center">
                Walmart Sales Forecaster
            </h1>
            <p class="text-lg text-gray-600 mb-6 text-center">
                Predict future sales based on historical data.
            </p>

            {"<p class='text-red-500 text-md mt-4 font-semibold text-center'>" + error_message + "</p>" if error_message else ""}

            <form method="POST" class="mb-8 p-4 border border-gray-200 rounded-md bg-gray-50 flex flex-col md:flex-row items-center justify-center gap-4">
                <label for="forecast_periods" class="text-gray-700 font-medium whitespace-nowrap">Forecast for (weeks):</label>
                <input type="number" id="forecast_periods" name="forecast_periods"
                       value="{forecast_periods_input}" min="1" step="1"
                       class="mt-1 block w-full md:w-auto px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm text-center"
                       placeholder="e.g., 52 for one year">
                <button type="submit"
                        class="mt-1 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    Generate Forecast
                </button>
            </form>

            {"<p class='text-lg text-gray-600 text-center mb-4'>Data loaded, preprocessed, and sales forecast generated.</p>" if not error_message else ""}

            {df_summary_html if not error_message else ""}
            {forecast_summary_html if not error_message else ""}

            {"" if not error_message else ""}
            {"<h3 class='text-2xl font-semibold text-gray-700 mt-8 mb-4 text-center'>Weekly Sales: Historical & Forecasted</h3>" if not error_message else ""}
            {"<div class='chart-container mx-auto'>" if not error_message else ""}
            {"<canvas id='salesChart'></canvas>" if not error_message else ""}
            {"</div>" if not error_message else ""}

            <p class="text-sm text-gray-500 mt-6 text-center">
                Next, we'll look into adding filtering options and a download feature.
            </p>
        </div>

        {"" if not error_message else ""}
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const chartData = {chart_data_json}; // Data passed from Flask

                if (Object.keys(chartData).length > 0 && chartData.labels && chartData.actuals && chartData.forecasts) {{
                    const ctx = document.getElementById('salesChart').getContext('2d');
                    new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: chartData.labels,
                            datasets: [
                                {{
                                    label: 'Actual Weekly Sales',
                                    data: chartData.actuals,
                                    borderColor: 'rgb(75, 192, 192)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                    fill: false,
                                    tension: 0.1,
                                    pointRadius: 3, // Make actual points visible
                                    pointHoverRadius: 5,
                                    segment: {{
                                        borderColor: ctx => (ctx.p0.parsed.x > (chartData.labels.length - chartData.forecasts.filter(x => x !== null).length -1 ) && ctx.p1.parsed.x > (chartData.labels.length - chartData.forecasts.filter(x => x !== null).length - 1)) ? 'transparent' : 'rgb(75, 192, 192)' // Hide line for forecast part
                                    }}
                                }},
                                {{
                                    label: 'Predicted Weekly Sales',
                                    data: chartData.forecasts,
                                    borderColor: 'rgb(255, 99, 132)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                    fill: false,
                                    tension: 0.1,
                                    borderDash: [5, 5], // Dashed line for forecast
                                    pointRadius: 0, // No points for forecast
                                    segment: {{
                                        borderColor: ctx => (ctx.p0.parsed.x < (chartData.labels.length - chartData.forecasts.filter(x => x !== null).length -1 ) ) ? 'transparent' : 'rgb(255, 99, 132)' // Only show line for forecast part
                                    }}
                                }},
                                {{
                                    label: 'Forecast Confidence Interval (Lower)',
                                    data: chartData.lower_bound,
                                    borderColor: 'rgba(255, 99, 132, 0.3)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                    fill: '-1', // Fill below this line, connecting to the previous dataset
                                    tension: 0.1,
                                    pointRadius: 0,
                                    borderWidth: 0,
                                    hidden: false // Show by default for confidence interval
                                }},
                                {{
                                    label: 'Forecast Confidence Interval (Upper)',
                                    data: chartData.upper_bound,
                                    borderColor: 'rgba(255, 99, 132, 0.3)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                    fill: '1', // Fill above this line, connecting to the previous dataset
                                    tension: 0.1,
                                    pointRadius: 0,
                                    borderWidth: 0,
                                    hidden: false // Show by default for confidence interval
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
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
                                    beginAtZero: false, // Allow Y-axis to not start at zero if values are large
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
                                    mode: 'index',
                                    intersect: false,
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
                                }},
                                legend: {{
                                    display: true,
                                    position: 'top'
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
    return html_content

# Run the application
if __name__ == '__main__':
    # Load data once when the application starts
    load_and_preprocess_data()
    app.run(debug=True)
