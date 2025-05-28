# app.py
from flask import Flask, render_template_string
import pandas as pd
import os # To help with file paths

# Initialize the Flask application
app = Flask(__name__)

# Define the path to your CSV file
# It's good practice to make this robust
DATA_FILE = os.path.join(os.path.dirname(__file__), 'walmart_customer_transactions.csv')

@app.route('/')
def home():
    """
    Loads, preprocesses, and displays a summary of the Walmart sales data.
    """
    df_summary_html = "<p class='text-lg text-gray-600'>No data loaded yet.</p>"
    error_message = None

    try:
        # Load the dataset
        df = pd.read_csv(DATA_FILE)

        # --- Initial Data Inspection and Preprocessing ---
        # 1. Convert 'Purchase_Date' to datetime objects
        df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'])

        # 2. Extract Year and Week of Year for aggregation
        df['Year'] = df['Purchase_Date'].dt.year
        df['Week'] = df['Purchase_Date'].dt.isocalendar().week.astype(int)

        # 3. Aggregate sales data by Year and Week
        # We need to handle potential multiple entries for the same week/year,
        # so we'll sum the Purchase_Amount.
        weekly_sales = df.groupby(['Year', 'Week'])['Purchase_Amount'].sum().reset_index()

        # 4. Sort the data by Year and Week
        weekly_sales = weekly_sales.sort_values(by=['Year', 'Week'])

        # 5. Optional: Create a 'Year_Week' string for easier display/plotting
        weekly_sales['Year_Week'] = weekly_sales['Year'].astype(str) + '-W' + \
                                    weekly_sales['Week'].astype(str).str.zfill(2)

        # Get a summary for display
        df_summary_html = f"""
        <h3 class="text-2xl font-semibold text-gray-700 mt-6 mb-4">Weekly Sales Data Summary</h3>
        <p class="text-md text-gray-600 mb-2">Total records loaded: **{len(df)}**</p>
        <p class="text-md text-gray-600 mb-2">Date Range of Raw Data: **{df['Purchase_Date'].min().strftime('%Y-%m-%d')}** to **{df['Purchase_Date'].max().strftime('%Y-%m-%d')}**</p>
        <p class="text-md text-gray-600 mb-4">Aggregated Weekly Sales Records: **{len(weekly_sales)}**</p>
        <h4 class="text-xl font-medium text-gray-700 mb-3">First 5 Weekly Sales Records:</h4>
        <div class="overflow-x-auto">
            <table class="min-w-full bg-white border border-gray-300 rounded-md">
                <thead>
                    <tr class="bg-gray-100 text-gray-700 uppercase text-sm leading-normal">
                        <th class="py-3 px-6 text-left">Year</th>
                        <th class="py-3 px-6 text-left">Week</th>
                        <th class="py-3 px-6 text-left">Year_Week</th>
                        <th class="py-3 px-6 text-right">Weekly Sales</th>
                    </tr>
                </thead>
                <tbody class="text-gray-600 text-sm font-light">
        """
        for _, row in weekly_sales.head(5).iterrows():
            df_summary_html += f"""
                    <tr class="border-b border-gray-200 hover:bg-gray-50">
                        <td class="py-3 px-6 text-left whitespace-nowrap">{row['Year']}</td>
                        <td class="py-3 px-6 text-left">{row['Week']}</td>
                        <td class="py-3 px-6 text-left">{row['Year_Week']}</td>
                        <td class="py-3 px-6 text-right">${row['Purchase_Amount']:,.2f}</td>
                    </tr>
            """
        df_summary_html += """
                </tbody>
            </table>
        </div>
        """

    except FileNotFoundError:
        error_message = f"Error: '{os.path.basename(DATA_FILE)}' not found in the same directory as app.py. Please ensure the CSV file is there."
        print(error_message)
    except Exception as e:
        error_message = f"An error occurred during data loading or processing: {e}"
        print(error_message)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Walmart Sales Forecaster - Data Loaded</title>
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
                Walmart Sales Forecaster - Data Ready!
            </h1>
            <p class="text-lg text-gray-600">
                Data loading and initial preprocessing complete.
            </p>
            {"<p class='text-red-500 text-md mt-4'>" + error_message + "</p>" if error_message else ""}
            {df_summary_html}
            <p class="text-sm text-gray-500 mt-6">
                Next, we'll begin building our forecasting model.
            </p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)