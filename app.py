from flask import Flask
from weather_prediction import WeatherForecast
from weather_api_client import run_with_config

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Flask!'

@app.route('/predict_weather')
def predict_weather():
    """
        Flask route to trigger the weather forecasting pipeline.

        Actions:
        - Instantiates the `WeatherForecast` class.
        - Executes the full pipeline via `wf.main()`, which includes:
            - Data preprocessing
            - Train-test split
            - Model training and validation
            - Evaluation
            - Future forecasting
            - Visualization and export

        Returns:
        - None. This route runs the pipeline and saves outputs (e.g., CSV and plot) to disk.
    """
    wf = WeatherForecast()
    wf.main()
    
    return 'Weather prediction completed!'

@app.route('/weather_get_data')
def weather_get_data():
    """
        Retrieves and processes weather data using the configured pipeline.

        Actions:
        - Executes the `run_with_config()` function, which handles data acquisition and any necessary preprocessing.
        - Returns a confirmation message upon successful completion.

        Returns:
        - str: A message indicating that weather data retrieval has finished.
    """
    run_with_config()
    return 'Weather data retrieval completed!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')