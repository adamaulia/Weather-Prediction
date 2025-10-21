from flask import Flask
from weather_prediction import WeatherForecast
from weather_api_client import run_with_config

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Flask!'

@app.route('/predict_weather')
def predict_weather():
    wf = WeatherForecast()
    wf.main()
    
    return 'Weather prediction completed!'

@app.route('/weather_get_data')
def weather_get_data():
    run_with_config()
    return 'Weather data retrieval completed!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')