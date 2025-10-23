import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os 

# skforecast imports
import skforecast
from skforecast.datasets import fetch_dataset
from skforecast.plot import set_dark_theme
from skforecast.sarimax import Sarimax
from skforecast.recursive import ForecasterSarimax
from skforecast.model_selection import TimeSeriesFold
from skforecast.model_selection import backtesting_sarimax
from skforecast.model_selection import grid_search_sarimax
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from helper import YamlHelper
import logging


# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger

class WeatherForecast():
    """
        generate hourly prediciton 

    """
    
    def __init__(self, config_file: str = "config_weather.yaml", config_predict : str = "config_weatherpredict.yaml"):
        """
            Initializes the WeatherForecast class by loading configuration files and setting up paths and logging.

            Parameters:
            - config_file (str): Path to the YAML configuration file containing weather data export settings.
                                Defaults to 'config_weather.yaml'.
            - config_predict (str): Path to the YAML configuration file containing machine learning prediction settings.
                                    Defaults to 'config_weatherpredict.yaml'.

            Actions:
            - Sets up a logger for tracking execution and debugging.
            - Logs the start time of the weather prediction process.
            - Loads configuration values using YamlHelper for both data and ML settings.
            - Determines the project directory and constructs paths for data storage and raw hourly weather data.
            - Logs the resolved paths for transparency and debugging.
        """
        
        self.logger = logging.getLogger(__name__)

        self.logger.info("="*60)
        self.logger.info(f"hourly weather prediction started at {datetime.now()}")
         
        self.data_config = YamlHelper(config_file)
        # data directory 
        self.project_dir =  os.path.dirname(os.path.realpath(__file__)) 
        self.data_path  = os.path.join(self.project_dir, self.data_config.get_values('export_weather.directory', False))
        self.raw_hourlydata_path = os.path.join(self.data_path, '.'.join([self.data_config.get_values('export_weather.filename_hourly', False),self.data_config.get_values('export_weather.format', False),]))
        
        self.ml_config = YamlHelper(config_predict)
        
        self.logger.info(f"raw hourly data path: {self.raw_hourlydata_path}")
        self.logger.info("project directory: {}".format(self.project_dir))
        
         
    
    def preprocessing(self):
        """
            Loads and preprocesses raw hourly weather data for downstream modeling.

            Actions:
            - Reads the raw hourly weather data CSV file from the configured path.
            - Logs the initial shape of the dataset.
            - Filters out forecasted data (rows where 'source' == 'fcst').
            - Selects relevant columns: 'datetime_full' and 'temp'.
            - Converts temperature values to integers.
            - Parses 'datetime_full' into datetime objects and sets it as the index.
            - Resamples the data to ensure an hourly frequency.
            - Logs the shape of the preprocessed DataFrame.

            Result:
            - Stores the cleaned and resampled DataFrame in `self.df_pre` for use in modeling.
        """
        
        
        df = pd.read_csv(self.raw_hourlydata_path)
        
        self.logger.info(f"raw data shape : {df.shape}")
        
        # remove forecast data from dataset 
        self.df_pre =  df[df['source']!='fcst'][['datetime_full','temp']]
        self.df_pre['temp'] = self.df_pre['temp'].astype(int)
        self.df_pre['datetime_full'] = pd.to_datetime(self.df_pre['datetime_full'])
        self.df_pre = self.df_pre.set_index('datetime_full')
        self.df_pre = self.df_pre.asfreq('H')
        
        self.logger.info(f"preprocessed data shape : {self.df_pre.shape}")
        
    
    def split_train_and_test(self):
        """
            Splits the preprocessed hourly weather data into training and testing sets for model development.

            Actions:
            - Retrieves the training size ratio from the ML configuration (default is 0.8 if not specified).
            - Calculates the index at which to split the data based on the total number of rows.
            - Assigns the first portion of the data to `self.data_train` and the remaining to `self.data_test`.
            - Logs the shapes of both training and testing datasets for verification.

            Result:
            - `self.data_train`: DataFrame containing the training portion of the time series.
            - `self.data_test`: DataFrame containing the testing portion of the time series.
        """
        
        train_size = self.ml_config.get_values('ml.train_size',0.8)
        end_train = int(self.df_pre.shape[0]*train_size)

        self.data_train = self.df_pre.iloc[:end_train]
        self.data_test  = self.df_pre.iloc[end_train:] 
        
        self.logger.info(f"train data shape : {self.data_train.shape}")
        self.logger.info(f"test data shape : {self.data_test.shape}")
    
    def fit_model(self, data_train = None):
        """
            Fits a SARIMAX time series model to the training dataset.

            Parameters:
            - data_train (pd.DataFrame, optional): Custom training dataset to override the default `self.data_train`.
                                                If None, the method uses `self.data_train` as the input.

            Actions:
            - Loads SARIMAX model parameters from the ML configuration file.
            - Fits the SARIMAX model to the provided or default training data.
            - Stores the fitted model in `self.model` for later prediction or evaluation.

            Note:
            - Assumes the training data is indexed by datetime and contains a single 'temp' column.
            - Logs model fitting progress and any relevant diagnostics.
        """

        self.logger.info(f"fitting model")
        self.logger.info(f"training data shape for model fitting: {data_train.shape}")
        
        warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
        self.model = Sarimax(order=(1,1,1), seasonal_order=(1,1,1,24))
        self.model.fit(y=data_train)
        self.model.summary()
        warnings.filterwarnings("default")
        
    
    def predict_model(self, len_data_test = None):
        """
            Generates predictions using the fitted SARIMAX model over a specified time horizon.

            Parameters:
            - len_data_test (int, optional): Number of future time steps (hours) to forecast.
                                            If None, the method should default to the length of `self.data_test`.

            Actions:
            - Logs the prediction duration.
            - Uses the SARIMAX model stored in `self.model` to forecast the specified number of steps.
            - Stores the predictions in `self.predictions_skforecast`.
            - Logs a preview of the predicted values.

            Returns:
            - pd.Series: Forecasted temperature values indexed by datetime.
        """
        self.logger.info(f"predicting model for {len_data_test} hours")
        self.predictions_skforecast = self.model.predict(steps=len_data_test) # days prediction change config in yaml file 
        
        self.logger.info(f"recent predicted values: {self.predictions_skforecast.head()}")
        
        return self.predictions_skforecast
        
    

    def save_model(self):
        pass 
    
    def export_result(self):
        pass 
    
    def train_and_validate_model(self):
        """
            Trains the SARIMAX model on the training dataset and validates it by generating predictions on the test dataset.

            Actions:
            - Logs the start of the training and validation process.
            - Calls `fit_model()` to train the SARIMAX model using `self.data_train`.
            - Calls `predict_model()` to forecast values for the duration of `self.data_test`.
            - Stores the predictions in `self.predictions_skforecast` for further evaluation or visualization.

            Result:
            - A trained SARIMAX model and its predictions over the test horizon.
        """

        self.logger.info(f"training and validating model")
        
        self.fit_model(data_train=self.data_train)
        self.predict_model(len_data_test=len(self.data_test))
        

    def forecast_future_horizon(self):
        """
            Forecasts future weather conditions over a configurable time horizon and exports the results.

            Actions:
            - Logs the start of the forecasting process.
            - Fits a SARIMAX model using the entire preprocessed dataset (`self.df_pre`).
            - Forecasts future temperature values for a number of hours defined in the ML config (`ml.predict_horizon_hour`).
            - Formats the forecast output:
                - Renames the prediction column to 'temp'.
                - Adds a 'label' column to distinguish forecasted data.
                - Rounds and converts temperature values to integers.
            - Combines historical and forecasted data into a single DataFrame (`self.df_post`).
            - Exports the combined dataset to a CSV file at the configured path (`weather_prediction.csv`).
            - Prints the last few rows of the combined DataFrame for quick inspection.

            Result:
            - A CSV file containing both historical and forecasted hourly temperature data, labeled accordingly.
        """
        self.logger.info(f"forecasting future horizon")
        
        self.fit_model(data_train=self.df_pre)
        self.forecast_future = self.predict_model(len_data_test=self.ml_config.get_values('ml.predict_horizon_hour', 3)) # days prediction change config in yaml file
        
        self.forecast_future.columns =['temp']
        self.forecast_future['label'] = 'forecast'
        self.forecast_future['temp'] = self.forecast_future['temp'].round().astype(int)

        self.df_post = self.df_pre.copy()
        self.df_post['label'] = 'historical'
        
        self.df_post = pd.concat([self.df_post, self.forecast_future], axis=0)
        
        self.path_export_address = os.path.join (self.data_path, self.ml_config.get_values("weather_prediction.csv", "weather_prediction.csv"))
        self.df_post.to_csv(self.path_export_address)
        
        print(self.df_post.tail())

    def evaluate_model(self):
        """
        Evaluates the performance of the fitted SARIMAX model using standard regression metrics.

        Actions:
        - Logs the start of the evaluation process.
        - Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE)
        between the actual test data (`self.data_test`) and the model's predictions (`self.predictions_skforecast`).
        - Stores the results in `self.evaluation_results` as a dictionary.
        - Logs the evaluation metrics for transparency and debugging.

        Result:
        - `self.evaluation_results`: Dictionary containing MAE, MSE, and RMSE values.
        """
        self.logger.info(f"evaluating model performance")
        
        mae = mean_absolute_error(self.data_test, self.predictions_skforecast)
        mse = mean_squared_error(self.data_test, self.predictions_skforecast)
        rmse = root_mean_squared_error(self.data_test, self.predictions_skforecast)
        
        self.evaluation_results = {
            'MAE': mae, 
            'MSE': mse,
            'RMSE': rmse
        }
        
        self.logger.info(f"Evaluation Results: {self.evaluation_results}")
        

    def save_time_series_plot(self, df_train, df_pred,  filename='time_series_prediction.png', title='Time Series Prediction'):
        """
        Saves a time series prediction plot from a DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the time series data.
        - x_axis (str): Column name for the x-axis (typically time).
        - y_axis (str): Column name for the y-axis (predicted values).
        - filename (str): Name of the output image file.
        - title (str): Title of the plot.
        """
        
        fig, ax = plt.subplots(figsize=(7, 3))
        df_train.plot(ax=ax, label='train')
        df_pred.plot(ax=ax, label='test')
        
        ax.set_title('Time Series Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(filename)  # You can change the filename and format (e.g., .pdf, .svg)
        plt.close()  # Optional: closes the figure to free memory


    def main(self):
        """
            Executes the full weather forecasting pipeline from data preprocessing to visualization.

            Workflow:
            - Logs the start of the pipeline.
            - Preprocesses raw hourly weather data (`self.preprocessing()`).
            - Splits the cleaned data into training and testing sets (`self.split_train_and_test()`).
            - Trains the SARIMAX model and validates it on the test set (`self.train_and_validate_model()`).
            - Evaluates model performance using MAE, MSE, and RMSE (`self.evaluate_model()`).
            - Forecasts future weather conditions for a configured horizon (`self.forecast_future_horizon()`).
            - Generates and saves a time series plot comparing historical and forecasted data (`self.save_time_series_plot()`).

            Result:
            - A complete forecasting cycle with evaluation metrics and a visual output saved as 'time_series_prediction.png'.
        """
        
        self.logger.info(f"weather forecasting main pipeline started")
        
        self.preprocessing()
        self.split_train_and_test()
        self.train_and_validate_model()
        self.evaluate_model()   
        self.forecast_future_horizon()
        self.save_time_series_plot(self.df_post, self.forecast_future, filename='time_series_prediction.png', title='Time Series Prediction')
        
         
        
if __name__ == "__main__":
    wf = WeatherForecast()
    wf.main()
