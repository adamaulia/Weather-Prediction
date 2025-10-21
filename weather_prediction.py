import pandas as pd
import numpy as np
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
            Initialize the WeatherForecast class with configuration files.
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
            Preprocess the raw hourly weather data for modeling.
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
            Split the preprocessed data into training and testing datasets
        """
        
        train_size = self.ml_config.get_values('ml.train_size',0.8)
        end_train = int(self.df_pre.shape[0]*train_size)

        self.data_train = self.df_pre.iloc[:end_train]
        self.data_test  = self.df_pre.iloc[end_train:] 
        
        self.logger.info(f"train data shape : {self.data_train.shape}")
        self.logger.info(f"test data shape : {self.data_test.shape}")
    
    def fit_model(self, data_train = None):
        """
            Fit the SARIMAX model to the training dataset.
            
        Args:
            data_train (_type_, optional): _description_. Defaults to None.
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
            Evaluate the fitted model on the test dataset.
        Args:
        
            len_data_test (_type_, optional): _description_. Defaults to None. hours of prediction.
            
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
            Train and validate the SARIMAX model using the training and testing datasets.
        """

        self.logger.info(f"training and validating model")
        
        self.fit_model(data_train=self.data_train)
        self.predict_model(len_data_test=len(self.data_test))
        

    def forecast_future_horizon(self):
        """ 
            Forecast future weather conditions for a specified horizon.
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
            Evaluate the performance of the SARIMAX model.
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
            Main entry point for the weather forecasting pipeline.
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
