import os
import yaml
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import requests
from helper import YamlHelper
import logging
import logging.handlers
import json



# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger

# Load environment variables from .env file
load_dotenv()

class VisualCrossingWeatherAPI:
    """
    A client for the Visual Crossing Weather API to fetch current and historical weather data.
    """
    
    def __init__(self, api_key: Optional[str] = None, config_file: str = "config_weather.yaml"):
        """
        Initialize the weather API client.
        
        Args:
            api_key (str, optional): Your Visual Crossing Weather API key.
                                   If not provided, will try to load from VISUAL_CROSSING_API_KEY environment variable.
            config_file (str): Path to the YAML configuration file
        """
        self.logger = logging.getLogger(__name__)

        self.config = YamlHelper(config_file)
        self.api_key = api_key or os.getenv('VISUAL_CROSSING_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Either pass it as a parameter or set VISUAL_CROSSING_API_KEY environment variable."
            )
        
        self.base_url = self.config.get_values('api.base_url', "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline")
        self.timeout = self.config.get_values('api.timeout', 30)
        
    def get_historical_days(self) -> int:
        """Get the default number of historical days from config"""
        return self.config.get_values('historical.days_back', 7)
    
    def get_current_weather(self, location: Optional[str] = None, include_forecast: Optional[bool] = None) -> Dict[Any, Any]:
        """
        Get current weather data for a location.
        
        Args:
            location (str, optional): Location (city, address, or coordinates). 
                                    Uses config primary location if not provided.
            include_forecast (bool, optional): Whether to include forecast data.
                                             Uses config setting if not provided.
            
        Returns:
            dict: Weather data response
        """
        self.logger.info("="*60)
        self.logger.info(f"get current weather") 
        
        # set primary location 
        location = location or self.config.get_values('location.primary', 'New York, NY')
        include_forecast = include_forecast if include_forecast is not None else self.config.get_values('data_options.include_forecast', False)
        
        self.logger.info(f"use location : {location}") 
        self.logger.info(f"include forecast {include_forecast}")
        
        url = f"{self.base_url}/{location}"
        
        params = {
            'key': self.api_key,
            'include': 'current',
            'contentType': 'json',
            'unitGroup': self.config.get_values('data_options.units', 'metric')
        }
        
        if include_forecast:
            params['include'] = 'days,current'
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # print(f"Error fetching current weather for {location}: {e}")
            self.logger.error(f"Error fetching current weather for {location}: {e}")
            return {}
        
    def get_historical_data(self, location: Optional[str] = None, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None, days_back: Optional[int] = None, 
                          include_hourly: Optional[bool] = None) -> Dict[Any, Any]:
        """
        Get historical weather data for a location and date range.
        
        Args:
            location (str, optional): Location. Uses config primary location if not provided.
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            days_back (int, optional): Number of days back from today. Uses config setting if not provided.
            include_hourly (bool, optional): Include hourly data. Uses config setting if not provided.
            
        Returns:
            dict: Historical weather data response
        """
        
        location = location or self.config.get_values('location.primary', 'New York, NY')
        include_hourly = include_hourly if include_hourly is not None else self.config.get_values('historical.include_hourly', False)
        
        self.logger.info(f"get historical data : {location}") 
        self.logger.info(f"include  hourly data :  {include_hourly}, if false then daily ")
        self.logger.info(f"start date : {start_date}")
        self.logger.info(f"end date : {end_date}")
        
        self.logger.info(f"default historical back date : {self.config.get_values('historical.days_back', 7)}")
        
        
        # If specific dates not provided, calculate from days_back
        if not start_date or not end_date:
            days_back = days_back or self.config.get_values('historical.days_back', 7)
            
            # Limit days for hourly data to prevent large responses
            if include_hourly:
                max_hourly_days = self.config.get_values('historical.hourly.max_days_hourly', 15)
                days_back = min(days_back, max_hourly_days)
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"
        
        # Determine what data to include
        include_params = ['days']
        if include_hourly:
            include_params.append('hours')
        
        params = {
            'key': self.api_key,
            'include': ','.join(include_params),
            'contentType': 'json',
            'unitGroup': self.config.get_values('data_options.units', 'metric')
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching historical data for {location}: {e}")
            return {}
        
    def get_hourly_historical_data(self, location: Optional[str] = None, start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None, days_back: Optional[int] = None) -> Dict[Any, Any]:
        """
        Get hourly historical weather data (dedicated method for hourly data).
        
        Args:
            location (str, optional): Location. Uses config primary location if not provided.
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            days_back (int, optional): Number of days back from today. Uses config setting if not provided.
            
        Returns:
            dict: Hourly historical weather data response
        """
        return self.get_historical_data(
            location=location,
            start_date=start_date,
            end_date=end_date,
            days_back=days_back,
            include_hourly=True
        )        
        
    def convert_to_dataframe(self, weather_data: Dict[Any, Any], data_type: str = 'daily') -> pd.DataFrame:
        """
        Convert weather data to pandas DataFrame.
        
        Args:
            weather_data (dict): Weather data from API response
            data_type (str): Type of data ('daily', 'hourly', 'current')
            
        Returns:
            pd.DataFrame: Weather data as DataFrame
        """
        if not weather_data or 'days' not in weather_data:
            return pd.DataFrame()
        
        if data_type == 'daily':
            return pd.json_normalize(weather_data['days'])
        elif data_type == 'hourly':
            hourly_data = []
            for day in weather_data['days']:
                if 'hours' in day:
                    date = day['datetime']
                    for hour in day['hours']:
                        # Add date information to each hour
                        hour_record = hour.copy()
                        hour_record['date'] = date
                        hour_record['datetime_full'] = f"{date} {hour.get('datetime', '00:00:00')}"
                        hourly_data.append(hour_record)
            return pd.json_normalize(hourly_data)
        elif data_type == 'current':
            if 'currentConditions' in weather_data:
                return pd.json_normalize([weather_data['currentConditions']])
        elif data_type == 'combined':
            # Create combined daily and hourly data
            daily_df = pd.json_normalize(weather_data['days'])
            hourly_data = []
            
            for day in weather_data['days']:
                if 'hours' in day:
                    for hour in day['hours']:
                        hour_record = hour.copy()
                        hour_record['date'] = day['datetime']
                        hour_record['datetime_full'] = f"{day['datetime']} {hour.get('datetime', '00:00:00')}"
                        # Add daily summary data to each hour
                        hour_record['daily_temp_max'] = day.get('tempmax')
                        hour_record['daily_temp_min'] = day.get('tempmin')
                        hour_record['daily_conditions'] = day.get('conditions')
                        hourly_data.append(hour_record)
            
            hourly_df = pd.json_normalize(hourly_data)
            return {'daily': daily_df, 'hourly': hourly_df}
        
        return pd.DataFrame()
    
    def set_output_filename(self):
        """
            this function generate filename 
        """
        # current file 
        project_dir =  os.path.dirname(os.path.realpath(__file__)) 
        
        # data directory 
        data_dir = os.path.join(project_dir, self.config.get_values('export_weather.directory', False))
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # file format csv/xlsx/ ... 
        fileformat = self.config.get_values('export_weather.format', False) 
        
        print(f"include hourly : {self.config.get_values('historical.include_hourly')}")
        print(f"data dir : {data_dir}")
        print(f"project dir: {project_dir}")
        
        
        if self.config.get_values('historical.include_hourly', False):
        # filename 
            full_path_filename = os.path.join(data_dir, self.config.get_values('export_weather.filename_hourly', False) + "." +fileformat)
        else : 
            full_path_filename = os.path.join(data_dir, self.config.get_values('export_weather.filename_daily', False) + "." +fileformat)
        
        return full_path_filename

    def export_df_to_file(self,df,filename):
        """
            export dataframe to file
        Args:
            df (_type_): _description_
            filename (_type_): _description_
        """
        fileformat = self.config.get_values('export_weather.format', False) 
        
        if fileformat =='csv':
            df.to_csv(filename,index=False)
        elif fileformat =='xlsx':
            df.to_excel(filename,index=False)
        else :
            raise "not supported format"
    
    def run_daily_historical_weather(self):
        """
            run daily historical weather
        Returns:
            _type_: _description_
        """
        # 
        historical_data = self.get_historical_data()
        if historical_data and 'days' in historical_data:
            df = self.convert_to_dataframe(historical_data, 'daily')
            if not df.empty:
                print(f"Daily data retrieved for {len(df)} days")
                print(df[['datetime', 'temp', 'tempmax', 'tempmin', 'conditions']].head())
                
                #save to file 
                if  self.config.get_values('export_weather.enabled', False) :
                    filename = self.set_output_filename()
                    self.export_df_to_file(df,filename)
                else :
                    pass 
                    
                return historical_data, df 
            
            else :
                pass 
        else : 
            # hourly 
            pass 
    
    def run_hourly_historical_weather(self):
        """
            run hourly historical weather
        """
        json_hourly_hist_data = self.get_hourly_historical_data()

        # if hourly_data:
        #     hourly_df = weather.convert_to_dataframe(hourly_data, 'hourly')
        #     print(f"Retrieved {len(hourly_df)} hourly records")
                
        # Check if hourly data is included
        hourly_df = self.convert_to_dataframe(json_hourly_hist_data, 'hourly')
        if not hourly_df.empty:
            print(f"\nHourly data retrieved: {len(hourly_df)} hourly records")
            print(hourly_df[['date', 'datetime', 'temp', 'humidity', 'conditions']].head(10))      
            
            # save to file 
            if self.config.get_values('export_weather.enabled', False) :
                filename = self.set_output_filename()
                self.export_df_to_file(hourly_df,filename)
            else :
                pass 
        else : 
            pass 
                
         

def run_with_config():
    """Example using configuration file settings"""
    print("=== Weather Data Using Config File ===")
    
    # Initialize with config file
    weather = VisualCrossingWeatherAPI()
    

    # export file dir setup 
    project_dir =  os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(project_dir, weather.config.get_values('export_weather.directory', False))
    fileformat = weather.config.get_values('export_weather.format', False) 
    
    # Get current weather for primary location from config
    print(f"\n1. Current weather for primary location")
    current_data = weather.get_current_weather()
    
    if current_data:
        print(f"Temperature: {current_data.get('currentConditions', {}).get('temp', 'N/A')}°")
        print(f"Conditions: {current_data.get('currentConditions', {}).get('conditions', 'N/A')}")
        
    # Get historical data using config settings
    print(f"\n2. Historical data for past {weather.get_historical_days()} days:")
    historical_data = weather.get_historical_data()
    
    if historical_data and 'days' in historical_data:
        df = weather.convert_to_dataframe(historical_data, 'daily')
        if not df.empty:
            print(f"Daily data retrieved for {len(df)} days")
            print(df[['datetime', 'temp', 'tempmax', 'tempmin', 'conditions']].head())
            # save to file 
            if weather.config.get_values('export_weather.enabled', False) :
                filename = os.path.join(data_dir, weather.config.get_values('export_weather.filename_daily', False) + "." +fileformat)
                weather.export_df_to_file(df,filename)

        
        # Check if hourly data is included
        if weather.config.get_values('historical.include_hourly', False):
            hourly_df = weather.convert_to_dataframe(historical_data, 'hourly')

            if not hourly_df.empty:
                print(f"\nHourly data retrieved: {len(hourly_df)} hourly records")
                print(hourly_df[['date', 'datetime', 'temp', 'humidity', 'conditions']].head(10))

                # save to file 
                if weather.config.get_values('export_weather.enabled', False) :
                    filename = os.path.join(data_dir, weather.config.get_values('export_weather.filename_hourly', False) + "." +fileformat)
                    weather.export_df_to_file(hourly_df,filename)
        
if __name__ == "__main__":
    # vwa = VisualCrossingWeatherAPI()
    # vwa.run_with_config()
    # result = vwa.get_current_weather()
    # result = vwa.get_historical_data()
    # print(result)
    # print(pd.DataFrame.from_dict(result))
    # with open('data.json', 'w') as f:
    #     json.dump(result, f)
    
    # print(pd.json_normalize(result))
    
    run_with_config()
    
