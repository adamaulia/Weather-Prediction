import requests
import json
import os
import yaml
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class WeatherConfig:
    """Configuration manager for weather API settings"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_file (str): Path to the YAML configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[Any, Any]:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(self.config_file):
                print(f"Warning: Config file '{self.config_file}' not found. Using defaults.")
                return self._get_default_config()
            
            with open(self.config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config if config else self._get_default_config()
        except Exception as e:
            print(f"Error loading config file: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[Any, Any]:
        """Return default configuration"""
        return {
            'location': {'primary': 'New York, NY'},
            'historical': {'days_back': 7},
            'data_options': {'units': 'metric', 'include_current': True},
            'output': {'pretty_print': True},
            'processing': {'auto_dataframe': True, 'round_decimals': 2}
        }
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path (str): Dot-separated path to the config value (e.g., 'location.primary')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class VisualCrossingWeatherAPI:
    """
    A client for the Visual Crossing Weather API to fetch current and historical weather data.
    """
    
    def __init__(self, api_key: Optional[str] = None, config_file: str = "config.yaml"):
        """
        Initialize the weather API client.
        
        Args:
            api_key (str, optional): Your Visual Crossing Weather API key.
                                   If not provided, will try to load from VISUAL_CROSSING_API_KEY environment variable.
            config_file (str): Path to the YAML configuration file
        """
        self.config = WeatherConfig(config_file)
        self.api_key = api_key or os.getenv('VISUAL_CROSSING_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Either pass it as a parameter or set VISUAL_CROSSING_API_KEY environment variable."
            )
        
        self.base_url = self.config.get('api.base_url', "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline")
        self.timeout = self.config.get('api.timeout', 30)
    
    def get_primary_location(self) -> str:
        """Get the primary location from config"""
        return self.config.get('location.primary', 'New York, NY')
    
    def get_historical_days(self) -> int:
        """Get the default number of historical days from config"""
        return self.config.get('historical.days_back', 7)
    
    def get_alternative_locations(self) -> List[str]:
        """Get list of alternative locations from config"""
        return self.config.get('location.alternatives', [])
    
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
        location = location or self.get_primary_location()
        include_forecast = include_forecast if include_forecast is not None else self.config.get('data_options.include_forecast', False)
        
        url = f"{self.base_url}/{location}"
        
        params = {
            'key': self.api_key,
            'include': 'current',
            'contentType': 'json',
            'unitGroup': self.config.get('data_options.units', 'metric')
        }
        
        if include_forecast:
            params['include'] = 'days,current'
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather for {location}: {e}")
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
        location = location or self.get_primary_location()
        include_hourly = include_hourly if include_hourly is not None else self.config.get('historical.include_hourly', False)
        
        # If specific dates not provided, calculate from days_back
        if not start_date or not end_date:
            days_back = days_back or self.get_historical_days()
            
            # Limit days for hourly data to prevent large responses
            if include_hourly:
                max_hourly_days = self.config.get('historical.hourly.max_days_hourly', 15)
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
            'unitGroup': self.config.get('data_options.units', 'metric')
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching historical data for {location}: {e}")
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
    
    def get_weather_for_date(self, location: str, date: str) -> Dict[Any, Any]:
        """
        Get weather data for a specific date.
        
        Args:
            location (str): Location (city, address, or coordinates)
            date (str): Date in YYYY-MM-DD format
            
        Returns:
            dict: Weather data for the specified date
        """
        url = f"{self.base_url}/{location}/{date}"
        
        params = {
            'key': self.api_key,
            'include': 'days',
            'contentType': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather for date: {e}")
            return {}
    
    def get_hourly_data(self, location: str, start_date: str, end_date: str) -> Dict[Any, Any]:
        """
        Get hourly weather data for a location and date range.
        
        Args:
            location (str): Location (city, address, or coordinates)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Hourly weather data response
        """
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"
        
        params = {
            'key': self.api_key,
            'include': 'hours',
            'contentType': 'json'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching hourly data: {e}")
            return {}
    
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
    
    def filter_hourly_by_time(self, hourly_df: pd.DataFrame, start_hour: int = 6, end_hour: int = 18) -> pd.DataFrame:
        """
        Filter hourly data by time range (e.g., daylight hours).
        
        Args:
            hourly_df (pd.DataFrame): Hourly weather DataFrame
            start_hour (int): Start hour (0-23)
            end_hour (int): End hour (0-23)
            
        Returns:
            pd.DataFrame: Filtered hourly data
        """
        if hourly_df.empty or 'datetime' not in hourly_df.columns:
            return hourly_df
        
        # Extract hour from datetime string (format: "HH:MM:SS")
        try:
            hourly_df['hour'] = hourly_df['datetime'].str.split(':').str[0].astype(int)
            filtered = hourly_df[(hourly_df['hour'] >= start_hour) & (hourly_df['hour'] <= end_hour)]
            return filtered.drop('hour', axis=1)  # Remove temporary hour column
        except Exception as e:
            print(f"Error filtering by time: {e}")
            return hourly_df
    
    def get_hourly_summary(self, hourly_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics from hourly data.
        
        Args:
            hourly_df (pd.DataFrame): Hourly weather DataFrame
            
        Returns:
            dict: Summary statistics
        """
        if hourly_df.empty:
            return {}
        
        summary = {}
        
        # Temperature statistics
        if 'temp' in hourly_df.columns:
            summary['temperature'] = {
                'avg': round(hourly_df['temp'].mean(), 2),
                'min': hourly_df['temp'].min(),
                'max': hourly_df['temp'].max(),
                'std': round(hourly_df['temp'].std(), 2)
            }
        
        # Humidity statistics
        if 'humidity' in hourly_df.columns:
            summary['humidity'] = {
                'avg': round(hourly_df['humidity'].mean(), 2),
                'min': hourly_df['humidity'].min(),
                'max': hourly_df['humidity'].max()
            }
        
        # Wind speed statistics
        if 'windspeed' in hourly_df.columns:
            summary['wind_speed'] = {
                'avg': round(hourly_df['windspeed'].mean(), 2),
                'min': hourly_df['windspeed'].min(),
                'max': hourly_df['windspeed'].max()
            }
        
        # Most common conditions
        if 'conditions' in hourly_df.columns:
            summary['most_common_condition'] = hourly_df['conditions'].mode().iloc[0] if not hourly_df['conditions'].mode().empty else 'N/A'
        
        summary['total_hours'] = len(hourly_df)
        summary['date_range'] = {
            'start': hourly_df['date'].min() if 'date' in hourly_df.columns else 'N/A',
            'end': hourly_df['date'].max() if 'date' in hourly_df.columns else 'N/A'
        }
        
        return summary

# Example usage functions
def example_with_config():
    """Example using configuration file settings"""
    print("=== Weather Data Using Config File ===")
    
    # Initialize with config file
    weather = VisualCrossingWeatherAPI()
     
    # Get current weather for primary location from config
    print(f"\n1. Current weather for primary location: {weather.get_primary_location()}")
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
        
        # Check if hourly data is included
        if weather.config.get('historical.include_hourly', False):
            hourly_df = weather.convert_to_dataframe(historical_data, 'hourly')
            if not hourly_df.empty:
                print(f"\nHourly data retrieved: {len(hourly_df)} hourly records")
                print(hourly_df[['date', 'datetime', 'temp', 'humidity', 'conditions']].head(10))
                                
                 # save to file 
                if weather.config.get('export_weather.enabled', False) :
                    
                    project_dir =  os.path.dirname(os.path.realpath(__file__))
                    data_dir = os.path.join(project_dir, weather.config.get('export_weather.directory', False))
                    fileformat = weather.config.get('export_weather.format', False) 
                    filename = os.path.join(data_dir, weather.config.get('export_weather.filename', False) + "." +fileformat)
                    
                    if fileformat == "csv":
                        hourly_df.to_csv(filename,index=False)
                    if filename == "xlsx":
                        hourly_df.to_excel(filename,index=False)
                    else : 
                        print("other format not yet supported")
                    
                    print('filename : ',filename)
                
                # Get hourly summary
                summary = weather.get_hourly_summary(hourly_df)
                print(f"\nHourly Summary:")
                print(f"- Total hours: {summary.get('total_hours', 0)}")
                print(f"- Temperature range: {summary.get('temperature', {}).get('min', 'N/A')}° - {summary.get('temperature', {}).get('max', 'N/A')}°")
                print(f"- Average temperature: {summary.get('temperature', {}).get('avg', 'N/A')}°")
                print(f"- Most common condition: {summary.get('most_common_condition', 'N/A')}")
                

            
                    # check location 
                    # if  not os.path.exists(directory):
                    #     os.makedirs(directory)
    
    # Process alternative locations from config
    print("\n3. Weather for alternative locations:")
    for location in weather.get_alternative_locations()[:2]:  # Limit to first 2
        current = weather.get_current_weather(location)
        if current and 'currentConditions' in current:
            temp = current['currentConditions'].get('temp', 'N/A')
            conditions = current['currentConditions'].get('conditions', 'N/A')
            print(f"{location}: {temp}° - {conditions}")

def example_hourly_analysis():
    """Example focusing on hourly data analysis"""
    print("\n=== Hourly Weather Data Analysis ===")
    
    weather = VisualCrossingWeatherAPI()
    
    # Get hourly historical data for the past 3 days
    print(f"Getting hourly data for {weather.get_primary_location()} (past 3 days)")
    hourly_data = weather.get_hourly_historical_data(days_back=3)
    
    if hourly_data:
        # Convert to DataFrame
        hourly_df = weather.convert_to_dataframe(hourly_data, 'hourly')
        
        if not hourly_df.empty:
            print(f"Retrieved {len(hourly_df)} hourly records")
            
            # Show sample data
            print("\nSample hourly data:")
            print(hourly_df[['date', 'datetime', 'temp', 'humidity', 'windspeed', 'conditions']].head())
            
            # Filter for daylight hours (6 AM to 6 PM)
            daylight_df = weather.filter_hourly_by_time(hourly_df, 6, 18)
            print(f"\nDaylight hours data (6 AM - 6 PM): {len(daylight_df)} records")
            
            # Get comprehensive summary
            full_summary = weather.get_hourly_summary(hourly_df)
            daylight_summary = weather.get_hourly_summary(daylight_df)
            
            print(f"\n📊 Full Day Analysis:")
            temp_stats = full_summary.get('temperature', {})
            print(f"- Temperature: {temp_stats.get('min', 'N/A')}° to {temp_stats.get('max', 'N/A')}° (avg: {temp_stats.get('avg', 'N/A')}°)")
            print(f"- Humidity: {full_summary.get('humidity', {}).get('min', 'N/A')}% to {full_summary.get('humidity', {}).get('max', 'N/A')}%")
            print(f"- Wind: avg {full_summary.get('wind_speed', {}).get('avg', 'N/A')} mph")
            
            print(f"\n🌅 Daylight Hours Analysis:")
            daylight_temp = daylight_summary.get('temperature', {})
            print(f"- Temperature: {daylight_temp.get('min', 'N/A')}° to {daylight_temp.get('max', 'N/A')}° (avg: {daylight_temp.get('avg', 'N/A')}°)")
            print(f"- Most common condition: {daylight_summary.get('most_common_condition', 'N/A')}")
            
            # Daily temperature patterns
            if 'temp' in hourly_df.columns and 'datetime' in hourly_df.columns:
                print(f"\n📈 Temperature Patterns by Hour:")
                hourly_df['hour'] = hourly_df['datetime'].str.split(':').str[0].astype(int)
                hourly_avg = hourly_df.groupby('hour')['temp'].mean().round(1)
                
                for hour in [6, 9, 12, 15, 18, 21]:
                    if hour in hourly_avg.index:
                        print(f"- {hour:02d}:00 - Average: {hourly_avg[hour]}°")

def example_combined_analysis():
    """Example showing combined daily and hourly analysis"""
    print("\n=== Combined Daily & Hourly Analysis ===")
    
    weather = VisualCrossingWeatherAPI()
    
    # Get historical data with hourly information
    historical_data = weather.get_historical_data(days_back=5, include_hourly=True)
    
    if historical_data:
        # Get combined data
        combined_data = weather.convert_to_dataframe(historical_data, 'combined')
        
        if isinstance(combined_data, dict) and 'daily' in combined_data and 'hourly' in combined_data:
            daily_df = combined_data['daily']
            hourly_df = combined_data['hourly']
            
            print(f"Daily records: {len(daily_df)}")
            print(f"Hourly records: {len(hourly_df)}")
            
            # Compare daily vs hourly temperature ranges
            print(f"\n🌡️  Temperature Comparison:")
            for _, day in daily_df.iterrows():
                date = day['datetime']
                daily_max = day.get('tempmax', 'N/A')
                daily_min = day.get('tempmin', 'N/A')
                
                # Get hourly data for this date
                day_hourly = hourly_df[hourly_df['date'] == date]
                if not day_hourly.empty and 'temp' in day_hourly.columns:
                    hourly_max = day_hourly['temp'].max()
                    hourly_min = day_hourly['temp'].min()
                    hourly_avg = day_hourly['temp'].mean().round(1)
                    
                    print(f"{date}:")
                    print(f"  Daily API: {daily_min}° - {daily_max}°")
                    print(f"  Hourly: {hourly_min}° - {hourly_max}° (avg: {hourly_avg}°)")

def example_custom_hourly_config():
    """Example showing custom hourly configuration"""
    print("\n=== Custom Hourly Configuration ===")
    
    # Create custom config for hourly analysis
    custom_config = {
        'location': {'primary': 'San Francisco, CA'},
        'historical': {
            'days_back': 2,
            'include_hourly': True,
            'hourly': {
                'enabled': True,
                'max_days_hourly': 7
            }
        },
        'data_options': {'units': 'metric'}
    }
    
    # Save temporary config
    with open('temp_hourly_config.yaml', 'w') as f:
        yaml.dump(custom_config, f)
    
    try:
        # Use custom config
        weather = VisualCrossingWeatherAPI(config_file='temp_hourly_config.yaml')
        print(f"Using custom config for: {weather.get_primary_location()}")
        
        # Get hourly data
        hourly_data = weather.get_hourly_historical_data()
        if hourly_data:
            hourly_df = weather.convert_to_dataframe(hourly_data, 'hourly')
            print(f"Retrieved {len(hourly_df)} hourly records")
            
            # Show temperature trend
            if not hourly_df.empty:
                summary = weather.get_hourly_summary(hourly_df)
                print(f"Temperature trend: {summary.get('temperature', {})}")
    
    finally:
        # Clean up temporary config
        if os.path.exists('temp_hourly_config.yaml'):
            os.remove('temp_hourly_config.yaml')

if __name__ == "__main__":
    print("Visual Crossing Weather API Client with Hourly Historical Data")
    print("=" * 65)
    print("Make sure to:")
    print("1. Set VISUAL_CROSSING_API_KEY in your .env file")
    print("2. Create config.yaml file with your settings")
    print("3. Install required packages: pip install pyyaml requests pandas python-dotenv")
    print("4. Set 'include_hourly: true' in config.yaml for hourly data")
    print("\nGet your free API key at: https://www.visualcrossing.com/weather-api")
    print("-" * 65)
    
    # Run examples using configuration
    try:
        example_with_config()
        # example_hourly_analysis()
        # example_combined_analysis()
        # example_custom_hourly_config()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure your .env and config.yaml files are properly configured.")