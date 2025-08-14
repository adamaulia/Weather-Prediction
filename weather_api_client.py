import requests
import json
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VisualCrossingWeatherAPI:
    """
    A client for the Visual Crossing Weather API to fetch current and historical weather data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the weather API client.
        
        Args:
            api_key (str, optional): Your Visual Crossing Weather API key.
                                   If not provided, will try to load from VISUAL_CROSSING_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('VISUAL_CROSSING_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Either pass it as a parameter or set VISUAL_CROSSING_API_KEY environment variable."
            )
        
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    def get_current_weather(self, location: str, include_forecast: bool = False) -> Dict[Any, Any]:
        """
        Get current weather data for a location.
        
        Args:
            location (str): Location (city, address, or coordinates)
            include_forecast (bool): Whether to include forecast data
            
        Returns:
            dict: Weather data response
        """
        url = f"{self.base_url}/{location}"
        
        params = {
            'key': self.api_key,
            'include': 'current',
            'contentType': 'json'
        }
        
        if include_forecast:
            params['include'] = 'days,current'
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return {}
    
    def get_historical_data(self, location: str, start_date: str, end_date: str) -> Dict[Any, Any]:
        """
        Get historical weather data for a location and date range.
        
        Args:
            location (str): Location (city, address, or coordinates)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Historical weather data response
        """
        url = f"{self.base_url}/{location}/{start_date}/{end_date}"
        
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
            print(f"Error fetching historical data: {e}")
            return {}
    
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
                    for hour in day['hours']:
                        hour['date'] = day['datetime']
                        hourly_data.append(hour)
            return pd.json_normalize(hourly_data)
        elif data_type == 'current':
            if 'currentConditions' in weather_data:
                return pd.json_normalize([weather_data['currentConditions']])
        
        return pd.DataFrame()

# Example usage functions
def example_current_weather(location: str = "New York, NY"):
    """
    Example of getting current weather
    
    Args:
        location (str): Location to get weather for (default: "New York, NY")
    """
    # Initialize the weather client (API key loaded from .env file)
    weather = VisualCrossingWeatherAPI()
    
    print(f"Getting current weather for: {location}")
    current_data = weather.get_current_weather(location, include_forecast=True)
    
    if current_data:
        print("Current Weather Data:")
        print(json.dumps(current_data, indent=2))
        
        # Convert to DataFrame
        df_current = weather.convert_to_dataframe(current_data, 'current')
        print("\nCurrent conditions as DataFrame:")
        print(df_current)

def example_historical_data(location: str = "London, UK", days_back: int = 7):
    """
    Example of getting historical weather data
    
    Args:
        location (str): Location to get weather for (default: "London, UK")
        days_back (int): Number of days back to fetch data (default: 7)
    """
    # Initialize the weather client (API key loaded from .env file)
    weather = VisualCrossingWeatherAPI()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    print(f"Getting historical weather for: {location} from {start_date} to {end_date}")
    historical_data = weather.get_historical_data(location, start_date, end_date)
    
    if historical_data:
        print("Historical Weather Data:")
        print(json.dumps(historical_data, indent=2))
        
        # Convert to DataFrame
        df_historical = weather.convert_to_dataframe(historical_data, 'daily')
        print("\nHistorical data as DataFrame:")
        print(df_historical[['datetime', 'temp', 'tempmax', 'tempmin', 'humidity', 'conditions']])

def example_hourly_data(location: str = "Chicago, IL", days_back: int = 1):
    """
    Example of getting hourly weather data
    
    Args:
        location (str): Location to get weather for (default: "Chicago, IL")
        days_back (int): Number of days back to fetch hourly data (default: 1)
    """
    # Initialize the weather client (API key loaded from .env file)
    weather = VisualCrossingWeatherAPI()
    
    # Calculate the target date
    target_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    print(f"Getting hourly weather for: {location} on {target_date}")
    hourly_data = weather.get_hourly_data(location, target_date, target_date)
    
    if hourly_data:
        print("Hourly Weather Data:")
        
        # Convert to DataFrame
        df_hourly = weather.convert_to_dataframe(hourly_data, 'hourly')
        print("Hourly data as DataFrame:")
        print(df_hourly[['date', 'datetime', 'temp', 'humidity', 'conditions']])

def get_weather_interactive():
    """Interactive function to get weather data with user input"""
    print("=== Interactive Weather Data Fetcher ===")
    
    # Get location from user
    location = input("Enter location (city, address, or coordinates): ").strip()
    if not location:
        location = "New York, NY"  # Default fallback
        print(f"Using default location: {location}")
    
    # Get data type preference
    print("\nSelect data type:")
    print("1. Current weather")
    print("2. Historical data (past week)")
    print("3. Hourly data (yesterday)")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    try:
        if choice == "1":
            example_current_weather(location)
        elif choice == "2":
            days = input("How many days back? (default: 7): ").strip()
            days_back = int(days) if days.isdigit() else 7
            example_historical_data(location, days_back)
        elif choice == "3":
            days = input("How many days back? (default: 1): ").strip()
            days_back = int(days) if days.isdigit() else 1
            example_hourly_data(location, days_back)
        else:
            print("Invalid choice. Getting current weather...")
            example_current_weather(location)
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Main function demonstrating various usage patterns"""
    print("Visual Crossing Weather API Client - Examples")
    print("=" * 50)
    
    # Example 1: Current weather for different locations
    print("\n1. Current Weather Examples:")
    example_current_weather("Tokyo, Japan")
    
    # Example 2: Historical data for different locations and periods
    print("\n2. Historical Data Examples:")
    example_historical_data("Paris, France", 14)  # 2 weeks of data
    
    # Example 3: Hourly data
    print("\n3. Hourly Data Examples:")
    example_hourly_data("Sydney, Australia", 2)  # 2 days ago
    
    # Example 4: Interactive mode
    print("\n4. Interactive Mode:")
    # get_weather_interactive()  # Uncomment to enable interactive mode

if __name__ == "__main__":
    print("Visual Crossing Weather API Client ready!")
    print("Make sure to set VISUAL_CROSSING_API_KEY in your .env file.")
    print("You can get a free API key at: https://www.visualcrossing.com/weather-api")
    
    # Run examples with different locations
    main()
    
    # Or run individual examples with custom locations:
    # example_current_weather("Miami, FL")
    # example_historical_data("Berlin, Germany", 30)  # 30 days back
    # example_hourly_data("Los Angeles, CA", 1)  # 1 day back
    
    # Or run interactive mode:
    # get_weather_interactive()