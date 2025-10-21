import yaml
import os
from typing import Optional, Dict, Any, List


class YamlHelper:
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
    
    # def _get_default_config(self) -> Dict[Any, Any]:
    #     """Return default configuration"""
    #     return {
    #         'location': {'primary': 'New York, NY'},
    #         'historical': {'days_back': 7},
    #         'data_options': {'units': 'metric', 'include_current': True},
    #         'output': {'pretty_print': True},
    #         'processing': {'auto_dataframe': True, 'round_decimals': 2}
    #     }
    
    def get_values(self, key_path: str, default=None):
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