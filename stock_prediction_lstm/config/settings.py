import os
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for the stock prediction system.
    """

    # API Configuration
    ALPHA_VANTAGE_API_KEY = None

    # Model Defaults
    DEFAULT_LOOKBACK = 20
    DEFAULT_PREDICTION_DAYS = 5
    DEFAULT_LSTM_UNITS = 100
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_SEQUENCE_LENGTH = 20
    DEFAULT_DROPOUT_RATE = 0.2

    # Data Defaults
    DEFAULT_PERIOD = "1y"
    DEFAULT_INTERVAL = "1d"

    # Feature Flags
    CACHE_ENABLED = True
    SENTIMENT_ANALYSIS_ENABLED = True

    # Directory Configuration (can be dynamically set)
    DATA_CACHE_DIR = None
    MODELS_DIR = None
    LOGS_DIR = None

    @classmethod
    def load_api_keys(cls):
        """
        Load API keys from the secure keys directory.

        Returns:
            bool: True if the API keys were loaded successfully, False otherwise.
        """
        try:
            # Try to import from the keys directory
            from .keys.api_keys import ALPHA_VANTAGE_API_KEY

            if ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE":
                logger.warning("API key not configured. Please update config/keys/api_keys.py")
                print("\n⚠️  API KEY CONFIGURATION REQUIRED")
                print("=" * 50)
                print("Please configure your Alpha Vantage API key:")
                print(
                    f"1. Navigate to: {os.path.abspath(os.path.join(os.path.dirname(__file__), 'keys'))}"
                )
                print("2. Edit api_keys.py")
                print("3. Replace 'YOUR_API_KEY_HERE' with your actual API key")
                print("4. Get a free key at: https://www.alphavantage.co/support/#api-key")
                print("=" * 50)
                return False

            cls.ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API_KEY
            logger.info("API keys loaded successfully")
            return True

        except ImportError:
            logger.error("API keys file not found. Please create config/keys/api_keys.py")
            print("\n❌ API KEYS FILE MISSING")
            print("=" * 50)
            keys_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "keys"))
            print(f"Please create your API keys file at: {keys_dir}/api_keys.py")
            print(f"You can copy the template: {keys_dir}/api_keys.example")
            print(
                "Get your free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key"
            )
            print("=" * 50)
            return False

        except Exception as e:
            logger.error(f"Error loading API keys: {str(e)}")
            return False


# Load API keys on import
Config.load_api_keys()
