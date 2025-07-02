import pandas as pd
import numpy as np

class TechnicalIndicatorGenerator:
    """
    Calculates technical indicators from price and volume data
    """
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Add common technical indicators to the dataframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with price and volume data
        price_col : str
            Column name for price data (default: 'close')
        volume_col : str
            Column name for volume data (default: 'volume')
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added technical indicators
        """
        df = df.copy()
        
        df['ma7'] = df[price_col].rolling(window=7).mean()
        df['ma14'] = df[price_col].rolling(window=14).mean()
        df['ma30'] = df[price_col].rolling(window=30).mean()
        
        df['roc5'] = df[price_col].pct_change(periods=5) * 100
        
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        df['ema12'] = df[price_col].ewm(span=12, adjust=False).mean()
        df['ema26'] = df[price_col].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        if volume_col in df.columns:
            df['volume_ma7'] = df[volume_col].rolling(window=7).mean()
            
            obv = [0]
            for i in range(1, len(df)):
                if df[price_col].iloc[i] > df[price_col].iloc[i-1]:
                    obv.append(obv[-1] + df[volume_col].iloc[i])
                elif df[price_col].iloc[i] < df[price_col].iloc[i-1]:
                    obv.append(obv[-1] - df[volume_col].iloc[i])
                else:
                    obv.append(obv[-1])
            df['obv'] = obv
        
        return df
