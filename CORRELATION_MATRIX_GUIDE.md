# Feature Correlation Matrix - User Guide

## Overview

The **Feature Correlation Matrix** is a new visualization added to the stock analysis dashboard that shows the relationships between different financial and technical indicators. This heatmap helps identify which features tend to move together (positive correlation) or in opposite directions (negative correlation).

## What is a Correlation Matrix?

A correlation matrix displays the correlation coefficients between pairs of variables in a color-coded heatmap format. The correlation coefficient ranges from -1 to +1:

- **+1.0**: Perfect positive correlation (features move exactly together)
- **0.0**: No correlation (features are independent)
- **-1.0**: Perfect negative correlation (features move exactly opposite)

## Color Coding

The heatmap uses an intuitive color scheme:

- 🔴 **Red tones**: Negative correlations (-1.0 to -0.3)
- ⚪ **White**: Near-zero correlations (-0.3 to +0.3)
- 🔵 **Blue tones**: Positive correlations (+0.3 to +1.0)

## Features Analyzed

The correlation matrix includes various financial indicators:

### Price Data
- **Close Price**: The stock's closing price
- **Price Change**: Daily percentage price change
- **Momentum**: Price difference over 5-day period

### Volume Data
- **Volume**: Trading volume
- **Volume Change**: Daily percentage volume change
- **Price×Volume**: Price-volume trend indicator

### Technical Indicators
- **RSI (14)**: Relative Strength Index
- **MACD**: Moving Average Convergence Divergence
- **MACD Signal**: MACD signal line
- **7-day MA / 30-day MA**: Moving averages
- **Bollinger Bands**: Upper, middle, and lower bands
- **Volatility**: Price volatility measure

### Sentiment Data
- **Positive Sentiment**: Bullish sentiment score
- **Negative Sentiment**: Bearish sentiment score
- **Avg Pos/Neg Sentiment**: 5-day moving averages

### Derived Ratios
- **High/Low Ratio**: Daily high vs low price ratio
- **Open/Close Ratio**: Opening vs closing price ratio

## How to Read the Matrix

1. **Diagonal Values**: Always 1.0 (perfect correlation with itself)
2. **Symmetric Matrix**: The value at (A,B) equals the value at (B,A)
3. **Strong Relationships**: Look for values above 0.7 or below -0.7
4. **Moderate Relationships**: Values between 0.3-0.7 or -0.3 to -0.7
5. **Weak Relationships**: Values between -0.3 and 0.3

## Investment Insights

### Positive Correlations to Watch
- **Price & Volume**: High correlation suggests strong momentum
- **Technical Indicators**: RSI, MACD moving together indicates trend confirmation
- **Sentiment & Price**: When positive sentiment correlates with price increases

### Negative Correlations to Watch
- **Price & Volatility**: Inverse relationship can indicate stability
- **Positive vs Negative Sentiment**: Natural opposition in market psychology
- **RSI & Price Change**: May indicate overbought/oversold conditions

## Practical Applications

### For Traders
- Identify leading indicators that predict price movements
- Spot divergences between correlated indicators
- Understand which technical signals to trust together

### For Risk Management
- Diversify by choosing uncorrelated indicators
- Identify when multiple signals confirm each other
- Understand interdependencies in your analysis

### For Strategy Development
- Build models using uncorrelated features to avoid redundancy
- Weight indicators based on their correlation with price
- Identify the most predictive feature combinations

## Example Interpretations

### Strong Positive Correlation (>0.7)
"Volume and Volume Change show 0.76 correlation"
→ *When trading volume spikes, it tends to stay elevated*

### Strong Negative Correlation (<-0.7)
"RSI and Price Change show -0.82 correlation"
→ *High RSI often coincides with price declines (overbought condition)*

### Weak Correlation (~0.0)
"Sentiment and Volume show 0.05 correlation"
→ *News sentiment doesn't directly drive trading volume*

## Tips for Analysis

1. **Look for Clusters**: Groups of highly correlated features
2. **Find Outliers**: Features that correlate differently than expected
3. **Check Price Relationships**: Which features best predict price movements
4. **Monitor Changes**: How correlations shift over different time periods
5. **Validate Intuition**: Confirm your market understanding with data

## Integration with Other Charts

Use the correlation matrix alongside:
- **Technical Indicators Chart**: Validate signal relationships
- **Price History**: Understand which factors drove historical movements
- **Future Predictions**: See which features the model weights most heavily

## Best Practices

- Focus on correlations with the target variable (Close Price)
- Look for unexpected relationships that provide new insights
- Consider time lag effects (some correlations may be delayed)
- Combine correlation analysis with fundamental analysis
- Remember: correlation doesn't imply causation

---

*This correlation matrix enhances your stock analysis by revealing the hidden relationships between different market factors, helping you make more informed investment decisions.*
