from crewai_tools import BaseTool
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from docx.shared import Inches
import yfinance as yf

class StockVisualizationTool(BaseTool):
    name: str = "Stock Visualization Tool"
    description: str = "Creates various charts and visualizations for stock data analysis, including price trends, volume analysis, correlation matrices, and technical indicators."

    def _run(self, stock_symbol: str, start_date: str, end_date: str) -> dict:
        # Fetch stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        
        # Generate visualizations
        visualizations = {}
        
        # Price and Volume Trend
        visualizations['price_volume_trend'] = self._create_price_volume_trend(stock_data)
        
        # Moving Averages
        visualizations['moving_averages'] = self._create_moving_averages(stock_data)
        
        # Candlestick Chart
        visualizations['candlestick_chart'] = self._create_candlestick_chart(stock_data)
        
        # Correlation Matrix
        visualizations['correlation_matrix'] = self._create_correlation_matrix(stock_data)
        
        # Returns Distribution
        visualizations['returns_distribution'] = self._create_returns_distribution(stock_data)
        
        return visualizations

    def _create_price_volume_trend(self, data):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(data.index, data['Close'], label='Close Price')
        ax1.set_title('Price and Volume Trend')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        ax2.bar(data.index, data['Volume'], label='Volume')
        ax2.set_ylabel('Volume')
        ax2.legend()
        
        plt.tight_layout()
        return self._fig_to_img(fig)

    def _create_moving_averages(self, data):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], label='Close Price')
        ax.plot(data.index, data['Close'].rolling(window=50).mean(), label='50-day MA')
        ax.plot(data.index, data['Close'].rolling(window=200).mean(), label='200-day MA')
        ax.set_title('Moving Averages')
        ax.set_ylabel('Price')
        ax.legend()
        plt.tight_layout()
        return self._fig_to_img(fig)

    def _create_candlestick_chart(self, data):
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 6))
        data_ohlc = data[['Open', 'High', 'Low', 'Close']].reset_index()
        data_ohlc['Date'] = data_ohlc['Date'].map(mdates.date2num)
        candlestick_ohlc(ax, data_ohlc.values, width=0.6, colorup='g', colordown='r')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_title('Candlestick Chart')
        plt.tight_layout()
        return self._fig_to_img(fig)

    def _create_correlation_matrix(self, data):
        correlation_matrix = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        return self._fig_to_img(fig)

    def _create_returns_distribution(self, data):
        returns = data['Close'].pct_change().dropna()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(returns, kde=True, ax=ax)
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Returns')
        plt.tight_layout()
        return self._fig_to_img(fig)

    def _fig_to_img(self, fig):
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close(fig)
        return img_buffer

# You can add more visualization methods as needed

stock_visualization_tool = StockVisualizationTool()