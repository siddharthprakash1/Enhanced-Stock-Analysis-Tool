import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew
from langchain_ollama import ChatOllama
import os
from sklearn.linear_model import LinearRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Set environment variables for Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_MODEL_NAME"] = "llama3"
os.environ["OPENAI_API_KEY"] = ""  # No API Key required for Ollama

# Initialize Llama 2 model
llm = ChatOllama(model="llama3")

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    return stock.history(start=start_date, end=end_date)

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    # Existing Moving Averages
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
    
    # Volume Analysis
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
    
    return data

# Function to calculate fundamental ratios
def calculate_fundamental_ratios(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    
    pe_ratio = info.get('trailingPE', None)
    pb_ratio = info.get('priceToBook', None)
    debt_to_equity = info.get('debtToEquity', None)
    roe = info.get('returnOnEquity', None)
    eps_growth = info.get('earningsQuarterlyGrowth', None)
    
    return {
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'Debt to Equity': debt_to_equity,
        'ROE': roe,
        'EPS Growth': eps_growth
    }

# Function to identify support and resistance levels
def identify_support_resistance(data):
    pivot_high = data['High'].rolling(window=5, center=True).max()
    pivot_low = data['Low'].rolling(window=5, center=True).min()
    
    resistance_levels = pivot_high[pivot_high == data['High']]
    support_levels = pivot_low[pivot_low == data['Low']]
    
    return support_levels, resistance_levels

# Function to calculate volatility measures
def calculate_volatility(data):
    data['Returns'] = data['Close'].pct_change()
    historical_volatility = data['Returns'].std() * np.sqrt(252)  # Annualized
    
    data['ATR'] = pd.DataFrame({
        'H-L': data['High'] - data['Low'],
        'H-PC': abs(data['High'] - data['Close'].shift(1)),
        'L-PC': abs(data['Low'] - data['Close'].shift(1))
    }).max(axis=1)
    
    return historical_volatility, data['ATR']

# Function to perform comparative analysis
def comparative_analysis(symbol, benchmark_symbol='SPY'):
    stock_data = yf.download(symbol, start="2020-01-01")
    benchmark_data = yf.download(benchmark_symbol, start="2020-01-01")
    
    stock_returns = stock_data['Close'].pct_change()
    benchmark_returns = benchmark_data['Close'].pct_change()
    
    beta = stock_returns.cov(benchmark_returns) / benchmark_returns.var()
    
    return beta

# Function to perform sentiment analysis
def sentiment_analysis(symbol):
    news = yf.Ticker(symbol).news
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(article['title'])['compound'] for article in news]
    return np.mean(sentiments)

# Enhanced CrewAI Agents
data_collector = Agent(
    role='Advanced Data Collector',
    goal='Collect comprehensive financial and market data for a given stock',
    backstory='You are a seasoned financial data scientist with decades of experience in gathering and preprocessing financial data from various sources. Your expertise spans across technical indicators, fundamental ratios, and market sentiment data.',
    allow_delegation=False,
    llm=llm
)

analyst = Agent(
    role='Senior Financial Analyst',
    goal='Perform in-depth analysis of financial data and identify complex market trends',
    backstory='You are a highly regarded financial analyst with a track record of accurately predicting market movements. Your analysis combines technical, fundamental, and sentiment factors to provide a holistic view of a stock\'s potential.',
    allow_delegation=False,
    llm=llm
)

report_generator = Agent(
    role='Executive Research Report Generator',
    goal='Create comprehensive and actionable research reports for high-level decision makers',
    backstory='You are an expert in distilling complex financial analyses into clear, concise, and impactful reports. Your reports have guided investment decisions at the highest levels of financial institutions.',
    allow_delegation=False,
    llm=llm
)

risk_assessor = Agent(
    role='Risk Assessment Specialist',
    goal='Evaluate and quantify the risk associated with investment opportunities',
    backstory='You are a risk management expert with a deep understanding of financial markets and statistical modeling. Your risk assessments have helped major institutions navigate volatile market conditions.',
    allow_delegation=False,
    llm=llm
)

# Main function to run the enhanced research assistant
def run_enhanced_research_assistant(symbol, start_date, end_date):
    # Fetch and preprocess data
    data = fetch_stock_data(symbol, start_date, end_date)
    data = calculate_technical_indicators(data)
    
    # Calculate additional metrics
    fundamental_ratios = calculate_fundamental_ratios(symbol)
    support_levels, resistance_levels = identify_support_resistance(data)
    historical_volatility, atr = calculate_volatility(data)
    beta = comparative_analysis(symbol)
    sentiment_score = sentiment_analysis(symbol)
    
    # Prepare comprehensive data summary
    latest_price = data['Close'].iloc[-1]
    data_summary = f"""
    Symbol: {symbol}
    Latest closing price: ${latest_price:.2f}
    50-day SMA: ${data['SMA50'].iloc[-1]:.2f}
    200-day SMA: ${data['SMA200'].iloc[-1]:.2f}
    RSI: {data['RSI'].iloc[-1]:.2f}
    MACD: {data['MACD'].iloc[-1]:.2f}
    Signal Line: {data['Signal_Line'].iloc[-1]:.2f}
    Bollinger Bands: Upper ${data['BB_Upper'].iloc[-1]:.2f}, Middle ${data['BB_Middle'].iloc[-1]:.2f}, Lower ${data['BB_Lower'].iloc[-1]:.2f}
    Volume: {data['Volume'].iloc[-1]}, 20-day Avg: {data['Volume_MA'].iloc[-1]:.0f}
    
    Fundamental Ratios:
    P/E Ratio: {fundamental_ratios['P/E Ratio']}
    P/B Ratio: {fundamental_ratios['P/B Ratio']}
    Debt to Equity: {fundamental_ratios['Debt to Equity']}
    ROE: {fundamental_ratios['ROE']}
    EPS Growth: {fundamental_ratios['EPS Growth']}
    
    Recent Support Level: ${support_levels.iloc[-1]:.2f}
    Recent Resistance Level: ${resistance_levels.iloc[-1]:.2f}
    
    Historical Volatility: {historical_volatility:.2f}
    Average True Range: {atr.iloc[-1]:.2f}
    Beta: {beta:.2f}
    Sentiment Score: {sentiment_score:.2f}
    
    Data period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}
    """

    # Task 1: Collect and Summarize Data
    collect_data_task = Task(
        description=f"Collect and summarize comprehensive stock data for {symbol} from {start_date} to {end_date}. Include technical indicators, fundamental ratios, volatility measures, and sentiment analysis.",
        agent=data_collector,
        expected_output="A detailed summary of the collected stock data including all key metrics and indicators."
    )

    # Task 2: Perform In-depth Analysis
    analyze_data_task = Task(
        description=f"Analyze the following financial data and provide comprehensive insights:\n{data_summary}\n"
                    "Provide: 1) Technical analysis including price trends, indicator signals, and chart patterns, "
                    "2) Fundamental analysis based on key ratios, 3) Volatility and risk assessment, "
                    "4) Comparative performance analysis, 5) Sentiment analysis interpretation. "
                    "Synthesize these factors to form an overall market outlook.",
        agent=analyst,
        expected_output="A comprehensive analysis of the stock data with insights on multiple facets of stock performance and market positioning."
    )

    # Task 3: Assess Risk
    assess_risk_task = Task(
        description=f"Based on the provided data and analysis for {symbol}, conduct a thorough risk assessment. "
                    "Consider: 1) Historical and implied volatility, 2) Beta and market correlation, "
                    "3) Fundamental risk factors, 4) Technical risk signals, 5) Market sentiment risks. "
                    "Provide a quantitative risk score and qualitative risk analysis.",
        agent=risk_assessor,
        expected_output="A detailed risk assessment report including quantitative metrics and qualitative analysis of potential risk factors."
    )

    # Task 4: Generate Comprehensive Report
    generate_report_task = Task(
        description="Generate a comprehensive yet concise research report based on the data summary, in-depth analysis, and risk assessment. "
                    "Include: 1) Executive summary, 2) Technical analysis highlights, 3) Fundamental analysis insights, "
                    "4) Risk assessment summary, 5) Market positioning and sentiment, 6) Actionable recommendations. "
                    "Ensure the report is professional, data-driven, and provides clear insights for decision-making.",
        agent=report_generator,
        expected_output="A professional, comprehensive, and actionable research report on the analyzed stock, suitable for high-level decision makers."
    )

    # Create and run the crew
    crew = Crew(
        agents=[data_collector, analyst, risk_assessor, report_generator],
        tasks=[collect_data_task, analyze_data_task, assess_risk_task, generate_report_task],
        verbose=True
    )

    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    report = run_enhanced_research_assistant(symbol, start_date, end_date)
    print(report)
