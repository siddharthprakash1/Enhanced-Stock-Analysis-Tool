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
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.style import WD_STYLE_TYPE
from tools import stock_visualization_tool 
from chat_groq_manager import ChatGroqManager

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Set environment variables for Ollama
"""
os.environ["OPENAI_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_MODEL_NAME"] = "llama3"
os.environ["OPENAI_API_KEY"] = ""  # No API Key required for Ollama

# Initialize Llama 3 model
llm = ChatOllama(model="llama3")
"""
groq_manager = ChatGroqManager()
llm = groq_manager.create_llm()

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
    goal='Collect and synthesize comprehensive financial and market data for the given stock',
    backstory='You are a seasoned financial data scientist with decades of experience in gathering and preprocessing financial data from various sources. Your expertise spans across technical indicators, fundamental ratios, and market sentiment data. You have a keen eye for identifying relevant data points that can provide unique insights into a stock\'s performance and potential.',
    allow_delegation=False,
    llm=llm
)

analyst = Agent(
    role='Senior Financial Analyst',
    goal='Perform in-depth analysis of financial data and identify complex market trends and patterns',
    backstory='You are a highly regarded financial analyst with a track record of accurately predicting market movements. Your analysis combines technical, fundamental, and sentiment factors to provide a holistic view of a stock\'s potential. You have a unique ability to synthesize disparate data points into coherent and actionable insights.',
    allow_delegation=False,
    llm=llm
)

report_generator = Agent(
    role='Executive Research Report Generator',
    goal='Create comprehensive, actionable, and visually appealing research reports for high-level decision makers',
    backstory='You are an expert in distilling complex financial analyses into clear, concise, and impactful reports. Your reports have guided investment decisions at the highest levels of financial institutions. You have a talent for presenting data in a visually compelling manner and structuring information for maximum clarity and impact.',
    allow_delegation=False,
    llm=llm
)

risk_assessor = Agent(
    role='Risk Assessment Specialist',
    goal='Evaluate and quantify the risk associated with investment opportunities, providing a nuanced understanding of potential downsides',
    backstory='You are a risk management expert with a deep understanding of financial markets and statistical modeling. Your risk assessments have helped major institutions navigate volatile market conditions. You have a particular talent for identifying hidden risks and providing context-specific risk mitigation strategies.',
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
        description=f"Collect and synthesize comprehensive stock data for {symbol} from {start_date} to {end_date}. Include technical indicators, fundamental ratios, volatility measures, and sentiment analysis. Identify any unique or standout data points that could provide special insights.",
        agent=data_collector,
        expected_output="A detailed summary of the collected stock data including all key metrics, indicators, and any noteworthy data anomalies or patterns."
    )

    # Task 2: Perform In-depth Analysis
    analyze_data_task = Task(
        description=f"Conduct a thorough analysis of the following financial data and provide comprehensive insights:\n{data_summary}\n"
                    "Provide: 1) Technical analysis including price trends, indicator signals, and chart patterns, with emphasis on unique formations or divergences. "
                    "2) Fundamental analysis based on key ratios, comparing to industry averages and historical trends. "
                    "3) Detailed volatility and risk assessment, including potential scenarios. "
                    "4) Comparative performance analysis against sector and market benchmarks. "
                    "5) Sentiment analysis interpretation, including potential impact on short and long-term price movements. "
                    "Synthesize these factors to form an overall market outlook and provide specific, actionable insights.",
        agent=analyst,
        expected_output="A comprehensive, nuanced analysis of the stock data with deep insights on multiple facets of stock performance, market positioning, and potential future scenarios."
    )

    # Task 3: Assess Risk
    assess_risk_task = Task(
        description=f"Based on the provided data and analysis for {symbol}, conduct a thorough and nuanced risk assessment. "
                    "Consider: 1) Historical and implied volatility, including potential black swan events. "
                    "2) Beta and market correlation, analyzing in different market conditions. "
                    "3) Fundamental risk factors, including industry-specific risks. "
                    "4) Technical risk signals, with emphasis on potential breakdown points. "
                    "5) Market sentiment risks, including potential shifts in investor perception. "
                    "Provide a quantitative risk score, qualitative risk analysis, and specific risk mitigation strategies.",
        agent=risk_assessor,
        expected_output="A detailed risk assessment report including quantitative metrics, qualitative analysis of potential risk factors, and actionable risk management strategies."
    )

    # Task 4: Generate Comprehensive Report
    generate_report_task = Task(
        description="Generate a comprehensive, visually appealing, and actionable research report based on the data summary, in-depth analysis, and risk assessment. "
                    "Include: 1) Executive summary with key takeaways. "
                    "2) Technical analysis highlights with clear visual representations. "
                    "3) Fundamental analysis insights, benchmarked against industry standards. "
                    "4) Detailed risk assessment summary with scenario analysis. "
                    "5) Market positioning and sentiment analysis, including potential catalysts. "
                    "6) Actionable recommendations with clear justifications. "
                    "7) Appendix with detailed data and methodologies. "
                    "Use the Stock Visualization Tool to create relevant charts and graphs to support your analysis. "
                    "Ensure the report is professional, data-driven, visually engaging, and provides clear, actionable insights for decision-making. "
                    "Structure the report in a logical flow, using headings, subheadings, bullet points, and data visualizations for clarity and impact.",
        agent=report_generator,
        expected_output="A professional, comprehensive, visually appealing, and actionable research report on the analyzed stock, suitable for high-level decision makers, formatted as a well-structured document with integrated visualizations."
    )


    # Create and run the crew
    crew = Crew(
        agents=[data_collector, analyst, risk_assessor, report_generator],
        tasks=[collect_data_task, analyze_data_task, assess_risk_task, generate_report_task],
        verbose=True
    )

    result = crew.kickoff()
    print("CrewOutput type:", type(result))
    print("CrewOutput attributes:", dir(result))
    
    # Try to access the content in different ways
    if hasattr(result, 'final_output'):
        return result.final_output
    elif isinstance(result, dict) and 'final_output' in result:
        return result['final_output']
    elif isinstance(result, str):
        return result
    else:
        # If we can't find the output in an expected format, return the string representation
        return str(result)
import base64
from io import BytesIO

def create_markdown_report(report_content, symbol, start_date, end_date):
    # Generate visualizations
    visualizations = stock_visualization_tool._run(symbol, start_date, end_date)
    
    # Start building the markdown content
    markdown_content = f"# Stock Analysis Report: {symbol}\n\n"
    markdown_content += f"Analysis period: {start_date} to {end_date}\n\n"
    
    # Add content
    if isinstance(report_content, str):
        sections = report_content.split('\n\n')
    else:
        print(f"Unexpected report content type: {type(report_content)}")
        sections = str(report_content).split('\n\n')  # Convert to string if not already
    
    for section in sections:
        if section.strip():  # Only process non-empty sections
            if section.strip().isupper():
                markdown_content += f"## {section.strip()}\n\n"
            elif section.strip().startswith('#'):
                markdown_content += f"### {section.strip()[2:]}\n\n"
            else:
                markdown_content += f"{section.strip()}\n\n"
            
            # Add relevant visualizations after each section
            if "TECHNICAL ANALYSIS" in section.upper():
                markdown_content += embed_image(visualizations['price_volume_trend'], "Price and Volume Trend")
                markdown_content += embed_image(visualizations['moving_averages'], "Moving Averages")
                markdown_content += embed_image(visualizations['candlestick_chart'], "Candlestick Chart")
            elif "FUNDAMENTAL ANALYSIS" in section.upper():
                markdown_content += embed_image(visualizations['correlation_matrix'], "Correlation Matrix")
            elif "RISK ASSESSMENT" in section.upper():
                markdown_content += embed_image(visualizations['returns_distribution'], "Returns Distribution")
    
    # Save the markdown content to a file
    with open(f"{symbol}_Stock_Analysis_Report.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Report generated and saved as {symbol}_Stock_Analysis_Report.md")

def embed_image(img_buffer, title):
    """Convert image buffer to base64 and embed in markdown."""
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return f"![{title}](data:image/png;base64,{img_str})\n\n"

# Example usage
# Example usage
if __name__ == "__main__":
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # One year of data
    
    print("Starting enhanced research assistant...")
    report_content = run_enhanced_research_assistant(symbol, start_date, end_date)
    print("Enhanced research assistant completed.")
    
    print("Generating markdown report...")
    create_markdown_report(report_content, symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    print("Script completed successfully.")