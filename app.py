import streamlit as st
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Replace with your own News API key
NEWS_API_KEY = '5843e8b1715a4c1fb6628befb47ca1e8'


def get_stock_news(stock_symbol, num_articles=10):
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={NEWS_API_KEY}&sortBy=publishedAt&pageSize={num_articles}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']
    return articles


def analyze_sentiment(article_text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(article_text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def get_stock_data(stock_symbol, start_date, end_date):
    try:
        stock = yf.Ticker(stock_symbol)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data found for {stock_symbol} in the specified date range.")
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {stock_symbol}: {str(e)}")
        return None


def calculate_portfolio_performance(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_stddev


def minimize_portfolio_volatility(weights, returns, cov_matrix):
    return -calculate_portfolio_performance(weights, returns, cov_matrix)[0]


def predict_future_returns(returns, portfolio_weights, days=30):
    # Calculate mean daily return
    mean_return = returns.mean()

    # Calculate projected future returns
    future_returns = mean_return * portfolio_weights * days
    return future_returns


def main():
    st.title("Stocks News Sentiment Analysis with Portfolio Optimization and Future Returns Prediction")
    st.subheader("By Subir Singh, Oct 2023")

    # Sidebar for user input
    st.sidebar.header("User Input")

    stock_symbols_portfolio = st.sidebar.text_input("Enter stock symbols for portfolio optimization (separated by spaces)", "AAPL GOOGL MSFT").split()
    start_date_portfolio = st.sidebar.text_input("Enter start date for historical data (YYYY-MM-DD)", "2022-01-01")
    end_date_portfolio = st.sidebar.text_input("Enter end date for historical data (YYYY-MM-DD)", "2023-01-01")

    st.sidebar.header("Sentiment Analysis")
    stock_symbols_sentiment = st.sidebar.text_input("Enter stock symbols for sentiment analysis (separated by spaces)", "AAPL GOOGL MSFT").split()

    st.sidebar.header("Future Returns Prediction")
    days_to_predict = st.sidebar.number_input("Enter the number of days for future returns prediction", min_value=1, value=30)

    st.sidebar.header("Nifty Benchmark")
    nifty_symbol = st.sidebar.text_input("Enter Nifty 50 Benchmark Symbol", "^NSEI")

    # Initialize stock_data dictionary
    stock_data = {}

    # Portfolio Optimization
    st.subheader("Performing Portfolio Optimization...")
    for stock_symbol in stock_symbols_portfolio:
        stock_df = get_stock_data(stock_symbol, start_date_portfolio, end_date_portfolio)
        if stock_df is not None:
            stock_data[stock_symbol] = stock_df

    if not stock_data:
        st.warning("No stock data available for portfolio optimization.")
    else:
        # Rest of the code for Portfolio Optimization

    # Sentiment Analysis
    st.subheader("Performing Sentiment Analysis...")
    for stock_symbol in stock_symbols_sentiment:
        news_articles = get_stock_news(stock_symbol, num_articles=10)

        if not news_articles:
            st.warning(f"No news articles found for {stock_symbol}.")
            continue

        # Rest of the code for Sentiment Analysis

    # ... (rest of the code remains unchanged)


if __name__ == "__main__":
    main()
